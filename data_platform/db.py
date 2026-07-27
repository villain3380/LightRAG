"""PG 连接池 + insight 入库（asyncpg，insight_writer 最小权限账号）。

密码走环境变量 INSIGHT_WRITER_PWD（不在代码里硬编码）。
"""
import asyncio
import asyncpg
import json
import os

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_pool() -> asyncpg.Pool:
    """连接池（insight_writer 账号，只能 INSERT/SELECT shared.insight）。

    检查当前 event loop：变了就关旧 pool 重建（asyncpg pool 绑 loop，跨 loop 失效）。
    真实 uvicorn 一个 loop，只建一次；TestClient 每请求新 loop，每次重建。
    """
    global _pool, _pool_loop
    loop = asyncio.get_running_loop()
    if _pool is None or _pool_loop is not loop:
        if _pool is not None:
            try:
                await _pool.close()
            except Exception:
                pass
        _pool = await asyncpg.create_pool(
            host=os.environ.get("DP_PG_HOST", "localhost"),
            port=int(os.environ.get("DP_PG_PORT", "5432")),
            user="insight_writer",
            password=os.environ["INSIGHT_WRITER_PWD"],
            database=os.environ.get("DP_PG_DB", "rag"),
            min_size=2,
            max_size=5,
        )
        _pool_loop = loop
    return _pool


# 白名单（和表 CHECK / SKILL.md 一致）
VALID_GRADES = {"S2", "S1", "A", "B", "C", "D"}
VALID_DOMAINS = {
    "financial_market",
    "financial_market/semiconductor",
    "financial_market/banking",
    "financial_market/real_estate",
    "technology",
    "policy",
    "daily_life",
    "other",
}


async def insert_insight_pg(
    title: str | None,
    summary: str | None,
    content: str,
    content_tokens: int | None,
    iv_grade: str,
    iv_desc: str | None,
    domain: str,
    tags: list[str] | None,
    source_type: str | None,
    source_url: str | None,
    source_title: str | None,
    raw_meta: dict | None,
    content_ref: str | None,
) -> int:
    """参数化 INSERT shared.insight，返回新 id。调用方负责白名单校验。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """INSERT INTO shared.insight
                 (title, summary, content, content_tokens, iv_grade, iv_desc, domain, tags,
                  source_type, source_url, source_title, raw_meta, content_ref)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13)
               RETURNING id""",
            title, summary, content, content_tokens, iv_grade, iv_desc, domain,
            tags or [], source_type, source_url, source_title,
            json.dumps(raw_meta or {}, ensure_ascii=False), content_ref,
        )


async def fetch_insight_brief(ids: list[int]) -> list[dict]:
    """按 id 列表取 insight 摘要信息（不含 content 全文，agent 检索返回用）。"""
    if not ids:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, title, summary, content_tokens, iv_grade, iv_desc,
                      domain, tags, source_type, source_url, source_title, content_ref
               FROM shared.insight WHERE id = ANY($1)""",
            ids,
        )
        return [dict(r) for r in rows]


async def fetch_insight_content(id: int) -> dict | None:
    """按需读原文（get_insight_content 用）。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, content FROM shared.insight WHERE id = $1", id
        )
        return dict(row) if row else None


# 白名单：允许批量更新的字段（content/content_ref/content_tokens 不允许改）
UPDATE_ALLOWED_FIELDS = {
    "source_title", "source_url", "source_type",
    "iv_grade", "iv_desc", "domain", "title", "summary",
}


async def update_insight_pg(where: dict, set_fields: dict) -> dict:
    """批量更新 shared.insight。返回 {updated: N}。

    Args:
        where: 条件（created_after/created_before YYYY-MM-DD, domain, iv_grade）
        set_fields: 更新字段（必须在 UPDATE_ALLOWED_FIELDS 白名单内）
    """
    args: list = []
    set_parts: list = []
    for k, v in set_fields.items():
        if k not in UPDATE_ALLOWED_FIELDS:
            raise ValueError(f"不允许更新字段: {k}（允许: {UPDATE_ALLOWED_FIELDS}）")
        args.append(v)
        set_parts.append(f"{k} = ${len(args)}")
    if not set_parts:
        raise ValueError("没有可更新的字段")

    where_parts: list = []
    if where.get("id"):
        args.append(where["id"])
        where_parts.append(f"id = ${len(args)}")
    if where.get("ids"):
        args.append(where["ids"])
        where_parts.append(f"id = ANY(${len(args)})")
    if where.get("title_contains"):
        args.append(f"%{where['title_contains']}%")
        where_parts.append(f"title ILIKE ${len(args)}")
    from datetime import date as _date
    if where.get("created_after"):
        args.append(_date.fromisoformat(where["created_after"]))
        where_parts.append(f"created_at >= ${len(args)}")
    if where.get("created_before"):
        args.append(_date.fromisoformat(where["created_before"]))
        where_parts.append(f"created_at < ${len(args)}")
    if where.get("domain"):
        args.append(where["domain"])
        where_parts.append(f"domain = ${len(args)}")
    if where.get("iv_grade"):
        args.append(where["iv_grade"])
        where_parts.append(f"iv_grade = ${len(args)}")

    sql = f"UPDATE shared.insight SET {', '.join(set_parts)}, updated_at = now()"
    if where_parts:
        sql += f" WHERE {' AND '.join(where_parts)}"

    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(sql, *args)
    count = int(result.split()[-1]) if result else 0
    return {"updated": count}
