"""PG 连接池 + insight 入库（asyncpg，insight_writer 最小权限账号）。

密码走环境变量 INSIGHT_WRITER_PWD（不在代码里硬编码）。
"""
import asyncio
import asyncpg
import json
import os
from datetime import date as _date

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
    "frontend_and_backend",
    "data_structures_and_algorithms",
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

# 熔断阈值：单次 update 影响行数超过此值且未显式 confirm_large 时拒绝
UPDATE_MAX_AFFECTED = 5


def _build_where(where: dict) -> tuple[list, list]:
    """构造 WHERE 子句的参数和片段，$N 从 1 起编号。返回 (args, parts)。"""
    args: list = []
    parts: list = []

    def add(value, fmt: str) -> None:
        args.append(value)
        parts.append(fmt.format(len(args)))

    if where.get("id"):
        add(where["id"], "id = ${}")
    if where.get("ids"):
        add(where["ids"], "id = ANY(${})")
    if where.get("title_contains"):
        add(f"%{where['title_contains']}%", "title ILIKE ${}")
    if where.get("created_after"):
        add(_date.fromisoformat(where["created_after"]), "created_at >= ${}")
    if where.get("created_before"):
        add(_date.fromisoformat(where["created_before"]), "created_at < ${}")
    if where.get("domain"):
        add(where["domain"], "domain = ${}")
    if where.get("iv_grade"):
        add(where["iv_grade"], "iv_grade = ${}")
    return args, parts


async def list_insights_pg(
    created_after: str | None = None,
    created_before: str | None = None,
    domain: str | None = None,
    iv_grade: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """按条件列出 insight 摘要（纯读，不含 content 全文）。

    agent 获取信息 / 查入库历史 / 取 ID 应该用这个，不要用 update。
    """
    where_args, where_parts = _build_where(
        {
            "created_after": created_after,
            "created_before": created_before,
            "domain": domain,
            "iv_grade": iv_grade,
        }
    )
    limit = max(1, min(int(limit), 200))
    sql = (
        "SELECT id, title, summary, content_tokens, iv_grade, iv_desc, "
        "domain, tags, source_type, source_url, source_title, created_at, updated_at "
        "FROM shared.insight"
    )
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    where_args.append(limit)
    sql += f" ORDER BY created_at DESC LIMIT ${len(where_args)}"
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *where_args)
    return [dict(r) for r in rows]


async def update_insight_pg(
    where: dict,
    set_fields: dict,
    *,
    dry_run: bool = False,
    confirm_large: bool = False,
    max_affected: int = UPDATE_MAX_AFFECTED,
) -> dict:
    """批量更新 shared.insight（dry-run + affected_rows 熔断 + 审计触发器）。

    流程：先 SELECT 受影响 id -> 熔断/dry_run 判断 -> 按 id 精确 UPDATE
    （UPDATE 由 trg_insight_audit 自动把 OLD.* 写进 ops.insight_update_log）。

    Args:
        where: 条件（id/ids/title_contains/created_after/created_before/domain/iv_grade）
        set_fields: 更新字段（必须在 UPDATE_ALLOWED_FIELDS 白名单内）
        dry_run: True 只预览受影响行，不落库
        confirm_large: 影响行数 > max_affected 时必须显式传 True 才执行
        max_affected: 熔断阈值

    Returns:
        dry_run: {"dry_run": True, "count": N, "ids": [...]}
        熔断拒绝: {"error": "...", "count": N, "ids": [...]}
        成功: {"updated": N, "ids": [...]}
    """
    set_args: list = []
    set_parts: list = []
    for k, v in set_fields.items():
        if k not in UPDATE_ALLOWED_FIELDS:
            raise ValueError(f"不允许更新字段: {k}（允许: {UPDATE_ALLOWED_FIELDS}）")
        set_args.append(v)
        set_parts.append(f"{k} = ${len(set_args)}")
    if not set_parts:
        raise ValueError("没有可更新的字段")

    where_args, where_parts = _build_where(where)
    if not where_parts:
        raise ValueError("拒绝无条件更新：where 必须至少包含一个筛选条件")

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. 先取受影响 id（预览 + 熔断计数 + UPDATE 精确目标，快照一致）
            select_sql = (
                "SELECT id FROM shared.insight WHERE "
                + " AND ".join(where_parts)
                + " ORDER BY id"
            )
            rows = await conn.fetch(select_sql, *where_args)
            ids = [r["id"] for r in rows]
            count = len(ids)

            if count == 0:
                return {"updated": 0, "ids": []}

            # 2. 熔断
            if count > max_affected and not confirm_large:
                return {
                    "error": (
                        f"影响 {count} 行（> 熔断阈值 {max_affected}）。"
                        f"确认无误后传 confirm_large=true 重试，或缩小范围；"
                        f"dry_run=true 可先预览。"
                    ),
                    "count": count,
                    "ids": ids,
                }

            # 3. dry-run
            if dry_run:
                return {"dry_run": True, "count": count, "ids": ids}

            # 4. 按预览的 id 精确更新（审计触发器记 OLD.*）
            update_args = list(set_args) + [ids]
            sql = (
                f"UPDATE shared.insight SET {', '.join(set_parts)}, updated_at = now() "
                f"WHERE id = ANY(${len(set_args) + 1})"
            )
            result = await conn.execute(sql, *update_args)
    updated = int(result.split()[-1]) if result else 0
    return {"updated": updated, "ids": ids}
