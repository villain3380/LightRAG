"""PG 连接池 + todo CRUD（asyncpg，todo_writer 最小权限账号）。

密码走环境变量 TODO_WRITER_PWD（不在代码里硬编码）。
todo_writer 只能 INSERT/SELECT/UPDATE shared.todo（无 DELETE；软删除用 status='cancelled'）。

updated_at / completed_at 由 trigger trg_todo_touch 自动管：
  - status='done' 时自动填 completed_at
  - status 从 done 改回非 done 时自动清 completed_at
应用层不用手动写这两个字段。
"""
import asyncio
import asyncpg
import json
import logging
import os
from datetime import date as _date

from .embedding import embed_todo
from .milvus_client import ensure_todo_collection, upsert_todo_vector, search_todo_vector

logger = logging.getLogger(__name__)


def _parse_date(s: str | None) -> _date | None:
    """'YYYY-MM-DD' 字符串 -> date 对象（asyncpg 的 date 列要求 date，不接受字符串）。"""
    return _date.fromisoformat(s) if s else None

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_pool() -> asyncpg.Pool:
    """连接池（todo_writer 账号，INSERT/SELECT/UPDATE shared.todo）。

    检查当前 event loop：变了就关旧 pool 重建（asyncpg pool 绑 loop，跨 loop 失效）。
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
            user="todo_writer",
            password=os.environ["TODO_WRITER_PWD"],
            database=os.environ.get("DP_PG_DB", "rag"),
            min_size=2,
            max_size=5,
        )
        _pool_loop = loop
    return _pool


# 白名单（和表 CHECK 一致）
VALID_STATUS = {"todo", "in_progress", "done", "cancelled"}
VALID_PRIORITY = {"P0", "P1", "P2", "P3"}


async def insert_todo_pg(
    title: str,
    detail: str | None = None,
    priority: str = "P2",
    due_date: str | None = None,
    tags: list[str] | None = None,
    domain: str | None = None,
    sort_order: int = 0,
    raw_meta: dict | None = None,
) -> int:
    """参数化 INSERT shared.todo，返回新 id。调用方负责白名单校验。

    status 强制为 'todo'（新建总是待办）；completed_at 由 trigger 在完成时填。
    due_date 传 'YYYY-MM-DD' 字符串。
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        tid = await conn.fetchval(
            """INSERT INTO shared.todo
                 (title, detail, status, priority, due_date, tags, domain, sort_order, raw_meta)
               VALUES ($1,$2,'todo',$3,$4,$5,$6,$7,$8::jsonb)
               RETURNING id""",
            title, detail, priority,
            _parse_date(due_date), tags or [], domain, sort_order,
            json.dumps(raw_meta or {}, ensure_ascii=False),
        )
    # 向量化 title+detail 写 Milvus（失败不回滚 PG，只 log；todo 仍可 CRUD，只是搜不到）
    try:
        ensure_todo_collection()
        dense = await embed_todo(f"{title} {detail or ''}".strip())
        upsert_todo_vector(tid, dense, "todo")
    except Exception as e:
        logger.warning(f"todo#{tid} 向量入库失败(不影响 PG): {e}")
    return tid


async def list_todo_pg(
    status: str | None = None,
    priority: str | None = None,
    domain: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """列表查询 shared.todo（含 detail）。按 priority(P0→P3) → due_date → created_at 排序。"""
    args: list = []
    where_parts: list = []
    if status:
        args.append(status)
        where_parts.append(f"status = ${len(args)}")
    if priority:
        args.append(priority)
        where_parts.append(f"priority = ${len(args)}")
    if domain:
        args.append(domain)
        where_parts.append(f"domain = ${len(args)}")

    args.append(limit)
    limit_idx = len(args)
    args.append(offset)
    offset_idx = len(args)

    sql = (
        "SELECT id, title, detail, status, priority, due_date, completed_at, "
        "tags, domain, sort_order, created_at, updated_at "
        "FROM shared.todo"
    )
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += (
        " ORDER BY array_position(ARRAY['P0','P1','P2','P3'], priority),"
        " due_date NULLS LAST, created_at DESC"
    )
    sql += f" LIMIT ${limit_idx} OFFSET ${offset_idx}"

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def get_todo_pg(id: int) -> dict | None:
    """取单条 todo（含 detail + raw_meta 全字段）。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, title, detail, status, priority, due_date, completed_at, "
            "tags, domain, sort_order, raw_meta, created_at, updated_at "
            "FROM shared.todo WHERE id = $1",
            id,
        )
        return dict(row) if row else None


# 白名单：允许更新的字段（created_at/updated_at/completed_at 由 trigger/默认管，不可改）
TODO_UPDATE_ALLOWED = {
    "title", "detail", "status", "priority", "due_date",
    "tags", "domain", "sort_order", "raw_meta",
}


async def update_todo_pg(id: int, set_fields: dict) -> dict:
    """更新单条 todo（白名单字段）。返回更新后的 {id, status, completed_at, updated_at}。

    - status='done' → trigger 自动填 completed_at
    - status 从 done 改回 → trigger 自动清 completed_at
    - due_date 传 'YYYY-MM-DD' 字符串或 null
    - tags 传 list 或 null
    - raw_meta 传 dict 或 null
    """
    args: list = []
    set_parts: list = []
    for k, v in set_fields.items():
        if k not in TODO_UPDATE_ALLOWED:
            raise ValueError(f"不允许更新字段: {k}（允许: {TODO_UPDATE_ALLOWED}）")
        if k == "raw_meta":
            args.append(json.dumps(v, ensure_ascii=False) if v is not None else None)
            set_parts.append(f"raw_meta = ${len(args)}::jsonb")
        elif k == "due_date":
            args.append(_parse_date(v) if v else None)
            set_parts.append(f"due_date = ${len(args)}")
        else:
            args.append(v)
            set_parts.append(f"{k} = ${len(args)}")
    if not set_parts:
        raise ValueError("没有可更新的字段")

    args.append(id)
    sql = (
        f"UPDATE shared.todo SET {', '.join(set_parts)} "
        f"WHERE id = ${len(args)} "
        f"RETURNING id, status, completed_at, updated_at"
    )
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *args)
    if not row:
        return {"error": "not found", "id": id}
    result = dict(row)
    # title/detail 变了 -> re-embed + upsert Milvus（status 也同步）
    if "title" in set_fields or "detail" in set_fields:
        try:
            full = await get_todo_pg(id)
            if full:
                dense = await embed_todo(f"{full['title']} {full.get('detail') or ''}".strip())
                upsert_todo_vector(id, dense, full["status"])
        except Exception as e:
            logger.warning(f"todo#{id} 向量更新失败: {e}")
    return result


async def todo_search_pg(query: str, top_k: int = 5) -> list[dict]:
    """语义搜索 todo（embed query -> Milvus dense ANN -> PG 取详情 + score）。

    返回 [{id, title, detail, status, priority, due_date, completed_at, tags,
           domain, created_at, updated_at, score}, ...]，按相似度降序。
    用于"我用模糊复述找某条 todo"的场景。
    """
    ensure_todo_collection()
    dense = await embed_todo(query)
    hits = search_todo_vector(dense, top_k)
    if not hits:
        return []
    ids = [h["id"] for h in hits]
    scores = {h["id"]: h["distance"] for h in hits}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, detail, status, priority, due_date, completed_at, "
            "tags, domain, created_at, updated_at FROM shared.todo WHERE id = ANY($1)",
            ids,
        )
    items = []
    for r in rows:
        d = dict(r)
        d["score"] = scores.get(d["id"])
        items.append(d)
    items.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    return items
