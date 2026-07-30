"""PG 连接池 + query_trace CRUD（asyncpg，query_trace_writer 最小权限账号）。

密码走环境变量 QUERY_TRACE_WRITER_PWD（不在代码里硬编码）。
query_trace_writer 只能 INSERT/SELECT shared.query_trace（append-only 日志，无 UPDATE/DELETE）。

每次 /query 调用落一行；turn_id 把一次用户消息的 N 个并行/串行 /query 调用归组。
lightrag 在 /query 完成后 fire-and-forget POST 到 data_platform /query_trace/ 上报。
"""
import asyncio
import asyncpg
import json
import logging
import os

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_pool() -> asyncpg.Pool:
    """连接池（query_trace_writer 账号，INSERT/SELECT shared.query_trace）。

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
            user="query_trace_writer",
            password=os.environ["QUERY_TRACE_WRITER_PWD"],
            database=os.environ.get("DP_PG_DB", "rag"),
            min_size=2,
            max_size=5,
        )
        _pool_loop = loop
    return _pool


# insert 允许的字段（和表列一致，除 id/created_at 由 DB 生成；rewritten 是 GENERATED 列由 PG 算）
INSERT_COLUMNS = {
    "turn_id", "trace_id", "call_index", "original_query", "search_query",
    "mode", "query_params", "source", "endpoint",
    "rewrite_ms", "retrieval_ms", "rerank_ms", "generation_ms",
    "kg_embed_ms", "kg_entity_ms", "kg_relation_ms", "kg_chunk_ms",
    "chunks_retrieved", "chunks_after_rerank", "chunks_final",
    "entities_count", "relations_count",
    "cache_hit", "llm_model", "context_tokens", "response_tokens",
    "status", "failure_stage", "error_msg",
}


async def insert_query_trace_pg(payload: dict) -> int:
    """INSERT shared.query_trace，返回新 id。

    payload 取 INSERT_COLUMNS 内的字段；query_params(dict) 序列化为 jsonb。
    缺省字段走表默认（status='success', cache_hit=false, query_params='{}' 等）。
    """
    cols: list[str] = []
    vals: list = []
    for k in INSERT_COLUMNS:
        if k not in payload:
            continue
        v = payload[k]
        if k == "query_params":
            cols.append(k)
            vals.append(json.dumps(v if v is not None else {}, ensure_ascii=False))
        elif k == "cache_hit":
            cols.append(k)
            vals.append(bool(v))
        else:
            cols.append(k)
            vals.append(v)

    # 构建 $1,$2,... 占位；query_params 用 ::jsonb
    placeholders = []
    for i, c in enumerate(cols, start=1):
        placeholders.append(f"${i}::jsonb" if c == "query_params" else f"${i}")
    col_sql = ", ".join(cols)
    ph_sql = ", ".join(placeholders)

    pool = await get_pool()
    async with pool.acquire() as conn:
        tid = await conn.fetchval(
            f"INSERT INTO shared.query_trace ({col_sql}) VALUES ({ph_sql}) RETURNING id",
            *vals,
        )
    return tid


async def list_query_trace_pg(
    turn_id: str | None = None,
    source: str | None = None,
    mode: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """列表查询 shared.query_trace（按 created_at DESC）。可按 turn_id/source/mode 筛选。"""
    args: list = []
    where_parts: list[str] = []
    if turn_id:
        args.append(turn_id)
        where_parts.append(f"turn_id = ${len(args)}")
    if source:
        args.append(source)
        where_parts.append(f"source = ${len(args)}")
    if mode:
        args.append(mode)
        where_parts.append(f"mode = ${len(args)}")

    args.append(limit)
    limit_idx = len(args)
    args.append(offset)
    offset_idx = len(args)

    sql = "SELECT * FROM shared.query_trace"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += f" ORDER BY created_at DESC LIMIT ${limit_idx} OFFSET ${offset_idx}"

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def get_query_trace_pg(id: int) -> dict | None:
    """取单条 query_trace（全字段）。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM shared.query_trace WHERE id = $1", id)
        return dict(row) if row else None


async def list_by_turn_pg(turn_id: str) -> list[dict]:
    """取一次 turn 的全部 /query 调用（按 call_index 排序）。用于看多路并行/串行。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM shared.query_trace WHERE turn_id = $1 ORDER BY call_index NULLS LAST, created_at",
            turn_id,
        )
    return [dict(r) for r in rows]
