"""检索路径追踪的 PG 读写（public schema，lightrag-server 自己拥有）。

用 rag 账号（POSTGRES_* env）直连 PG，写 public.query_trace + public.query_trace_chunk，
并为 trace 页面提供读接口。chunk 正文不在本 store（在 lightrag text_chunks KV store，
由 router 用 rag.text_chunks.get_by_id 取）。

落库是 fire-and-forget：query_routes.py 在 /query 完成后调 insert_trace，失败只 log，
不影响查询本身。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_pool() -> asyncpg.Pool:
    """连接池（rag 账号，读写 public.query_trace*）。loop 变了就重建。"""
    global _pool, _pool_loop
    loop = asyncio.get_running_loop()
    if _pool is None or _pool_loop is not loop:
        if _pool is not None:
            try:
                await _pool.close()
            except Exception:
                pass
        _pool = await asyncpg.create_pool(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "rag"),
            password=os.environ.get("POSTGRES_PASSWORD", "rag"),
            database=os.environ.get("POSTGRES_DATABASE", "rag"),
            min_size=1,
            max_size=3,
        )
        _pool_loop = loop
    return _pool


# query_trace 列（和 to_ship_dict() 的 key 对齐，除 chunks）
QT_COLUMNS = [
    "turn_id", "trace_id", "call_index", "original_query", "search_query",
    "mode", "query_params", "source", "endpoint",
    "rewrite_ms", "retrieval_ms", "rerank_ms", "generation_ms",
    "kg_embed_ms", "kg_entity_ms", "kg_relation_ms", "kg_chunk_ms",
    "chunks_retrieved", "chunks_after_rerank", "chunks_final",
    "entities_count", "relations_count",
    "cache_hit", "llm_model", "context_tokens", "response_tokens",
    "status", "failure_stage", "error_msg",
]

# query_trace_chunk 列（和 to_chunk_ship_list() 的 key 对齐，除 trace_id 已含）
QTC_COLUMNS = [
    "trace_id", "chunk_id", "file_path", "sources",
    "dense_rank", "sparse_rank", "entity_rank", "relation_rank",
    "pre_rerank_position", "post_rerank_rank", "rerank_score", "in_final_context",
]


async def insert_trace(payload: dict, chunks: list[dict]) -> str | None:
    """写一条 query_trace + N 条 query_trace_chunk。返回 trace_id，失败返回 None。

    payload: QueryTraceTracker.to_ship_dict()；chunks: to_chunk_ship_list()。
    """
    trace_id = payload.get("trace_id")
    if not trace_id:
        return None
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. query_trace 行
            cols: list[str] = []
            vals: list[Any] = []
            for c in QT_COLUMNS:
                if c not in payload:
                    continue
                cols.append(c)
                v = payload[c]
                if c == "query_params":
                    v = json.dumps(v if v is not None else {}, ensure_ascii=False)
                elif c == "cache_hit":
                    v = bool(v)
                vals.append(v)
            placeholders = [
                f"${i}::jsonb" if cols[i - 1] == "query_params" else f"${i}"
                for i in range(1, len(cols) + 1)
            ]
            await conn.execute(
                f"INSERT INTO public.query_trace ({', '.join(cols)}) "
                f"VALUES ({', '.join(placeholders)})",
                *vals,
            )

            # 2. query_trace_chunk 行（批量）
            if chunks:
                rows: list[tuple] = []
                for ch in chunks:
                    row: list[Any] = []
                    for c in QTC_COLUMNS:
                        v = ch.get(c)
                        if c == "sources":
                            v = v if v is not None else []
                        elif c == "in_final_context":
                            v = bool(v)
                        row.append(v)
                    rows.append(tuple(row))
                # sources 是 TEXT[]，asyncpg 直接收 list
                col_sql = ", ".join(QTC_COLUMNS)
                ph_sql = ", ".join(f"${i+1}" for i in range(len(QTC_COLUMNS)))
                await conn.executemany(
                    f"INSERT INTO public.query_trace_chunk ({col_sql}) "
                    f"VALUES ({ph_sql})",
                    rows,
                )
    return trace_id


# ---- 读接口（trace 页面用）----


async def list_turns(limit: int = 50) -> list[dict]:
    """turn 列表，时间倒序。每个 turn 一行（original_query/时间/trace 数/chunk 数/来源）。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT t.turn_id,
                   MIN(t.original_query) AS original_query,
                   MAX(t.created_at) AS latest_created_at,
                   COUNT(*) AS trace_count,
                   SUM(CASE WHEN c.id IS NOT NULL THEN 1 ELSE 0 END) AS chunk_count,
                   MIN(t.source) AS source
            FROM public.query_trace t
            LEFT JOIN public.query_trace_chunk c ON c.trace_id = t.trace_id
            GROUP BY t.turn_id
            ORDER BY latest_created_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(r) for r in rows]


async def list_traces_by_turn(turn_id: str) -> list[dict]:
    """该 turn 下的 query_trace 行（每个 /query 调用），按 call_index 排。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM public.query_trace WHERE turn_id = $1 "
            "ORDER BY call_index NULLS LAST, created_at",
            turn_id,
        )
    return [dict(r) for r in rows]


async def list_chunks_by_trace(trace_id: str) -> list[dict]:
    """该 trace 的 chunk 明细行。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM public.query_trace_chunk WHERE trace_id = $1 "
            "ORDER BY post_rerank_rank NULLS LAST, pre_rerank_position NULLS LAST",
            trace_id,
        )
    return [dict(r) for r in rows]
