"""Retrieval path trace viewer API (public.query_trace*).

读 public.query_trace + public.query_trace_chunk，按需取 chunk 正文（from
lightrag text_chunks KV store）。供 webui 的 trace 页面用：
  GET /query_trace/turns              turn 列表(时间倒序)
  GET /query_trace/turn/{turn_id}     该 turn 下的 trace 行
  GET /query_trace/chunk/{chunk_id}   按需取 chunk 正文
  GET /query_trace/{trace_id}/chunks  该 trace 的 chunk 明细行
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from lightrag.api.query_trace_store import (
    list_chunks_by_trace,
    list_traces_by_turn,
    list_turns,
)
from lightrag.utils import logger


def create_query_trace_routes(rag, api_key: Optional[str] = None):
    router = APIRouter(prefix="/query_trace", tags=["query_trace"])

    @router.get("/turns")
    async def turns(limit: int = 50):
        """turn 列表，时间倒序（每个 turn：original_query/时间/trace 数/chunk 数/来源）。"""
        try:
            return await list_turns(limit)
        except Exception as e:
            logger.error(f"list turns failed: {e}", exc_info=True)
            raise HTTPException(500, f"list turns failed: {e}")

    @router.get("/turn/{turn_id}")
    async def traces_by_turn(turn_id: str):
        """该 turn 下的 query_trace 行（每个 /query 调用），按 call_index 排。"""
        try:
            return await list_traces_by_turn(turn_id)
        except Exception as e:
            logger.error(f"list traces by turn failed: {e}", exc_info=True)
            raise HTTPException(500, f"list traces failed: {e}")

    @router.get("/chunk/{chunk_id}")
    async def chunk_content(chunk_id: str):
        """按需取 chunk 正文（from lightrag text_chunks KV store）。

        trace_chunk 表只存 chunk 身份+元数据；看正文时前端调这里按 chunk_id 取。
        """
        try:
            kv_list = await rag.text_chunks.get_by_ids([chunk_id])
            data = kv_list[0] if kv_list else None
            if not isinstance(data, dict) or not data:
                raise HTTPException(404, f"chunk not found: {chunk_id}")
            return {
                "chunk_id": chunk_id,
                "content": data.get("content", ""),
                "file_path": data.get("file_path", ""),
                "chunk_order_index": data.get("chunk_order_index"),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"get chunk {chunk_id} failed: {e}", exc_info=True)
            raise HTTPException(500, f"get chunk failed: {e}")

    @router.get("/{trace_id}/chunks")
    async def chunks_by_trace(trace_id: str):
        """该 trace 的 chunk 明细行（按 post_rerank_rank 排）。"""
        try:
            return await list_chunks_by_trace(trace_id)
        except Exception as e:
            logger.error(f"list chunks by trace failed: {e}", exc_info=True)
            raise HTTPException(500, f"list chunks failed: {e}")

    return router
