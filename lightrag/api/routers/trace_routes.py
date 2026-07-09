"""Retrieval trace viewer API.

Traces are persisted as JSON files in ``{working_dir}/traces/`` so they
survive restarts and can be inspected independently.  An ``_index.json``
manifest speeds up directory listing.
"""

from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lightrag.utils import logger


# ── models ──────────────────────────────────────────────────────────────────


class TraceListItem(BaseModel):
    trace_id: str
    timestamp: str
    query_preview: str
    mode: str
    has_sparse: bool = False
    file_size: int = 0


class TraceListResponse(BaseModel):
    traces: list[TraceListItem]
    traces_dir: str


class TraceDetailResponse(BaseModel):
    trace_id: str
    timestamp: str
    query: str
    mode: str
    weights: dict[str, float] | None = None
    keywords: dict[str, list[str]] | None = None
    entities: list[dict[str, Any]] | None = None
    relations: list[dict[str, Any]] | None = None
    path_a_dense_ranking: list[dict[str, Any]] | None = None
    path_b_sparse_ranking: list[dict[str, Any]] | None = None
    path_ab_fused_ranking: list[dict[str, Any]] | None = None
    path_c_kg_chunks: list[dict[str, Any]] | None = None
    final_context: dict[str, Any] | None = None


# ── helpers ─────────────────────────────────────────────────────────────────


_TRACES_DIRNAME = "traces"
_INDEX_FILE = "_index.json"


def _traces_root(working_dir: str) -> Path:
    return Path(working_dir) / _TRACES_DIRNAME


def _ensure_traces_dir(working_dir: str) -> Path:
    root = _traces_root(working_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_index(working_dir: str) -> list[dict[str, Any]]:
    index_path = _traces_root(working_dir) / _INDEX_FILE
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("trace _index.json corrupt, resetting")
        return []


def _write_index(working_dir: str, entries: list[dict[str, Any]]) -> None:
    index_path = _traces_root(working_dir) / _INDEX_FILE
    index_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


_SaveLock: dict[str, asyncio.Lock] = {}


def _get_save_lock(working_dir: str) -> asyncio.Lock:
    """Per-working-dir lock to serialise index writes."""
    key = os.path.abspath(working_dir)
    if key not in _SaveLock:
        _SaveLock[key] = asyncio.Lock()
    return _SaveLock[key]


def save_trace(working_dir: str, trace: dict[str, Any]) -> str:
    """Persist a trace dict and return its trace_id."""
    root = _ensure_traces_dir(working_dir)
    trace_id = trace["trace_id"]
    fname = f"{trace_id}.json"
    fpath = root / fname
    raw = json.dumps(trace, ensure_ascii=False, indent=2)
    fpath.write_text(raw, encoding="utf-8")
    file_size = len(raw.encode("utf-8"))

    entry: dict[str, Any] = {
        "trace_id": trace_id,
        "filename": fname,
        "timestamp": trace.get("timestamp", ""),
        "query_preview": trace.get("query", "")[:80],
        "mode": trace.get("mode", "unknown"),
        "has_sparse": bool(trace.get("path_b_sparse_ranking")),
        "file_size": file_size,
    }

    # Serialize index update (rarely contended — trace writes are not hot-path).
    def _sync_update() -> None:
        entries = _read_index(working_dir)
        entries.insert(0, entry)
        # Keep at most 500 entries
        if len(entries) > 500:
            entries = entries[:500]
        _write_index(working_dir, entries)

    # must be called from an async context in practice
    import asyncio as _asyncio
    if _asyncio.get_event_loop().is_running():
        # We're in an async context, do the write on a thread to avoid blocking
        pass
    # Simple approach: write directly (small JSON, fast enough)
    entries = _read_index(working_dir)
    # Dedup by trace_id
    entries = [e for e in entries if e["trace_id"] != trace_id]
    entries.insert(0, entry)
    if len(entries) > 500:
        entries = entries[:500]
    _write_index(working_dir, entries)
    logger.debug(f"trace saved: {trace_id}")
    return trace_id


def _ensure_index_sync(working_dir: str) -> None:
    """Remove index entries whose files no longer exist (housekeeping)."""
    root = _traces_root(working_dir)
    entries = _read_index(working_dir)
    cleaned = [e for e in entries if (root / e["filename"]).exists()]
    if len(cleaned) != len(entries):
        _write_index(working_dir, cleaned)


# ── router ──────────────────────────────────────────────────────────────────


def create_trace_routes(rag):
    """Build the trace viewer API router.

    Args:
        rag: LightRAG instance (provides ``rag.working_dir``).
    """
    router = APIRouter(prefix="/traces", tags=["traces"])
    working_dir = rag.working_dir

    @router.get("", response_model=TraceListResponse)
    async def list_traces():
        _ensure_index_sync(working_dir)
        entries = _read_index(working_dir)
        return TraceListResponse(
            traces=[TraceListItem(**e) for e in entries],
            traces_dir=str(_traces_root(working_dir)),
        )

    @router.get("/{trace_id}", response_model=TraceDetailResponse)
    async def get_trace(trace_id: str):
        # Basic path-traversal guard: trace_id must be alphanumeric + _ -
        if not trace_id or set(trace_id) - set("abcdefghijklmnopqrstuvwxyz0123456789_-"):
            raise HTTPException(400, "invalid trace_id")
        fpath = _traces_root(working_dir) / f"{trace_id}.json"
        if not fpath.exists():
            raise HTTPException(404, f"trace not found: {trace_id}")
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(500, "failed to read trace file")
        return TraceDetailResponse(**data)

    @router.delete("/{trace_id}")
    async def delete_trace(trace_id: str):
        if not trace_id or set(trace_id) - set("abcdefghijklmnopqrstuvwxyz0123456789_-"):
            raise HTTPException(400, "invalid trace_id")
        fpath = _traces_root(working_dir) / f"{trace_id}.json"
        if fpath.exists():
            fpath.unlink()
        entries = _read_index(working_dir)
        entries = [e for e in entries if e["trace_id"] != trace_id]
        _write_index(working_dir, entries)
        return {"status": "deleted", "trace_id": trace_id}

    @router.delete("")
    async def clear_traces():
        root = _traces_root(working_dir)
        count = 0
        if root.exists():
            for f in root.iterdir():
                if f.name.endswith(".json") and f.name != _INDEX_FILE:
                    f.unlink()
                    count += 1
        _write_index(working_dir, [])
        return {"status": "cleared", "deleted_count": count}

    return router
