"""Query path trace tracker (retrieval pipeline telemetry).

Records per-/query-call stage timings (rewrite/retrieval/rerank/generation),
recall counts, cache hit, and failure info for the /query retrieval path.
One tracker per /query call.

Propagation strategy: contextvars - mirrors resource_tracker.py. The tracker
is activated in the route layer (query_routes.py) around ``rag.aquery_llm``;
deep calls in operate.py / utils.py read it via ``get_active_query_tracker``
without any signature changes. asyncio.create_task / gather copy the current
context, so the tracker is visible to every retrieval sub-task. Query paths
that never activate a tracker (no route-level activation) pay zero overhead.

The route layer ships the finished tracker to data_platform
``shared.query_trace`` (fire-and-forget) after ``aquery_llm`` returns, so
lightrag core never depends on data_platform.

Field semantics for the rewrite/retrieval split:
- ``rewrite_ms``: agent LLM latency from user message to the /query tool call.
  Measured on the agent side (dp_server) and passed in; None for direct
  queries that bypass the agent.
- ``retrieval_ms``: the main query->chunks vector search (query_hybrid).
- ``rerank_ms``: apply_rerank_if_enabled (None if rerank didn't fire).
- ``generation_ms``: cache lookup + LLM generation (includes the cache-hit
  branch; ``cache_hit`` explains a near-zero value).

Multi-query: one agent turn may issue N /query calls (parallel or sequential).
Each call gets its own tracker/row; they share ``turn_id`` (and, for a parallel
batch, ``rewrite_ms``) so the N rows group cleanly.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variable
# ---------------------------------------------------------------------------

# The per-/query-call tracker currently in scope. None outside a traced query.
# asyncio.create_task / gather copy the current context, so a tracker set in
# the route handler is visible to every retrieval/generation sub-call without
# threading it through signatures.
_current_tracker: contextvars.ContextVar["QueryTraceTracker | None"] = (
    contextvars.ContextVar("lightrag_query_trace_tracker", default=None)
)


def get_active_query_tracker() -> "QueryTraceTracker | None":
    """Return the tracker active in the current async context, or None."""
    return _current_tracker.get()


def activate_query_tracker(
    tracker: "QueryTraceTracker",
) -> contextvars.Token:
    """Set ``tracker`` as active for the current async scope.

    Pair with ``deactivate_query_tracker`` in a try/finally.
    """
    return _current_tracker.set(tracker)


def deactivate_query_tracker(token: contextvars.Token) -> None:
    try:
        _current_tracker.reset(token)
    except (LookupError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class QueryTraceTracker:
    """Collects stage timings + recall info for one /query call.

    ``start_stage`` / ``end_stage`` are synchronous (no awaits), so under
    asyncio's single-thread model they are atomic between awaits; a
    threading.Lock additionally guards the measured fields for any
    executor-based callers. Stage names map to ``<name>_ms`` attributes:
    ``retrieval`` / ``rerank`` / ``generation``.
    """

    def __init__(
        self,
        *,
        turn_id: str,
        trace_id: str,
        call_index: int | None,
        original_query: str,
        search_query: str,
        mode: str,
        source: str,
        endpoint: str | None,
        rewrite_ms: float | None,
        query_params: dict[str, Any],
    ) -> None:
        # request-level (from QueryParam / request)
        self.turn_id = turn_id
        self.trace_id = trace_id
        self.call_index = call_index
        self.original_query = original_query
        self.search_query = search_query
        self.mode = mode
        self.source = source
        self.endpoint = endpoint
        self.rewrite_ms = rewrite_ms  # from agent; None for direct queries
        self.query_params = query_params or {}
        # measured (filled by operate.py / utils.py)
        self.retrieval_ms: float | None = None
        self.rerank_ms: float | None = None
        self.generation_ms: float | None = None
        # retrieval sub-stages (KG modes only; break down _perform_kg_search).
        # For mix: retrieval_ms ≈ kg_embed_ms + kg_entity_ms + kg_relation_ms
        # + kg_chunk_ms (sequential). naive leaves these None.
        self.kg_embed_ms: float | None = None
        self.kg_entity_ms: float | None = None
        self.kg_relation_ms: float | None = None
        self.kg_chunk_ms: float | None = None
        self.cache_hit = False
        self.chunks_retrieved: int | None = None
        self.chunks_after_rerank: int | None = None
        self.chunks_final: int | None = None
        self.entities_count: int | None = None
        self.relations_count: int | None = None
        self.llm_model: str = ""
        self.context_tokens: int | None = None
        self.response_tokens: int | None = None
        # status
        self.status = "success"
        self.failure_stage: str | None = None
        self.error_msg: str | None = None
        self._stage_starts: dict[str, float] = {}
        self._lock = threading.Lock()

    # ---- stage timing -----------------------------------------------------

    def start_stage(self, name: str) -> None:
        self._stage_starts[name] = time.perf_counter()

    def end_stage(self, name: str) -> float | None:
        start = self._stage_starts.pop(name, None)
        if start is None:
            return None
        dur = (time.perf_counter() - start) * 1000
        with self._lock:
            setattr(self, f"{name}_ms", dur)
        return dur

    # ---- recall counts ----------------------------------------------------

    def set_chunks_retrieved(self, n: int) -> None:
        """Record the pre-rerank chunk count (first call wins; later
        entity/relation sub-retrievals don't overwrite)."""
        with self._lock:
            if self.chunks_retrieved is None:
                self.chunks_retrieved = n

    def set_chunks_after_rerank(self, n: int) -> None:
        with self._lock:
            if self.chunks_after_rerank is None:
                self.chunks_after_rerank = n

    # ---- failure ----------------------------------------------------------

    def record_failure(self, stage: str, msg: str | None) -> None:
        self.status = "failed"
        self.failure_stage = stage
        self.error_msg = msg

    def record_no_results(self) -> None:
        self.status = "no_results"

    # ---- serialization ----------------------------------------------------

    def to_ship_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "trace_id": self.trace_id,
            "call_index": self.call_index,
            "original_query": self.original_query,
            "search_query": self.search_query,
            "rewritten": bool(
                self.original_query
                and self.search_query
                and self.original_query != self.search_query
            ),
            "mode": self.mode,
            "query_params": self.query_params,
            "source": self.source,
            "endpoint": self.endpoint,
            "rewrite_ms": self.rewrite_ms,
            "retrieval_ms": self.retrieval_ms,
            "rerank_ms": self.rerank_ms,
            "generation_ms": self.generation_ms,
            "kg_embed_ms": self.kg_embed_ms,
            "kg_entity_ms": self.kg_entity_ms,
            "kg_relation_ms": self.kg_relation_ms,
            "kg_chunk_ms": self.kg_chunk_ms,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_after_rerank": self.chunks_after_rerank,
            "chunks_final": self.chunks_final,
            "entities_count": self.entities_count,
            "relations_count": self.relations_count,
            "cache_hit": self.cache_hit,
            "llm_model": self.llm_model,
            "context_tokens": self.context_tokens,
            "response_tokens": self.response_tokens,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "error_msg": self.error_msg,
        }
