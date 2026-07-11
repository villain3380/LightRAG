"""Ingestion resource consumption tracker.

Records every LLM and embedding call made during a single document's
ingestion so the cost breakdown of a dense+sparse+KG run is visible
per-stage and per-source. Embedding is always dense+sparse in this
project; the KG path adds LLM entity-extraction + entity/relation
embedding on top of the baseline chunk embedding, and this tracker makes
that cost difference concrete in one record.

Propagation strategy: contextvars. A tracker is activated in
``process_single_document`` and every LLM/embedding call made anywhere
under it (including through ``asyncio.create_task`` / ``gather`` and the
rate-limit wrappers) reads the active tracker via ``get_active_tracker``.
No function signatures change to thread the tracker through, so recording
cannot be silently dropped at an intermediate layer - the classic
"records don't propagate" failure mode. Query paths and other callers
that never activate a tracker pay zero overhead.

A report JSON is persisted to ``{working_dir}/resource_metrics/`` in a
``finally`` block, so even a FAILED ingestion preserves its partial
record - the user already paid the LLM/embedding cost, so the record
must survive.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

# The per-document tracker currently in scope. None outside ingestion.
# asyncio.create_task / gather copy the current context, so a tracker set
# in process_single_document is visible to every chunk/entity/relation
# sub-task without threading it through signatures.
_current_tracker: contextvars.ContextVar["IngestionResourceTracker | None"] = (
    contextvars.ContextVar("lightrag_resource_tracker", default=None)
)

# The vector-storage namespace currently being embedded ("chunks" /
# "entities" / "relationships"). Set by storage flush paths so embedding
# records carry which store they served. Empty when the caller is not a
# storage flush (e.g. V-strategy chunking, query-time embedding).
_current_embedding_namespace: contextvars.ContextVar[str] = (
    contextvars.ContextVar("lightrag_embedding_namespace", default="")
)


def get_active_tracker() -> "IngestionResourceTracker | None":
    """Return the tracker active in the current async context, or None."""
    return _current_tracker.get()


def set_embedding_namespace(namespace: str) -> contextvars.Token:
    """Mark the embedding namespace for the current scope.

    Storage flush callers wrap their embed calls with this so embedding
    records are attributable to chunks / entities / relationships.
    """
    return _current_embedding_namespace.set(namespace or "")


def reset_embedding_namespace(token: contextvars.Token) -> None:
    try:
        _current_embedding_namespace.reset(token)
    except (LookupError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    """One LLM call (or cache hit) observed during ingestion."""

    stage: str  # cache_type: "extract" | "summary" | ...
    source_id: str  # chunk_id / entity_name / relation_key / ""
    cache_hit: bool  # True => no actual API call, served from LLM cache
    input_tokens: int  # estimated (prompt + system + history)
    output_tokens: int  # estimated (response); 0 on cache hit
    duration_ms: float  # wall-clock of the actual call; ~0 on cache hit
    model: str
    role: str  # which LLM role served the call, e.g. "extract"


@dataclass
class EmbeddingCallRecord:
    """One embedding API call observed during ingestion."""

    namespace: str  # "chunks" | "entities" | "relationships" | ""
    text_count: int  # number of texts embedded in this call
    token_count: int  # estimated total input tokens across all texts
    duration_ms: float
    sparse: bool  # whether sparse vectors were also produced (dense+sparse)
    model: str


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class IngestionResourceTracker:
    """Collects LLM/embedding resource records for one document ingestion.

    record_*() are synchronous (no awaits), so under asyncio's
    single-thread model they are atomic between awaits; a threading.Lock
    additionally guards the lists for any executor-based callers.
    """

    def __init__(
        self,
        doc_id: str,
        tokenizer: Any = None,
        llm_model: str = "",
        embedding_model: str = "",
    ) -> None:
        self.doc_id = doc_id
        self.tokenizer = tokenizer  # Tokenizer with .encode(str) -> list[int]
        self.llm_model = llm_model or ""
        self.embedding_model = embedding_model or ""
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.process_options: str = ""
        self.file_path: str = ""
        self.llm_calls: list[LLMCallRecord] = []
        self.embedding_calls: list[EmbeddingCallRecord] = []
        self.stage_timers: dict[str, float] = {}
        self._stage_starts: dict[str, float] = {}
        self._lock = threading.Lock()

    # ---- stage timing -----------------------------------------------------

    def start_stage(self, name: str) -> None:
        self._stage_starts[name] = time.perf_counter()

    def end_stage(self, name: str) -> float:
        start = self._stage_starts.pop(name, None)
        dur = (time.perf_counter() - start) * 1000 if start is not None else 0.0
        self.stage_timers[name] = self.stage_timers.get(name, 0.0) + dur
        return dur

    def mark_finished(self) -> None:
        self.finished_at = time.time()

    # ---- token estimation -------------------------------------------------

    def _estimate_tokens(self, text: Any) -> int:
        if not text or self.tokenizer is None:
            return 0
        if not isinstance(text, str):
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # A tokenizer failure must never break ingestion.
            return 0

    # ---- LLM recording ----------------------------------------------------

    def record_llm_call(
        self,
        *,
        stage: str,
        source_id: str | None,
        cache_hit: bool,
        user_prompt: str | None,
        system_prompt: str | None,
        history_messages: list | None,
        response: str | None,
        duration_ms: float,
        role: str = "extract",
    ) -> None:
        input_tokens = self._estimate_tokens(user_prompt) + self._estimate_tokens(
            system_prompt
        )
        if history_messages:
            for msg in history_messages:
                if isinstance(msg, dict):
                    input_tokens += self._estimate_tokens(msg.get("content"))
                elif isinstance(msg, str):
                    input_tokens += self._estimate_tokens(msg)
        # Output tokens are only incurred when the model actually ran
        # (cache hit => 0 cost).
        output_tokens = 0 if cache_hit else self._estimate_tokens(response)
        rec = LLMCallRecord(
            stage=stage or "",
            source_id=source_id or "",
            cache_hit=cache_hit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            model=self.llm_model,
            role=role,
        )
        with self._lock:
            self.llm_calls.append(rec)

    # ---- Embedding recording ---------------------------------------------

    def record_embedding_call(
        self,
        *,
        texts: list[str],
        duration_ms: float,
        sparse: bool,
    ) -> None:
        token_count = 0
        for t in texts:
            token_count += self._estimate_tokens(t)
        namespace = _current_embedding_namespace.get() or ""
        rec = EmbeddingCallRecord(
            namespace=namespace,
            text_count=len(texts),
            token_count=token_count,
            duration_ms=duration_ms,
            sparse=sparse,
            model=self.embedding_model,
        )
        with self._lock:
            self.embedding_calls.append(rec)

    # ---- summary / serialization -----------------------------------------

    def to_summary(self) -> dict:
        llm_by_stage: dict[str, dict] = {}
        for r in self.llm_calls:
            s = llm_by_stage.setdefault(
                r.stage or "unknown",
                {
                    "calls": 0,
                    "cache_hits": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "duration_ms": 0.0,
                },
            )
            s["calls"] += 1
            if r.cache_hit:
                s["cache_hits"] += 1
            s["input_tokens"] += r.input_tokens
            s["output_tokens"] += r.output_tokens
            s["duration_ms"] += r.duration_ms

        emb_by_ns: dict[str, dict] = {}
        for r in self.embedding_calls:
            ns = r.namespace or "unknown"
            s = emb_by_ns.setdefault(
                ns,
                {
                    "calls": 0,
                    "text_count": 0,
                    "token_count": 0,
                    "duration_ms": 0.0,
                    "sparse": r.sparse,
                },
            )
            s["calls"] += 1
            s["text_count"] += r.text_count
            s["token_count"] += r.token_count
            s["duration_ms"] += r.duration_ms

        total_llm_calls = len(self.llm_calls)
        total_llm_cache_hits = sum(1 for r in self.llm_calls if r.cache_hit)
        total_llm_input = sum(r.input_tokens for r in self.llm_calls)
        total_llm_output = sum(r.output_tokens for r in self.llm_calls)
        total_emb_calls = len(self.embedding_calls)
        total_emb_tokens = sum(r.token_count for r in self.embedding_calls)
        total_emb_texts = sum(r.text_count for r in self.embedding_calls)

        wall_ms = 0.0
        for v in self.stage_timers.values():
            wall_ms += v

        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "process_options": self.process_options,
            "stages_ms": dict(self.stage_timers),
            "llm_by_stage": llm_by_stage,
            "embedding_by_namespace": emb_by_ns,
            "totals": {
                "llm_calls": total_llm_calls,
                "llm_cache_hits": total_llm_cache_hits,
                "llm_input_tokens": total_llm_input,
                "llm_output_tokens": total_llm_output,
                "llm_total_tokens": total_llm_input + total_llm_output,
                "embedding_calls": total_emb_calls,
                "embedding_text_count": total_emb_texts,
                "embedding_token_count": total_emb_tokens,
                "stage_wall_ms": round(wall_ms, 1),
            },
        }

    def to_dict(self) -> dict:
        self.mark_finished()
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "process_options": self.process_options,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "summary": self.to_summary(),
            "stage_timers_ms": dict(self.stage_timers),
            "llm_calls": [asdict(r) for r in self.llm_calls],
            "embedding_calls": [asdict(r) for r in self.embedding_calls],
        }

    def save(self, dir_path: str) -> str:
        """Persist the full report as JSON. Returns the written path.

        Creates dir_path if missing. Never raises on normal filesystem
        errors - the caller wraps this in a best-effort finally.
        """
        os.makedirs(dir_path, exist_ok=True)
        ts = int(self.created_at * 1000)
        # doc_id looks like "doc-<md5>"; keep it filename-safe.
        safe_doc = self.doc_id.replace("/", "_").replace("\\", "_")
        fname = f"{safe_doc}_{ts}.json"
        path = os.path.join(dir_path, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path


def activate_tracker(tracker: IngestionResourceTracker) -> contextvars.Token:
    """Set ``tracker`` as the active tracker for the current async scope.

    Pair with ``deactivate_tracker`` in a try/finally. Returns the token
    needed to reset.
    """
    return _current_tracker.set(tracker)


def deactivate_tracker(token: contextvars.Token) -> None:
    try:
        _current_tracker.reset(token)
    except (LookupError, ValueError):
        pass
