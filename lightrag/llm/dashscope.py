"""DashScope (Aliyun Bailian) text-embedding binding.

DashScope ``text-embedding-v4`` can return BOTH dense and sparse vectors, but
the ``output_type`` parameter is honored ONLY by the native DashScope API -
the OpenAI-compatible endpoint (``/compatible-mode/v1/embeddings``) silently
ignores it and returns dense alone. This binding therefore calls the native
API directly via httpx so a single request can yield dense + sparse.

It is a peer of ``openai_embed``: decorate it with
``@wrap_embedding_func_with_attrs`` and pass it as ``embedding_func=...`` to
``LightRAG`` exactly like any other embedding function.

Two call modes (same binding, same form as ``openai_embed``):

* Dense-only (drop-in, like ``openai_embed``)::

      rag = LightRAG(embedding_func=dashscope_embed, ...)
      vec = await rag.embedding_func(texts)          # -> np.ndarray (n, 1024)

* Dense + sparse (for hybrid retrieval)::

      res = await rag.embedding_func.aembed(texts, with_sparse=True)
      res.dense   # np.ndarray (n, 1024)
      res.sparse  # list[dict[int, float]]  ({token_id: weight} per text)
"""

import os
from typing import Any

import httpx
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from lightrag.utils import EmbeddingResult, logger, wrap_embedding_func_with_attrs

# Native DashScope text-embedding endpoint. The OpenAI-compatible endpoint does
# NOT support output_type (sparse), so we must use this one. Override via env
# for dedicated (WorkspaceId-scoped) deployments.
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
)
_TEXT_EMBEDDING_PATH = "/services/embeddings/text-embedding/text-embedding"


def _normalize_sparse(sp: Any) -> dict[int, float]:
    """Reduce a DashScope ``sparse_embedding`` to ``{token_id: weight}``.

    Native format: ``[{"index": int, "token": str, "value": float}, ...]``.
    The ``token`` field (human-readable subword) is kept out of the score path.
    """
    if not sp:
        return {}
    out: dict[int, float] = {}
    for item in sp:
        if isinstance(item, dict) and "index" in item and "value" in item:
            out[int(item["index"])] = float(item["value"])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out[int(item[0])] = float(item[1])
    return out


def _is_transient(exc: BaseException) -> bool:
    """Retry transient httpx errors: timeouts, connection errors, HTTP 5xx."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
    model_name="text-embedding-v4",
    send_dimensions=True,
    supports_asymmetric=True,
    supports_sparse=True,
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception(_is_transient),
)
async def dashscope_embed(
    texts: list[str],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    embedding_dim: int | None = None,
    context: str = "document",
    with_sparse: bool = False,
    timeout: float = 60.0,
) -> np.ndarray | EmbeddingResult:
    """Embed ``texts`` via the DashScope native API.

    Args:
        texts: Texts to embed.
        api_key: DashScope API key. Defaults to ``EMBEDDING_BINDING_API_KEY`` then
            ``DASHSCOPE_API_KEY``.
        base_url: Native API base URL. Defaults to ``DASHSCOPE_BASE_URL`` env
            (``https://dashscope.aliyuncs.com/api/v1``).
        model: Embedding model. Defaults to ``EMBEDDING_MODEL`` env or
            ``text-embedding-v4``.
        embedding_dim: Dense vector dimension (auto-injected by the
            ``EmbeddingFunc`` wrapper via ``send_dimensions``).
        context: ``"query"`` or ``"document"``. Informational only - v4 needs no
            asymmetric prefix; accepted for parity with ``openai_embed``.
        with_sparse: ``False`` (default) returns a dense ``np.ndarray``;
            ``True`` returns an :class:`EmbeddingResult` with dense + sparse.
        timeout: Per-request timeout in seconds.

    Returns:
        ``np.ndarray`` of shape ``(n, dim)`` when ``with_sparse=False``, else
        :class:`EmbeddingResult`.

    Raises:
        ValueError: API key not configured.
        httpx.HTTPStatusError: Non-transient HTTP error (4xx).
    """
    api_key = api_key or os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv(
        "DASHSCOPE_API_KEY", ""
    )
    base = (base_url or DASHSCOPE_BASE_URL).rstrip("/")
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    if not api_key:
        raise ValueError(
            "DashScope API key not set. Configure EMBEDDING_BINDING_API_KEY or "
            "DASHSCOPE_API_KEY (or pass api_key=)."
        )

    # Fast path: nothing to embed - avoid a round-trip.
    if not texts:
        dim = embedding_dim or 0
        dense = np.zeros((0, dim), dtype=np.float32)
        if with_sparse:
            return EmbeddingResult(dense=dense, sparse=[])
        return dense

    # output_type selects dense / sparse / both; dimension controls dense dim.
    parameters: dict[str, Any] = {
        "output_type": "dense&sparse" if with_sparse else "dense"
    }
    if embedding_dim is not None:
        parameters["dimension"] = embedding_dim

    body = {
        "model": model,
        "input": {"texts": texts},
        "parameters": parameters,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = base + _TEXT_EMBEDDING_PATH

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    embeddings = data["output"]["embeddings"]
    # Preserve input order - DashScope does not guarantee ordering of items.
    embeddings.sort(key=lambda e: e.get("text_index", 0))

    dense = np.array([e["embedding"] for e in embeddings], dtype=np.float32)

    if not with_sparse:
        return dense

    sparse = [_normalize_sparse(e.get("sparse_embedding")) for e in embeddings]
    logger.debug(
        f"dashscope_embed: {len(texts)} texts, dense={dense.shape}, "
        f"sparse token counts={[len(s) for s in sparse]}"
    )
    return EmbeddingResult(dense=dense, sparse=sparse)
