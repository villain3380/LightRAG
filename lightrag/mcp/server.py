"""
LightRAG MCP Server — wraps LightRAG REST API as an MCP server.

Transport modes
  stdio (default) — local use only, no network port.
    The agent (Claude Code / Hermes) launches this process and talks via
    stdin/stdout.  Use this when the agent runs on the same machine.

  sse — Server-Sent Events over HTTP, needs a TCP port.
    Use this to let remote agents (another PC on the same WiFi, a VM, or
    a container) connect.  Set LIGHTRAG_MCP_TRANSPORT=sse and bind to
    0.0.0.0 with a chosen port.

Usage
  # 1. Start the REST API server first (separate terminal):
  lightrag-server

  # 2a. Local agent (stdio, default):
  lightrag-mcp-server

  # 2b. Remote agents (SSE):
  LIGHTRAG_MCP_TRANSPORT=sse LIGHTRAG_MCP_HOST=0.0.0.0 LIGHTRAG_MCP_PORT=8720 lightrag-mcp-server

Configuration (via .env or env vars)
  LightRAG REST API connection
    LIGHTRAG_API_BASE_URL — REST API base URL (default: http://127.0.0.1:9621)
    LIGHTRAG_API_KEY      — optional API key sent as X-API-Key header
    LIGHTRAG_MCP_TIMEOUT  — HTTP request timeout in seconds (default: 120)

  MCP transport
    LIGHTRAG_MCP_TRANSPORT — "stdio" (default) or "sse"
    LIGHTRAG_MCP_HOST      — bind address for SSE (default: 127.0.0.1)
    LIGHTRAG_MCP_PORT      — bind port for SSE (default: 8720)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── Logging ────────────────────────────────────────────────────────────
# MCP stdio transport uses stdout for JSON-RPC — ALL logging goes to stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("lightrag.mcp")

# ── Config ─────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=".env", override=False)

API_BASE_URL = os.getenv("LIGHTRAG_API_BASE_URL", "http://127.0.0.1:9621").rstrip("/")
API_KEY = os.getenv("LIGHTRAG_API_KEY", None)
DEFAULT_TIMEOUT = int(os.getenv("LIGHTRAG_MCP_TIMEOUT", "120"))

MCP_TRANSPORT = os.getenv("LIGHTRAG_MCP_TRANSPORT", "stdio")
MCP_HOST = os.getenv("LIGHTRAG_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("LIGHTRAG_MCP_PORT", "8720"))

# ── MCP Server ─────────────────────────────────────────────────────────
mcp = FastMCP(
    "lightrag",
    host=MCP_HOST,
    port=MCP_PORT,
    instructions=(
        "LightRAG knowledge base server. You can search documents, insert new "
        "knowledge, browse document lists, explore the knowledge graph, and "
        "query data without LLM generation. All destructive operations "
        "(delete, clear, cancel) are intentionally excluded for safety."
    ),
)

# ── HTTP helpers ───────────────────────────────────────────────────────


def _client() -> httpx.AsyncClient:
    """Build an httpx AsyncClient with base URL and optional API key."""
    headers: dict[str, str] = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return httpx.AsyncClient(
        base_url=API_BASE_URL,
        headers=headers,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
    )


async def _get(path: str, **params: Any) -> dict[str, Any]:
    """GET request with query params, returning parsed JSON."""
    async with _client() as cli:
        resp = await cli.get(path, params=params)
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST request with JSON body, returning parsed JSON."""
    async with _client() as cli:
        resp = await cli.post(path, json=body or {})
        resp.raise_for_status()
        return resp.json()


async def _stream_post(path: str, body: dict[str, Any] | None = None) -> str:
    """POST request with streaming NDJSON response, returns concatenated text."""
    async with _client() as cli:
        async with cli.stream("POST", path, json=body or {}) as resp:
            resp.raise_for_status()
            chunks: list[str] = []
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    chunks.append(line)
                    continue
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                if "response" in obj:
                    chunks.append(obj["response"])
            return "".join(chunks)


def _fmt(data: Any) -> str:
    """Format any value as a readable JSON string."""
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── Tools: Query ───────────────────────────────────────────────────────


@mcp.tool(
    name="lightrag_search",
    description=(
        "Search the LightRAG knowledge base with LLM-generated answer. "
        "Performs hybrid retrieval (knowledge graph + vector search) and "
        "generates a natural-language response from the retrieved context. "
        "Use this when you need a comprehensive answer to a question."
    ),
)
async def lightrag_search(
    query: str,
    mode: str = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    only_need_context: bool = False,
    response_type: str = "Multiple Paragraphs",
    include_references: bool = True,
    user_prompt: str | None = None,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
) -> str:
    """
    Args:
        query: The search query text (min 3 characters).
        mode: Retrieval mode — "local" (entity-focused), "global" (relationship-
              focused), "hybrid" (both), "naive" (vector only), "mix" (KG +
              vector, default).
        top_k: Number of top entities/relationships to retrieve. Defaults to
               server setting (usually 60).
        chunk_top_k: Number of text chunks to retrieve and rerank.
        only_need_context: If True, returns only retrieved context without LLM
                           generation.
        response_type: Format hint for the LLM response. Common values:
                       "Multiple Paragraphs", "Single Paragraph", "Bullet Points".
        include_references: If True, includes citation references in the response.
        user_prompt: Extra instruction injected into the LLM prompt for
                     customizing the answer style.
        hl_keywords: High-level keywords that prioritize certain entities in
                     retrieval.
        ll_keywords: Low-level keywords that refine the retrieval focus.
    """
    body: dict[str, Any] = {"query": query, "mode": mode, "stream": False}
    if top_k is not None:
        body["top_k"] = top_k
    if chunk_top_k is not None:
        body["chunk_top_k"] = chunk_top_k
    if only_need_context:
        body["only_need_context"] = True
    if response_type:
        body["response_type"] = response_type
    body["include_references"] = include_references
    if user_prompt:
        body["user_prompt"] = user_prompt
    if hl_keywords:
        body["hl_keywords"] = hl_keywords
    if ll_keywords:
        body["ll_keywords"] = ll_keywords

    logger.info("lightrag_search: query=%r mode=%s", query, mode)
    result = await _post("/query", body)
    return _fmt(result)


@mcp.tool(
    name="lightrag_search_data",
    description=(
        "Retrieve structured data from the LightRAG knowledge base WITHOUT "
        "LLM generation. Returns entities, relationships, chunks, and "
        "references matching the query. Use this when you need raw retrieved "
        "data for further processing or when you want to inspect what the "
        "knowledge base contains about a topic without AI summarization."
    ),
)
async def lightrag_search_data(
    query: str,
    mode: str = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    include_chunk_content: bool = True,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
) -> str:
    """
    Args:
        query: The search query text (min 3 characters).
        mode: Retrieval mode — "local", "global", "hybrid", "naive", "mix"
              (default "mix").
        top_k: Number of top entities/relationships to retrieve.
        chunk_top_k: Number of text chunks to retrieve and rerank.
        include_chunk_content: If True, includes full chunk content in results.
        hl_keywords: High-level priority keywords.
        ll_keywords: Low-level refinement keywords.
    """
    body: dict[str, Any] = {"query": query, "mode": mode}
    if top_k is not None:
        body["top_k"] = top_k
    if chunk_top_k is not None:
        body["chunk_top_k"] = chunk_top_k
    body["include_chunk_content"] = include_chunk_content
    if hl_keywords:
        body["hl_keywords"] = hl_keywords
    if ll_keywords:
        body["ll_keywords"] = ll_keywords

    logger.info("lightrag_search_data: query=%r mode=%s", query, mode)
    result = await _post("/query/data", body)
    return _fmt(result)


@mcp.tool(
    name="lightrag_stream_search",
    description=(
        "Streaming search — same retrieval as lightrag_search but returns "
        "the LLM answer token-by-token (collected into a single result). "
        "Use this for long-form answers where you want the full response."
    ),
)
async def lightrag_stream_search(
    query: str,
    mode: str = "mix",
    top_k: int | None = None,
    response_type: str = "Multiple Paragraphs",
    user_prompt: str | None = None,
) -> str:
    """
    Args:
        query: The search query text.
        mode: Retrieval mode — "local", "global", "hybrid", "naive", "mix".
        top_k: Number of top entities/relationships to retrieve.
        response_type: Format hint for the LLM response.
        user_prompt: Extra instruction for customizing the answer style.
    """
    body: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "stream": True,
        "response_type": response_type,
    }
    if top_k is not None:
        body["top_k"] = top_k
    if user_prompt:
        body["user_prompt"] = user_prompt

    logger.info("lightrag_stream_search: query=%r mode=%s", query, mode)
    result = await _stream_post("/query/stream", body)
    return result


# ── Tools: Documents ───────────────────────────────────────────────────


@mcp.tool(
    name="lightrag_insert_text",
    description=(
        "Insert a single text document into the LightRAG knowledge base. "
        "The document will be chunked, embedded, and indexed automatically. "
        "Returns a track_id for monitoring processing progress."
    ),
)
async def lightrag_insert_text(
    text: str,
    file_path: str | None = None,
    description: str | None = None,
    split_by_character: str | None = None,
) -> str:
    """
    Args:
        text: The full text content to insert.
        file_path: Optional logical file path for citation (e.g., "reports/
                   quarterly.md"). Does NOT read from disk — use
                   lightrag_upload_document for physical files.
        description: Optional human-readable description for the document.
        split_by_character: If set, split text by this character before
                            token-level chunking (e.g., "\\n\\n" for paragraphs).
    """
    body: dict[str, Any] = {"text": text}
    if file_path:
        body["file_path"] = file_path
    if description:
        body["description"] = description
    if split_by_character:
        body["split_by_character"] = split_by_character

    logger.info("lightrag_insert_text: len=%d file_path=%s", len(text), file_path)
    result = await _post("/documents/text", body)
    return _fmt(result)


@mcp.tool(
    name="lightrag_insert_texts",
    description=(
        "Insert multiple text documents into the LightRAG knowledge base in "
        "a single batch. Each document is chunked, embedded, and indexed."
    ),
)
async def lightrag_insert_texts(
    texts: list[str],
    file_paths: list[str] | None = None,
    descriptions: list[str] | None = None,
) -> str:
    """
    Args:
        texts: List of text content strings to insert.
        file_paths: Optional parallel list of logical file paths for citation.
        descriptions: Optional parallel list of human-readable descriptions.
    """
    body: dict[str, Any] = {"texts": texts}
    if file_paths:
        body["file_paths"] = file_paths
    if descriptions:
        body["descriptions"] = descriptions

    logger.info("lightrag_insert_texts: count=%d", len(texts))
    result = await _post("/documents/texts", body)
    return _fmt(result)


@mcp.tool(
    name="lightrag_upload_document",
    description=(
        "Upload a file from the local filesystem into the LightRAG knowledge "
        "base. The file is copied into the server's input directory and "
        "processed (parsed, chunked, embedded, indexed). Supported formats "
        "include .txt, .md, .pdf, .docx, .pptx, .xlsx, .csv, .html, and more."
    ),
)
async def lightrag_upload_document(
    file_path: str,
) -> str:
    """
    Args:
        file_path: Absolute or relative path to the file on the MCP host's
                   filesystem. The file must exist and be readable.
    """
    # Read the file and upload it via multipart/form-data
    path = Path(file_path)
    if not path.exists():
        return _fmt({"status": "error", "message": f"File not found: {file_path}"})

    file_size = path.stat().st_size
    logger.info("lightrag_upload_document: %s (%d bytes)", path.name, file_size)

    headers: dict[str, str] = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    async with httpx.AsyncClient(
        base_url=API_BASE_URL,
        headers=headers,
        timeout=httpx.Timeout(300.0),  # uploads can be large
    ) as cli:
        # FastAPI expects multipart form with "file" field
        with open(path, "rb") as f:
            files = {"file": (path.name, f)}
            resp = await cli.post("/documents/upload", files=files)
            resp.raise_for_status()
            return _fmt(resp.json())


@mcp.tool(
    name="lightrag_list_docs",
    description=(
        "List documents in the LightRAG knowledge base with pagination, "
        "status filtering, and sorting."
    ),
)
async def lightrag_list_docs(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    status_filters: list[str] | None = None,
    sort_field: str = "updated_at",
    sort_direction: str = "desc",
) -> str:
    """
    Args:
        page: Page number (1-based).
        page_size: Number of documents per page (max 100).
        status_filter: Single status value to filter by.
            Values: "pending", "parsing", "analyzing", "processing",
            "preprocessed", "processed", "failed".
        status_filters: Multiple status values (takes precedence over
                        status_filter).
        sort_field: Field to sort by — "created_at", "updated_at", "id",
                    "file_path".
        sort_direction: "asc" or "desc".
    """
    body: dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "sort_field": sort_field,
        "sort_direction": sort_direction,
    }
    if status_filters:
        body["status_filters"] = status_filters
    elif status_filter:
        body["status_filter"] = status_filter

    logger.info("lightrag_list_docs: page=%d size=%d", page, page_size)
    result = await _post("/documents/paginated", body)
    return _fmt(result)


@mcp.tool(
    name="lightrag_get_doc_status",
    description=(
        "Get the processing status of documents by track_id. Returns which "
        "documents were submitted together and their current pipeline status."
    ),
)
async def lightrag_get_doc_status(
    track_id: str,
) -> str:
    """
    Args:
        track_id: The tracking ID returned by lightrag_insert_text,
                  lightrag_insert_texts, or lightrag_upload_document.
    """
    logger.info("lightrag_get_doc_status: track_id=%s", track_id)
    result = await _get(f"/documents/track_status/{track_id}")
    return _fmt(result)


@mcp.tool(
    name="lightrag_get_status_counts",
    description=(
        "Get the count of documents in each processing status. Useful for "
        "checking if a batch insert has completed processing."
    ),
)
async def lightrag_get_status_counts() -> str:
    """Returns counts of documents grouped by status."""
    logger.info("lightrag_get_status_counts")
    result = await _get("/documents/status_counts")
    return _fmt(result)


@mcp.tool(
    name="lightrag_get_pipeline_status",
    description=(
        "Get the current pipeline processing status — whether the pipeline "
        "is idle or busy, how many documents are queued, and current progress."
    ),
)
async def lightrag_get_pipeline_status() -> str:
    """Returns pipeline health and progress information."""
    logger.info("lightrag_get_pipeline_status")
    result = await _get("/documents/pipeline_status")
    return _fmt(result)


@mcp.tool(
    name="lightrag_get_doc_chunks",
    description=(
        "Get all chunks of a specific document. Shows how the document was "
        "split during ingestion — useful for inspecting chunking quality "
        "and understanding what the retriever sees."
    ),
)
async def lightrag_get_doc_chunks(
    doc_id: str,
) -> str:
    """
    Args:
        doc_id: The document ID (can be found via lightrag_list_docs).
    """
    logger.info("lightrag_get_doc_chunks: doc_id=%s", doc_id)
    result = await _get(f"/documents/{doc_id}/chunks")
    return _fmt(result)


# ── Tools: Knowledge Graph ─────────────────────────────────────────────


@mcp.tool(
    name="lightrag_knowledge_graph",
    description=(
        "Retrieve a subgraph from the LightRAG knowledge graph. Starting "
        "from nodes matching the given label, traverses relationships up to "
        "a specified depth. Returns entities and their relationships. Use "
        "this to explore connections between concepts."
    ),
)
async def lightrag_knowledge_graph(
    label: str = "",
    max_depth: int = 3,
    max_nodes: int | None = None,
) -> str:
    """
    Args:
        label: Graph node label to start from (empty string = all labels).
               Use lightrag_graph_labels to discover available labels.
        max_depth: Maximum graph traversal depth (1-5, default 3).
        max_nodes: Maximum number of nodes to return. Default is unlimited
                   (server-capped).
    """
    body: dict[str, Any] = {"label": label, "max_depth": max_depth}
    if max_nodes is not None:
        body["max_nodes"] = max_nodes

    logger.info("lightrag_knowledge_graph: label=%r depth=%d", label, max_depth)
    result = await _post("/graphs", body)
    return _fmt(result)


@mcp.tool(
    name="lightrag_graph_labels",
    description=(
        "List all entity labels/types in the knowledge graph. Helps you "
        "understand what kinds of entities have been extracted from documents."
    ),
)
async def lightrag_graph_labels() -> str:
    """Returns all distinct entity labels in the knowledge graph."""
    logger.info("lightrag_graph_labels")
    result = await _get("/graph/label/list")
    return _fmt(result)


@mcp.tool(
    name="lightrag_graph_popular_labels",
    description=(
        "Get the most popular (most connected) entity labels in the "
        "knowledge graph, ranked by node degree."
    ),
)
async def lightrag_graph_popular_labels() -> str:
    """Returns labels ranked by popularity (node degree)."""
    logger.info("lightrag_graph_popular_labels")
    result = await _get("/graph/label/popular")
    return _fmt(result)


@mcp.tool(
    name="lightrag_graph_search_labels",
    description=(
        "Fuzzy search for entity labels in the knowledge graph by keyword. "
        "Use this to find relevant entity types before querying the full graph."
    ),
)
async def lightrag_graph_search_labels(
    keyword: str,
) -> str:
    """
    Args:
        keyword: Search keyword for fuzzy-matching label names.
    """
    logger.info("lightrag_graph_search_labels: keyword=%r", keyword)
    result = await _get("/graph/label/search", query=keyword)
    return _fmt(result)


@mcp.tool(
    name="lightrag_check_entity",
    description=(
        "Check whether a specific entity exists in the knowledge graph."
    ),
)
async def lightrag_check_entity(
    entity_name: str,
) -> str:
    """
    Args:
        entity_name: The name of the entity to check.
    """
    logger.info("lightrag_check_entity: %r", entity_name)
    result = await _get("/graph/entity/exists", entity_name=entity_name)
    return _fmt(result)


# ── Tools: System ──────────────────────────────────────────────────────


@mcp.tool(
    name="lightrag_health_check",
    description=(
        "Check the health and configuration of the LightRAG REST API server. "
        "Returns server status, storage backend info, LLM/embedding "
        "configuration, and whether the server is ready to accept requests."
    ),
)
async def lightrag_health_check() -> str:
    """Returns system health and configuration status."""
    logger.info("lightrag_health_check")
    result = await _get("/health")
    return _fmt(result)


# ── Resources ──────────────────────────────────────────────────────────


@mcp.resource(
    uri="kb://documents/{doc_id}",
    name="Document chunks",
    description=(
        "Access the chunked content of a document in the knowledge base."
    ),
    mime_type="application/json",
)
async def get_document_resource(doc_id: str) -> str:
    """Returns all chunks for the given document."""
    logger.info("resource: kb://documents/%s", doc_id)
    result = await _get(f"/documents/{doc_id}/chunks")
    return _fmt(result)


# ── Prompts ────────────────────────────────────────────────────────────


@mcp.prompt(
    name="knowledge_qa",
    description=(
        "Optimized prompt template for knowledge-base question answering. "
        "Guides the model to answer strictly based on retrieved results, "
        "cite sources, and acknowledge gaps when information is unavailable."
    ),
)
async def knowledge_qa_prompt(query: str) -> str:
    """Build a structured QA prompt for the given query."""
    return (
        "You are answering a question using the LightRAG knowledge base.\n\n"
        "## Rules\n"
        "1. Answer ONLY based on the retrieved knowledge. Do NOT fabricate.\n"
        "2. Cite specific sources (document file_path or chunk_id) for each claim.\n"
        "3. If the knowledge base does not contain enough information, explicitly\n"
        "   state that the information is not available — never guess.\n"
        "4. Structure your answer clearly with sections when appropriate.\n"
        "5. When citing, use the reference_id values from the retrieved data.\n\n"
        "## Query\n"
        f"{query}\n\n"
        "## Steps\n"
        "1. First, use lightrag_search to retrieve relevant knowledge.\n"
        "2. If more factual detail is needed, use lightrag_search_data to inspect\n"
        "   entities, relationships, and raw chunks.\n"
        "3. If you need to understand the document landscape, use "
        "lightrag_list_docs.\n"
        "4. Synthesize your answer from the retrieved data.\n"
    )


# ── Entry Point ────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — start the MCP server.

    Transport is controlled by ``LIGHTRAG_MCP_TRANSPORT``:

    ``stdio`` (default)
        Local agent usage. The agent spawns this process and talks
        JSON-RPC over stdin/stdout. No network port is opened.

    ``sse``
        Remote agent usage. Starts an HTTP server on
        ``LIGHTRAG_MCP_HOST``:``LIGHTRAG_MCP_PORT`` that accepts MCP
        connections via Server-Sent Events. Other machines on the same
        network (or VMs) can connect to this endpoint.

    The LightRAG REST API server must be running BEFORE this::

        lightrag-server                     # terminal 1
        lightrag-mcp-server                 # terminal 2 (stdio)
        # or for remote access:
        LIGHTRAG_MCP_TRANSPORT=sse LIGHTRAG_MCP_HOST=0.0.0.0 lightrag-mcp-server
    """
    logger.info(
        "LightRAG MCP server starting (transport=%s, API=%s)",
        MCP_TRANSPORT,
        API_BASE_URL,
    )
    if MCP_TRANSPORT == "sse":
        logger.info(
            "SSE endpoint: http://%s:%d/sse  |  messages: http://%s:%d/messages/",
            MCP_HOST,
            MCP_PORT,
            MCP_HOST,
            MCP_PORT,
        )
    mcp.run(transport=MCP_TRANSPORT)


if __name__ == "__main__":
    main()
