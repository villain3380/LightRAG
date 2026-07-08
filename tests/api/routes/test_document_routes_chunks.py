import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
sys.argv = _original_argv

create_document_routes = _document_routes.create_document_routes
DocStatus = _base.DocStatus

pytestmark = pytest.mark.offline


class _FakeDocStatusStorage:
    def __init__(self, docs):
        self.docs = docs

    async def get_by_id(self, doc_id):
        return self.docs.get(doc_id)


class _FakeTextChunksStorage:
    def __init__(self, chunks):
        self.chunks = chunks

    async def get_by_ids(self, ids):
        return [self.chunks.get(i) for i in ids]


_chunks = {
    "chunk-a": {
        "_id": "chunk-a",
        "content": "# Section A\n\nBody A with **markdown**.",
        "tokens": 20,
        "chunk_order_index": 0,
        "heading": {"level": 1, "text": "Section A"},
        "file_path": "doc1.pdf",
    },
    "chunk-b": {
        "_id": "chunk-b",
        "content": "Body B.",
        "tokens": 10,
        "chunk_order_index": 1,
        "file_path": "doc1.pdf",
    },
}

_docs = {
    "doc1": {
        "content_summary": "summary",
        "content_length": 100,
        "file_path": "doc1.pdf",
        "status": DocStatus.PROCESSED,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "track_id": "upload_xxx",
        "chunks_count": 2,
        # Intentionally reversed order to verify ordering by chunk_order_index.
        "chunks_list": ["chunk-b", "chunk-a", "chunk-missing"],
        "metadata": {"author": "test"},
        "content_hash": "hash1",
        "error_msg": None,
    },
    "doc-empty": {
        "content_summary": "empty",
        "content_length": 0,
        "file_path": "empty.pdf",
        "status": DocStatus.PROCESSED,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "track_id": "upload_empty",
        "chunks_count": 0,
        "chunks_list": [],
        "metadata": {},
        "content_hash": "hash2",
        "error_msg": None,
    },
}

_rag = SimpleNamespace(
    doc_status=_FakeDocStatusStorage(_docs),
    text_chunks=_FakeTextChunksStorage(_chunks),
)
_app = FastAPI()
_app.include_router(
    create_document_routes(_rag, SimpleNamespace(), api_key="test-key")
)
_client = TestClient(_app)
_headers = {"X-API-Key": "test-key"}


def test_get_document_chunks_returns_ordered_chunks():
    response = _client.get("/documents/doc1/chunks", headers=_headers)
    assert response.status_code == 200
    body = response.json()

    meta = body["doc_metadata"]
    assert meta["id"] == "doc1"
    assert meta["chunks_count"] == 2
    assert meta["status"] == "processed"
    assert meta["file_path"] == "doc1.pdf"
    assert meta["track_id"] == "upload_xxx"
    assert meta["metadata"] == {"author": "test"}

    chunks = body["chunks"]
    assert len(chunks) == 2  # chunk-missing skipped
    # Ordered by chunk_order_index (0 before 1), not by chunks_list position.
    assert chunks[0]["chunk_id"] == "chunk-a"
    assert chunks[0]["order_index"] == 0
    assert chunks[0]["tokens"] == 20
    assert "Section A" in chunks[0]["content"]
    assert chunks[0]["heading"] == {"level": 1, "text": "Section A"}
    assert chunks[1]["chunk_id"] == "chunk-b"
    assert chunks[1]["order_index"] == 1


def test_get_document_chunks_404_when_doc_missing():
    response = _client.get("/documents/nope/chunks", headers=_headers)
    assert response.status_code == 404


def test_get_document_chunks_empty_chunks_list():
    response = _client.get("/documents/doc-empty/chunks", headers=_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == []
    assert body["doc_metadata"]["chunks_count"] == 0


def test_get_document_chunks_requires_auth():
    response = _client.get("/documents/doc1/chunks")
    assert response.status_code in (401, 403)
