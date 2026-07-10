#!/usr/bin/env python3
"""
Migrate existing chunks from dense-only collection to dense+sparse collection.

Reads all chunks from ``chunks_text_embedding_v4_1024d``, re-embeds each
with DashScope native API (dense + sparse), and upserts into
``chunks_text_embedding_v4_1024d_sparse``.

No document re-parsing. No KG re-extraction. ~5 minutes for 477 chunks.

Usage:
    uv run python lightrag/evaluation/migrate_chunks_to_sparse.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from pymilvus import MilvusClient

# ── Config ──────────────────────────────────────────────────────────────────
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB = os.getenv("MILVUS_DB_NAME", "lightrag")
SOURCE_COL = "chunks_text_embedding_v4_1024d"
TARGET_COL = "chunks_text_embedding_v4_1024d_sparse"
BATCH_SIZE = 10  # dashscope batch size


async def main():
    from lightrag.llm.dashscope import dashscope_embed

    # 1 ── Read all chunks from dense-only collection ──
    client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB)

    # Get total count
    stats = client.get_collection_stats(SOURCE_COL)
    total = stats["row_count"]
    print(f"Source collection '{SOURCE_COL}': {total} chunks")

    # Query all (Milvus query iterator or paginated)
    all_chunks = []
    offset = 0
    page_size = 100
    while True:
        results = client.query(
            collection_name=SOURCE_COL,
            filter="id != ''",
            output_fields=["id", "content", "full_doc_id", "file_path", "created_at"],
            limit=page_size,
            offset=offset,
        )
        if not results:
            break
        all_chunks.extend(results)
        offset += page_size
        print(f"  read {len(all_chunks)}/{total}...", end="\r")
    print(f"\n  read {len(all_chunks)} chunks total")

    # 2 ── Re-embed with dense+sparse ──
    contents = [c["content"] for c in all_chunks]
    dense_list = []
    sparse_list = []
    api_key = os.getenv("EMBEDDING_BINDING_API_KEY", "")

    n_batches = (len(contents) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.perf_counter()
    for i in range(0, len(contents), BATCH_SIZE):
        batch = contents[i : i + BATCH_SIZE]
        bnum = i // BATCH_SIZE + 1
        print(f"  embedding batch {bnum}/{n_batches} ({len(batch)} texts)...", end="\r")
        res = await dashscope_embed.aembed(batch, with_sparse=True, api_key=api_key)
        dense_list.append(res.dense)
        sparse_list.extend(res.sparse)
    elapsed = time.perf_counter() - t0
    print(f"\n  embedded {len(all_chunks)} chunks in {elapsed:.1f}s "
          f"({len(all_chunks)/elapsed:.0f} chunks/s)")

    dense_all = np.concatenate(dense_list)

    # 3 ── Upsert to sparse collection ──
    records = []
    for idx, chunk in enumerate(all_chunks):
        records.append({
            "id": chunk["id"],
            "vector": dense_all[idx].tolist(),
            "sparse_vector": sparse_list[idx],
            "full_doc_id": chunk.get("full_doc_id", ""),
            "content": chunk["content"],
            "file_path": chunk.get("file_path", ""),
            "created_at": chunk.get("created_at", 0),
        })

    # Batch insert
    n_upsert_batches = (len(records) + 50 - 1) // 50
    print(f"  upserting {len(records)} records to '{TARGET_COL}'...")
    for j in range(0, len(records), 50):
        bn = j // 50 + 1
        batch = records[j : j + 50]
        print(f"    batch {bn}/{n_upsert_batches}...", end="\r")
        client.upsert(collection_name=TARGET_COL, data=batch)
    print(f"\n  upsert done")

    # 4 ── Verify ──
    target_stats = client.get_collection_stats(TARGET_COL)
    print(f"\n[DONE] Target collection '{TARGET_COL}': {target_stats['row_count']} chunks")


if __name__ == "__main__":
    asyncio.run(main())
