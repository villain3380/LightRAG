"""Integration test: Milvus dense+sparse hybrid chunk retrieval.

Requires:
  - Milvus running (MILVUS_URI in .env, here http://localhost:19530).
  - DashScope API key in .env (EMBEDDING_BINDING_API_KEY).

Directly exercises the storage layer (no LLM, no KG extraction):
  1. LightRAG(embedding_func=dashscope_embed, chunk_retrieval_mode=dense_sparse,
              vector_storage=MilvusVectorDBStorage).
  2. Upsert 3 chunks directly into chunks_vdb + flush (aembed with_sparse).
  3. Verify the Milvus collection schema has a sparse_vector field.
  4. query_hybrid (dense+sparse) vs query (dense) for a related and unrelated
     query; assert the target chunk ranks #1 for the related query and the
     unrelated query does not surface it at the top.

Run:  python examples/test_milvus_sparse_retrieval.py
"""

import os
import sys
import asyncio

from dotenv import load_dotenv

load_dotenv()

from lightrag import LightRAG  # noqa: E402
from lightrag.llm.dashscope import dashscope_embed  # noqa: E402
from lightrag.utils import compute_mdhash_id, logger  # noqa: E402

WORKING_DIR = "./milvus_sparse_test"
WORKSPACE = "sparse_test"
MODE = "dense_sparse"  # chunk_retrieval_mode

# Three chunks: A is the retrieval target; B/C are distractors.
CHUNKS = [
    "LightRAG is a retrieval-augmented generation framework. It builds a "
    "knowledge graph from documents and supports local, global, hybrid, and "
    "naive query modes for flexible retrieval.",
    "Python is a popular high-level programming language used for web "
    "development, data analysis, and automation.",
    "The Mediterranean climate features mild wet winters and hot dry summers.",
]
Q_RELATED = "What query modes does LightRAG support?"
Q_UNRELATED = "How do I cook beef wellington?"


async def dummy_complete(*a, **k):
    return ""


def top_contents(results):
    return [r.get("content")[:50] for r in results]


def rank_of(results, needle_prefix):
    for i, r in enumerate(results):
        if (r.get("content") or "").startswith(needle_prefix):
            return i
    return None


async def main():
    if not os.getenv("EMBEDDING_BINDING_API_KEY"):
        print("EMBEDDING_BINDING_API_KEY not set")
        sys.exit(1)
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        workspace=WORKSPACE,
        embedding_func=dashscope_embed,
        llm_model_func=dummy_complete,
        vector_storage="MilvusVectorDBStorage",
        addon_params={"chunk_retrieval_mode": MODE},
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
    )
    await rag.initialize_storages()

    cvdb = rag.chunks_vdb
    print(f"sparse_enabled = {cvdb.sparse_enabled}")
    print(f"final_namespace = {cvdb.final_namespace}")
    assert cvdb.sparse_enabled, "sparse_enabled should be True for chunks+dense_sparse"

    try:
        # 1) Direct upsert + flush (triggers aembed with_sparse at flush time).
        doc_id = compute_mdhash_id("doc1", prefix="doc-")
        data = {}
        for i, txt in enumerate(CHUNKS):
            cid = compute_mdhash_id(txt, prefix="chunk-")
            data[cid] = {
                "content": txt,
                "full_doc_id": doc_id,
                "file_path": f"test_{i}.txt",
            }
        await cvdb.upsert(data)
        await cvdb.index_done_callback()  # flush: embed dense+sparse, write Milvus
        print(f"upserted + flushed {len(data)} chunks")

        # 2) Verify collection schema has sparse_vector.
        desc = cvdb._client.describe_collection(cvdb.final_namespace)
        field_names = [f["name"] for f in desc["fields"]]
        print(f"collection fields: {field_names}")
        assert "sparse_vector" in field_names, "sparse_vector field missing!"
        print("sparse_vector field present OK")

        # 3) Retrieval: hybrid vs dense.
        print("\n=== related query ===")
        print(f"  {Q_RELATED!r}")
        hyb_rel = await cvdb.query_hybrid(Q_RELATED, top_k=3)
        den_rel = await cvdb.query(Q_RELATED, top_k=3)
        print(f"  hybrid top3: {top_contents(hyb_rel)}")
        print(f"  dense  top3: {top_contents(den_rel)}")

        print("\n=== unrelated query ===")
        print(f"  {Q_UNRELATED!r}")
        hyb_unrel = await cvdb.query_hybrid(Q_UNRELATED, top_k=3)
        print(f"  hybrid top3: {top_contents(hyb_unrel)}")

        # 4) Sanity: target chunk (CHUNKS[0]) ranks #1 on related, not #1 on unrelated.
        target_prefix = CHUNKS[0][:30]
        rel_rank = rank_of(hyb_rel, target_prefix)
        unrel_rank = rank_of(hyb_unrel, target_prefix)
        print(f"\ntarget rank (hybrid, related):   {rel_rank}")
        print(f"target rank (hybrid, unrelated): {unrel_rank}")
        ok = rel_rank == 0 and (unrel_rank is None or unrel_rank > 0)
        print("\n=== sanity ===")
        print(f"target #1 on related AND not #1 on unrelated ? {ok}")
        if ok:
            print("PASS: Milvus dense+sparse hybrid retrieval works end-to-end.")
        else:
            print("WARN: inspect ranks above.")
            sys.exit(1)

        # 5) Weight variation: (1.0, 0.0) short-circuits to dense; (0.7, 0.3) hybrid.
        print("\n=== weight variation ===")
        dense_only = await cvdb.query_hybrid(
            Q_RELATED, top_k=3, dense_weight=1.0, sparse_weight=0.0
        )
        w73 = await cvdb.query_hybrid(
            Q_RELATED, top_k=3, dense_weight=0.7, sparse_weight=0.3
        )
        do_rank = rank_of(dense_only, target_prefix)
        w73_rank = rank_of(w73, target_prefix)
        print(f"  (1.0, 0.0) target rank: {do_rank}  (dense-only short-circuit)")
        print(f"  (0.7, 0.3) target rank: {w73_rank}  (weighted hybrid)")
        print(f"  (1.0, 0.0) top3: {top_contents(dense_only)}")
        w_ok = do_rank == 0 and w73_rank == 0
        print(f"  both rank target #1 ? {w_ok}")
        if not w_ok:
            print("WARN: weight variation unexpected.")
            sys.exit(1)
    finally:
        # Drop the test collection so re-runs start clean.
        try:
            cvdb._client.drop_collection(cvdb.final_namespace)
            print(f"\ndropped collection {cvdb.final_namespace}")
        except Exception as e:
            logger.warning(f"drop_collection failed: {e}")
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
