"""Standalone probe: confirm DashScope text-embedding-v4 can return BOTH dense
and sparse vectors, and that dense cosine + sparse inner-product scoring behave
sanely on a related vs unrelated query.

KEY FINDING (see examples/probe_sparse_api_shape.py):
  The OpenAI-compatible endpoint (/compatible-mode/v1/embeddings) silently
  ignores `output_type` and returns dense ONLY. Sparse requires the DashScope
  NATIVE API (/api/v1/services/embeddings/text-embedding/text-embedding) with
  `parameters.output_type`. So a dense+sparse binding must be a native-API
  client, NOT a tweak of openai_embed.

This script does NOT modify any lightrag project source. It imports only
`cosine_similarity` from lightrag.utils (same math the project uses).

Contract under test (the new dense+sparse binding prototype):
    embed_dense_sparse(texts) -> (dense: np.ndarray (n, dim),
                                  sparse: list[dict[int, float]])
where each sparse dict is {token_id: weight}.

Run:  python examples/probe_dense_sparse_embed.py
"""

import os
import sys
import math
import asyncio

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import httpx

from lightrag.utils import cosine_similarity

EMBED_API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

NATIVE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "text-embedding/text-embedding"
)


def _normalize_sparse(sp) -> dict[int, float]:
    """Reduce any sparse shape to {token_id(int): weight(float)}.

    DashScope native format: [{"index": int, "token": str, "value": float}, ...].
    Also handles {indices,values} / {id:weight} / [[id,weight],..] for portability.
    """
    if not sp:
        return {}
    if isinstance(sp, list):
        out: dict[int, float] = {}
        for item in sp:
            if isinstance(item, dict) and "index" in item and "value" in item:
                out[int(item["index"])] = float(item["value"])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out[int(item[0])] = float(item[1])
        return out
    if isinstance(sp, dict):
        if "indices" in sp and "values" in sp:
            return {int(i): float(v) for i, v in zip(sp["indices"], sp["values"])}
        return {int(k): float(v) for k, v in sp.items()}
    return {}


async def embed_dense_sparse(
    texts: list[str], client: httpx.AsyncClient
) -> tuple[np.ndarray, list[dict[int, float]]]:
    """Prototype of the new DashScope native dense+sparse binding.

    One call with output_type=dense&sparse yields both. Results are sorted by
    text_index to preserve input order (DashScope does not guarantee order).
    """
    body = {
        "model": EMBED_MODEL,
        "input": {"texts": texts},
        "parameters": {"output_type": "dense&sparse", "dimension": EMBED_DIM},
    }
    headers = {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json",
    }
    r = await client.post(NATIVE_URL, json=body, headers=headers)
    r.raise_for_status()
    j = r.json()
    embs = j["output"]["embeddings"]
    embs.sort(key=lambda e: e.get("text_index", 0))

    dense = np.array([e["embedding"] for e in embs], dtype=np.float32)
    sparse = [_normalize_sparse(e.get("sparse_embedding")) for e in embs]
    return dense, sparse


def sparse_inner_product(q: dict[int, float], d: dict[int, float]) -> float:
    """Inner product over shared token ids.

    Standard sparse-retrieval score (Milvus SPARSE_FLOAT_VECTOR IP metric /
    BGE-M3 sparse retrieval both use this).
    """
    return sum(w * d[t] for t, w in q.items() if t in d)


def sparse_cosine(q: dict[int, float], d: dict[int, float]) -> float:
    if not q or not d:
        return 0.0
    dot = sparse_inner_product(q, d)
    nq = math.sqrt(sum(w * w for w in q.values()))
    nd = math.sqrt(sum(w * w for w in d.values()))
    return dot / (nq * nd) if nq and nd else 0.0


CHUNK = (
    "LightRAG is a retrieval-augmented generation framework that builds a "
    "knowledge graph from documents. It extracts entities and relationships, "
    "then supports local, global, hybrid, and naive query modes."
)
Q_RELATED = "What query modes does LightRAG support?"
Q_UNRELATED = "How do I cook beef wellington?"


async def main():
    if not EMBED_API_KEY:
        print("EMBEDDING_BINDING_API_KEY not set in .env")
        sys.exit(1)
    print(f"model={EMBED_MODEL} dim={EMBED_DIM}")
    print(f"endpoint=NATIVE ({NATIVE_URL})")
    print(f"chunk ({len(CHUNK)} chars): {CHUNK[:80]}...")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1) embed the chunk (document side)
        d_doc, s_doc = await embed_dense_sparse([CHUNK], client)
        print(
            f"\nchunk dense: shape={d_doc.shape}  "
            f"sparse: {len(s_doc[0])} tokens, "
            f"sample={dict(list(s_doc[0].items())[:5])}"
        )

        # 2) embed two queries (query side)
        d_q, s_q = await embed_dense_sparse([Q_RELATED, Q_UNRELATED], client)

        # 3) dual-path scoring
        print("\n=== dual-path scoring (query vs chunk) ===")
        results = {}
        for label, qtext, dvec, svec in [
            ("RELATED  ", Q_RELATED, d_q[0], s_q[0]),
            ("UNRELATED", Q_UNRELATED, d_q[1], s_q[1]),
        ]:
            d_cos = float(cosine_similarity(dvec, d_doc[0]))
            s_ip = sparse_inner_product(svec, s_doc[0])
            s_cos = sparse_cosine(svec, s_doc[0])
            shared = set(svec) & set(s_doc[0])
            results[label.strip().lower()] = (d_cos, s_ip)
            print(f"[{label}] '{qtext}'")
            print(f"    dense  cosine     = {d_cos:.4f}")
            print(f"    sparse IP          = {s_ip:.4f}  (shared tokens: {len(shared)})")
            print(f"    sparse cosine      = {s_cos:.4f}")

        # 4) sanity check
        rel_cos, rel_ip = results["related"]
        unrel_cos, unrel_ip = results["unrelated"]
        print("\n=== sanity ===")
        print(f"dense  : related({rel_cos:.4f}) > unrelated({unrel_cos:.4f}) ? {rel_cos > unrel_cos}")
        print(f"sparse : related({rel_ip:.4f}) > unrelated({unrel_ip:.4f}) ? {rel_ip > unrel_ip}")
        if rel_cos > unrel_cos and rel_ip > unrel_ip:
            print("PASS: both dense and sparse rank related > unrelated.")
        else:
            print("WARN: expected related > unrelated on BOTH paths - inspect above.")


if __name__ == "__main__":
    asyncio.run(main())
