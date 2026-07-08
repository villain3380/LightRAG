"""Verify the project-integrated dashscope_embed binding (not the standalone probe).

Checks:
  1. dashscope_embed is an EmbeddingFunc with supports_sparse=True.
  2. __call__ (dense path, drop-in like openai_embed) -> np.ndarray (n, 1024).
  3. aembed(with_sparse=True) -> EmbeddingResult(dense, sparse).
  4. Dual scoring (dense cosine + sparse IP) ranks related > unrelated.

Run:  python examples/test_dashscope_binding.py
"""

import os
import sys
import asyncio

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from lightrag.llm.dashscope import dashscope_embed  # noqa: E402
from lightrag.utils import EmbeddingResult, cosine_similarity  # noqa: E402


def sparse_inner_product(q: dict[int, float], d: dict[int, float]) -> float:
    return sum(w * d[t] for t, w in q.items() if t in d)


CHUNK = (
    "LightRAG is a retrieval-augmented generation framework that builds a "
    "knowledge graph from documents. It extracts entities and relationships, "
    "then supports local, global, hybrid, and naive query modes."
)
Q_RELATED = "What query modes does LightRAG support?"
Q_UNRELATED = "How do I cook beef wellington?"


async def main():
    if not (os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
        print("API key not set")
        sys.exit(1)

    # 1) binding attributes
    print(f"supports_sparse    = {dashscope_embed.supports_sparse}")
    print(f"supports_asymmetric= {dashscope_embed.supports_asymmetric}")
    print(f"embedding_dim      = {dashscope_embed.embedding_dim}")
    print(f"model_name         = {dashscope_embed.model_name}")
    assert dashscope_embed.supports_sparse is True, "supports_sparse should be True"

    # 2) dense path via __call__ (drop-in, same form as openai_embed)
    d_doc = await dashscope_embed([CHUNK], context="document")
    assert isinstance(d_doc, np.ndarray), f"__call__ must return ndarray, got {type(d_doc)}"
    assert d_doc.shape == (1, dashscope_embed.embedding_dim), f"bad dense shape {d_doc.shape}"
    print(f"\n__call__  dense     -> {d_doc.shape}  (drop-in like openai_embed) OK")

    # 3) sparse path via aembed
    res_doc = await dashscope_embed.aembed([CHUNK], context="document", with_sparse=True)
    assert isinstance(res_doc, EmbeddingResult), f"aembed must return EmbeddingResult, got {type(res_doc)}"
    assert res_doc.sparse is not None and len(res_doc.sparse) == 1
    print(f"aembed    dense+sparse -> dense={res_doc.dense.shape}, "
          f"sparse tokens={len(res_doc.sparse[0])}, "
          f"sample={dict(list(res_doc.sparse[0].items())[:3])}")

    # 4) queries (query context) via aembed
    res_q = await dashscope_embed.aembed(
        [Q_RELATED, Q_UNRELATED], context="query", with_sparse=True
    )

    # 5) dual-path scoring
    print("\n=== dual-path scoring (query vs chunk) ===")
    scores = {}
    for label, qtext, dvec, svec in [
        ("RELATED  ", Q_RELATED, res_q.dense[0], res_q.sparse[0]),
        ("UNRELATED", Q_UNRELATED, res_q.dense[1], res_q.sparse[1]),
    ]:
        d_cos = float(cosine_similarity(dvec, res_doc.dense[0]))
        s_ip = sparse_inner_product(svec, res_doc.sparse[0])
        shared = set(svec) & set(res_doc.sparse[0])
        scores[label.strip().lower()] = (d_cos, s_ip)
        print(f"[{label}] '{qtext}'")
        print(f"    dense  cosine = {d_cos:.4f}")
        print(f"    sparse IP     = {s_ip:.4f}  (shared tokens: {len(shared)})")

    rel_cos, rel_ip = scores["related"]
    unrel_cos, unrel_ip = scores["unrelated"]
    print("\n=== sanity ===")
    ok_dense = rel_cos > unrel_cos
    ok_sparse = rel_ip > unrel_ip
    print(f"dense  : related({rel_cos:.4f}) > unrelated({unrel_cos:.4f}) ? {ok_dense}")
    print(f"sparse : related({rel_ip:.4f}) > unrelated({unrel_ip:.4f}) ? {ok_sparse}")
    if ok_dense and ok_sparse:
        print("PASS: binding works end-to-end (dense __call__ + sparse aembed).")
    else:
        print("WARN: expected related > unrelated on both paths.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
