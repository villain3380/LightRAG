"""search_insight()：query 向量化 -> Milvus hybrid 检索 -> PG 查详情。

hybrid 检索 = dense(ANN) + sparse(ANN) + WeightedRanker 融合。
返回 insight 摘要（不含 content 全文，按需调 get_insight_content）。
"""
from pymilvus import AnnSearchRequest, WeightedRanker

from . import db
from .embedding import embed_summary
from .milvus_client import get_client, COLLECTION


async def search_insight(
    query: str,
    domain: str | None = None,
    top_k: int = 5,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[dict]:
    """语义检索 insight（dense+sparse hybrid）。

    Args:
        query: 查询文本
        domain: 过滤领域（如 financial_market/semiconductor），None=不过滤
        top_k: 返回条数
        dense_weight / sparse_weight: hybrid 融合权重

    Returns:
        [{id, title, summary, content_tokens, iv_grade, iv_desc, domain, tags,
          source_type, source_url, source_title, content_ref, score}, ...]
        不含 content 全文。按 score 降序。
    """
    # 1. query 向量化（dense+sparse）
    dense, sparse = await embed_summary(query)

    # 2. Milvus hybrid 检索（dense ANN + sparse ANN + WeightedRanker）
    client = get_client()
    req_dense = AnnSearchRequest(
        data=[dense], anns_field="vector",
        param={"metric_type": "COSINE"}, limit=top_k * 3,
    )
    req_sparse = AnnSearchRequest(
        data=[sparse], anns_field="sparse_vector",
        param={"metric_type": "IP"}, limit=top_k * 3,
    )
    filter_expr = f'domain == "{domain}"' if domain else ""
    results = client.hybrid_search(
        COLLECTION,
        [req_dense, req_sparse],
        WeightedRanker(dense_weight, sparse_weight),
        limit=top_k,
        filter=filter_expr,
        output_fields=["id"],
    )
    hits = results[0] if results else []
    ids = [hit["id"] for hit in hits]
    scores = {hit["id"]: hit.get("distance") for hit in hits}

    # 3. PG 查详情（不含 content 全文）+ 加 score
    briefs = await db.fetch_insight_brief(ids)
    for b in briefs:
        b["score"] = scores.get(b["id"])
    briefs.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    return briefs
