"""Milvus 连接 + insight 向量入库（lightrag db）。

collection: insight_text_embedding_v4_1024d_sparse
字段: id(INT64) / vector(FLOAT_VECTOR 1024) / sparse_vector(SPARSE_FLOAT_VECTOR) / domain / iv_grade
索引: vector AUTOINDEX/COSINE, sparse_vector SPARSE_INVERTED_INDEX/IP

pymilvus 3.0 MilvusClient API（无 ORM deprecation）。
"""
import os
from pymilvus import MilvusClient

COLLECTION = "insight_text_embedding_v4_1024d_sparse"
_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    """全局 MilvusClient（连 lightrag db，复用 .env MILVUS_DB_NAME=lightrag）。"""
    global _client
    if _client is None:
        _client = MilvusClient(
            uri=os.environ.get("DP_MILVUS_URI", "http://localhost:19530"),
            db_name=os.environ.get("DP_MILVUS_DB", "lightrag"),
        )
    return _client


def insert_insight_vector(
    id: int,
    dense: list[float],
    sparse: dict[int, float],
    domain: str,
    iv_grade: str,
) -> None:
    """插入一条 insight 向量（dense+sparse + 过滤字段）。

    Args:
        id: = PG shared.insight.id（主键，关联 PG）
        dense: dense 向量 list[float]（dim=1024，来自 aembed result.dense[0]）
        sparse: sparse 向量 {token_id: weight}（来自 aembed result.sparse[0]）
        domain / iv_grade: 过滤字段（检索时 expr 用）
    """
    client = get_client()
    client.insert(
        COLLECTION,
        [{
            "id": id,
            "vector": dense,
            "sparse_vector": sparse,
            "domain": domain,
            "iv_grade": iv_grade,
        }],
    )
