"""Milvus 连接 + insight 向量入库（lightrag db）。

collection: insight_text_embedding_v4_1024d_sparse
字段: id(INT64) / vector(FLOAT_VECTOR 1024) / sparse_vector(SPARSE_FLOAT_VECTOR) / domain / iv_grade
索引: vector AUTOINDEX/COSINE, sparse_vector SPARSE_INVERTED_INDEX/IP

pymilvus 3.0 MilvusClient API（无 ORM deprecation）。
"""
import os
from pymilvus import MilvusClient, DataType

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


# ===== todo collection（title+detail 语义检索，dense only） =====

TODO_COLLECTION = "todo_detail_embedding_v4_1024d"


def ensure_todo_collection() -> None:
    """建 todo collection（id + vector(1024) + status）。幂等，首次插入/搜索前调用。

    id = PG shared.todo.id（主键，关联 PG）；status 冗余存一份便于检索时按状态过滤。
    """
    client = get_client()
    if client.has_collection(TODO_COLLECTION):
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("status", DataType.VARCHAR, max_length=20)
    idx = client.prepare_index_params()
    idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(TODO_COLLECTION, schema=schema, index_params=idx)


def upsert_todo_vector(id: int, dense: list[float], status: str) -> None:
    """upsert 一条 todo 向量（id 主键，重复则覆盖；update detail 后重新 embed 调这个）。"""
    client = get_client()
    client.upsert(
        TODO_COLLECTION,
        [{"id": id, "vector": dense, "status": status}],
    )


def search_todo_vector(dense: list[float], top_k: int = 5) -> list[dict]:
    """dense ANN 搜索 todo。返回 [{id, distance, status}]，按相似度降序。"""
    client = get_client()
    results = client.search(
        TODO_COLLECTION,
        data=[dense],
        anns_field="vector",
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        output_fields=["status"],
    )
    hits = results[0] if results else []
    return [
        {"id": h["id"], "distance": h.get("distance"),
         "status": (h.get("entity") or {}).get("status")}
        for h in hits
    ]


def delete_todo_vector(ids: list[int]) -> None:
    """删除 todo 向量（物理删 PG todo 时同步调，避免 Milvus 孤儿向量）。

    注意：todo_writer 不能 DELETE shared.todo（软删除用 status='cancelled'），
    所以正常流程不会触发；此函数供 rag owner 物理清理时使用。
    """
    if not ids:
        return
    client = get_client()
    client.delete(TODO_COLLECTION, filter=f"id in {ids}")
