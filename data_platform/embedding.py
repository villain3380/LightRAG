"""复用 lightrag 的 dashscope_embed（dense+sparse，dim=1024）。

dashscope_embed 已被 @wrap_embedding_func_with_attrs 装饰成 EmbeddingFunc 实例
（supports_sparse=True），直接调 .aembed(with_sparse=True) 拿 EmbeddingResult。

api_key/model 从 lightrag-woo/.env 读（EMBEDDING_BINDING_API_KEY / DASHSCOPE_API_KEY / EMBEDDING_MODEL）。
"""
from pathlib import Path

from dotenv import load_dotenv

# 加载 lightrag-woo 根 .env（DashScope 配置）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from lightrag.llm.dashscope import dashscope_embed  # noqa: E402


async def embed_summary(summary: str) -> tuple[list[float], dict[int, float]]:
    """summary 向量化，返回 (dense, sparse)。

    Returns:
        dense: list[float]，dim=1024（result.dense[0].tolist()）
        sparse: {token_id: weight}（result.sparse[0]）
    """
    result = await dashscope_embed.aembed([summary], with_sparse=True)
    dense = result.dense[0].tolist()
    sparse = result.sparse[0] if result.sparse else {}
    return dense, sparse
