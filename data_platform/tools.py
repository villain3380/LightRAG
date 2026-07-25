"""本地 tool 封装（中台 agent 直接 import 调用）。

同步包装 async 函数（asyncio.run）。适合中台 agent（非 async 上下文）调用。
FastAPI 等异步上下文直接用 ingest.py/search.py 的 async 函数，不走这里。
"""
import asyncio

from .ingest import ingest_insight
from .search import search_insight
from .db import fetch_insight_content

# 持久 event loop：asyncio.run 每次新建 loop 会导致 db pool 跨 loop 失效，
# 用一个全局持久 loop，db pool 绑它，多次 tool 调用复用。
_loop = asyncio.new_event_loop()


def _run(coro):
    """在持久 loop 上跑协程（同步返回）。"""
    return _loop.run_until_complete(coro)


def ingest_insight_tool(params: dict) -> dict:
    """入库 insight。

    Args:
        params: ingest_insight 的参数 dict（title/summary/content或content_ref/iv_grade/domain/tags/...）

    Returns:
        {"id": int, "status": "ok"}
    """
    return _run(ingest_insight(**params))


def search_insight_tool(
    query: str, domain: str | None = None, top_k: int = 5,
) -> list[dict]:
    """语义检索 insight（返回摘要列表，不含 content 全文）。"""
    return _run(search_insight(query, domain, top_k))


def get_insight_content_tool(id: int) -> dict | None:
    """按需读 insight 原文（agent 深入时调）。"""
    return _run(fetch_insight_content(id))
