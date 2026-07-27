"""FastAPI HTTP API（前端/其他服务调用）。

启动：python -m uvicorn data_platform.main:app --reload --port 9700
（先设环境变量 INSIGHT_WRITER_PWD=iw_2026）

接口约定见 CONTRACT.md。
"""
from fastapi import FastAPI
from pydantic import BaseModel

from .ingest import ingest_insight
from .search import search_insight
from .db import fetch_insight_content, update_insight_pg

app = FastAPI(title="data_platform", version="0.1.0")


class InsightIn(BaseModel):
    title: str
    summary: str
    content: str | None = None
    content_ref: str | None = None
    iv_grade: str = "S2"
    iv_desc: str | None = None
    domain: str = "other"
    tags: list[str] | None = None
    source_type: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    raw_meta: dict | None = None


class SearchIn(BaseModel):
    query: str
    domain: str | None = None
    top_k: int = 5


@app.post("/ingest/insight")
async def ingest_insight_api(req: InsightIn):
    """入库 insight（参数化 + 向量化 summary + 写 PG/Milvus）。"""
    return await ingest_insight(**req.model_dump())


@app.post("/insight/search")
async def search_insight_api(req: SearchIn):
    """语义检索 insight（hybrid，返回摘要列表）。"""
    return await search_insight(req.query, req.domain, req.top_k)


@app.get("/insight/{insight_id}/content")
async def get_content_api(insight_id: int):
    """按需读 insight 原文。"""
    row = await fetch_insight_content(insight_id)
    return row if row else {"error": "not found", "id": insight_id}


class UpdateInsightIn(BaseModel):
    where: dict = {}
    set: dict = {}


@app.put("/insight/update")
async def update_insight_api(req: UpdateInsightIn):
    """批量更新 insight。where 条件 + set 字段（白名单）。"""
    try:
        return await update_insight_pg(req.where, req.set)
    except ValueError as e:
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}
