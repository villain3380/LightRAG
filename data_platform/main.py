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
from .db_emfetch import em_fetch_ingest_pg
from .db_todo import insert_todo_pg, list_todo_pg, get_todo_pg, update_todo_pg, todo_search_pg

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


@app.post("/ingest/em_fetch")
async def ingest_em_fetch_api(data: dict):
    """批量入库 em-fetch JSON（10 种数据类型，market_data + event_data）。"""
    try:
        return await em_fetch_ingest_pg(data)
    except Exception as e:
        return {"error": str(e)}


# ===== todo =====


class TodoCreateIn(BaseModel):
    title: str
    detail: str | None = None
    priority: str = "P2"
    due_date: str | None = None  # 'YYYY-MM-DD'
    tags: list[str] | None = None
    domain: str | None = None
    sort_order: int = 0
    raw_meta: dict | None = None


class TodoUpdateIn(BaseModel):
    set: dict


@app.post("/todo/")
async def create_todo_api(req: TodoCreateIn):
    """新建 todo（status 强制 todo；completed_at 由 trigger 完成时填）。"""
    try:
        tid = await insert_todo_pg(**req.model_dump())
        return {"id": tid}
    except Exception as e:
        return {"error": str(e)}


@app.get("/todo/")
async def list_todo_api(
    status: str | None = None,
    priority: str | None = None,
    domain: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """列表查询 todo（按 priority P0->P3, due_date, created_at 排序）。"""
    return await list_todo_pg(status, priority, domain, limit, offset)


@app.get("/todo/{todo_id}")
async def get_todo_api(todo_id: int):
    """取单条 todo（含 detail + raw_meta 全字段）。"""
    row = await get_todo_pg(todo_id)
    return row if row else {"error": "not found", "id": todo_id}


@app.put("/todo/{todo_id}")
async def update_todo_api(todo_id: int, req: TodoUpdateIn):
    """更新 todo（白名单字段）。trigger 自动管 updated_at + completed_at。"""
    try:
        return await update_todo_pg(todo_id, req.set)
    except ValueError as e:
        return {"error": str(e)}


@app.put("/todo/{todo_id}/complete")
async def complete_todo_api(todo_id: int):
    """便捷：标记完成（trigger 自动填 completed_at）。"""
    return await update_todo_pg(todo_id, {"status": "done"})


class TodoSearchIn(BaseModel):
    query: str
    top_k: int = 5


@app.post("/todo/search")
async def search_todo_api(req: TodoSearchIn):
    """语义搜索 todo（用模糊复述找某条待办，按相似度返回）。"""
    return await todo_search_pg(req.query, req.top_k)


@app.get("/health")
async def health():
    return {"status": "ok"}
