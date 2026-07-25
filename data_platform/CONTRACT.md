# data_platform 接口约定（CONTRACT）

> 数据中台后端。本文件是模块接口契约，改接口先改这里。
> 配套表结构见 [../woo_doc/数据中台结构.md](../woo_doc/数据中台结构.md)，DDL 见 [../woo_doc/init_data_platform.sql](../woo_doc/init_data_platform.sql)。

## 模块职责

data_platform 是数据中台后端，提供 insight 入库/检索 + 行情入库（em_fetch）+ 交易记录。
对外提供三种接入（同一套核心逻辑）：
- **本地 tool**（`tools.py`）：中台 agent 直接 import 调用
- **HTTP API**（`main.py` FastAPI）：前端/其他服务
- **MCP server**（`mcp_server.py`，后续）：外部 agent 如 pi-agent

当前不做：mcp_server、auth（后续加）。

## 配置

### PostgreSQL
- host=localhost port=5432 database=rag
- 账号（最小权限，密码走环境变量）：
  - `insight_writer` / `INSIGHT_WRITER_PWD`（shared.insight INSERT/SELECT）
  - `market_data_writer` / `MARKET_DATA_WRITER_PWD`（market_data + event_data INSERT/SELECT）
  - `trading_writer` / `TRADING_WRITER_PWD`（trading INSERT/SELECT）

### Milvus
- uri=http://localhost:19530，db_name=lightrag（复用 .env MILVUS_DB_NAME）
- insight collection：`insight_text_embedding_v4_1024d_sparse`
  - 字段：id(INT64) / vector(FLOAT_VECTOR 1024) / sparse_vector(SPARSE_FLOAT_VECTOR) / domain(VARCHAR) / iv_grade(VARCHAR)
  - 索引：vector AUTOINDEX/COSINE，sparse_vector SPARSE_INVERTED_INDEX/IP

### embedding
- 复用 lightrag 的 `aembed`（DashScope text-embedding-v4，dim=1024，dense+sparse）
- 向量化对象：insight 的 **summary**（不是 content 全文）

## 接口

### `ingest_insight()` - 入库一条 insight
```python
async def ingest_insight(
    title: str,
    summary: str,
    content: str | None = None,        # 直接文本
    content_ref: str | None = None,    # 文件引用（有值则加载文件填 content）
    iv_grade: str,                     # S2/S1/A/B/C/D
    iv_desc: str | None = None,
    domain: str,                       # 见 VALID_DOMAINS
    tags: list[str] | None = None,
    source_type: str | None = None,
    source_url: str | None = None,
    source_title: str | None = None,
    raw_meta: dict | None = None,
) -> dict:
    """
    流程：
    1. content 或 content_ref 至少给一个；content_ref 有值 -> 读文件填 content
    2. 算 content_tokens
    3. 校验 iv_grade ∈ VALID_GRADES、domain ∈ VALID_DOMAINS
    4. 参数化 INSERT shared.insight (insight_writer) -> 拿 id
    5. summary 向量化 (aembed -> dense+sparse)
    6. INSERT Milvus insight collection (id, vector, sparse_vector, domain, iv_grade)
    7. 返回 {"id": int, "status": "ok"}
    """
```

### `search_insight()` - 语义检索（后续）
```python
async def search_insight(
    query: str,
    domain: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    query 向量化 -> 查 Milvus insight collection（domain 过滤 + hybrid）
    返回 [{id, title, summary, content_tokens, iv_grade, domain, tags, score}, ...]
    不含 content 全文（agent 按需调 get_insight_content）。
    """
```

### `get_insight_content()` - 按需读原文（后续）
```python
async def get_insight_content(id: int) -> dict:
    """返回 {id, content}，agent 深入原文时调。"""
```

## HTTP API（FastAPI，`main.py`）
- `POST /ingest/insight` -> ingest_insight
- `POST /insight/search` -> search_insight
- `GET  /insight/{id}/content` -> get_insight_content

## 本地 tool（`tools.py`，中台 agent 调）
- `ingest_insight_tool(params: dict) -> dict`
- `search_insight_tool(query: str, domain: str|None, top_k: int) -> list[dict]`

## 白名单
- `VALID_GRADES = {"S2","S1","A","B","C","D"}`
- `VALID_DOMAINS = {"financial_market", "financial_market/semiconductor", "financial_market/banking", "financial_market/real_estate", "technology", "policy", "daily_life", "other"}`

## 文件结构
```
data_platform/
├── CONTRACT.md          # 本文件（接口约定）
├── __init__.py
├── db.py                # PG 连接池 + insert_insight_pg
├── milvus_client.py     # Milvus 连接 + insert_insight_vector
├── embedding.py         # 复用 lightrag aembed，返回 dense+sparse
├── ingest.py            # ingest_insight() 核心（编排 db + embedding + milvus）
├── search.py            # search_insight()（后续）
├── tools.py             # 本地 tool 封装（后续）
└── main.py              # FastAPI HTTP API（后续）
```
