"""ingest_insight() 核心：编排 db + embedding + milvus。

接口约定见 CONTRACT.md。
"""
from . import db
from .embedding import embed_summary
from .milvus_client import insert_insight_vector


def _load_content_ref(content_ref: str) -> str:
    """读 content_ref 指向的文件，返回文本内容。"""
    with open(content_ref, "r", encoding="utf-8") as f:
        return f.read()


def _count_tokens(text: str) -> int:
    """粗略估算 token 数（中文 1 字 ≈ 1-2 token，用字符数近似）。

    后续可换 tiktoken 精确计算（和 LLM 对齐）。
    """
    return len(text)


async def ingest_insight(
    title: str,
    summary: str,
    content: str | None = None,
    content_ref: str | None = None,
    iv_grade: str = "S2",
    iv_desc: str | None = None,
    domain: str = "other",
    tags: list[str] | None = None,
    source_type: str | None = None,
    source_url: str | None = None,
    source_title: str | None = None,
    raw_meta: dict | None = None,
) -> dict:
    """入库一条 insight（PG shared.insight + Milvus insight collection）。

    流程见 CONTRACT.md。返回 {"id": int, "status": "ok"}。

    Raises:
        ValueError: content/content_ref 都没给；iv_grade/domain 不在白名单。
    """
    # 1. 解析 content：content_ref 有值则加载文件
    if content_ref:
        content = _load_content_ref(content_ref)
    if not content:
        raise ValueError("content 和 content_ref 至少给一个")

    # 2. 算 content_tokens
    content_tokens = _count_tokens(content)

    # 3. 校验白名单
    if iv_grade not in db.VALID_GRADES:
        raise ValueError(f"iv_grade 非法: {iv_grade}，合法: {db.VALID_GRADES}")
    if domain not in db.VALID_DOMAINS:
        raise ValueError(f"domain 非法: {domain}，合法: {db.VALID_DOMAINS}")

    # 4. 参数化 INSERT PG -> 拿 id
    insight_id = await db.insert_insight_pg(
        title=title, summary=summary, content=content, content_tokens=content_tokens,
        iv_grade=iv_grade, iv_desc=iv_desc, domain=domain, tags=tags,
        source_type=source_type, source_url=source_url, source_title=source_title,
        raw_meta=raw_meta, content_ref=content_ref,
    )

    # 5. summary 向量化（dense+sparse）
    dense, sparse = await embed_summary(summary)

    # 6. INSERT Milvus insight collection
    insert_insight_vector(
        id=insight_id, dense=dense, sparse=sparse, domain=domain, iv_grade=iv_grade,
    )

    # 7. 返回
    return {"id": insight_id, "status": "ok"}
