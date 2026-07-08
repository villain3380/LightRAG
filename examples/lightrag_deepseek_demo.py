import os
import asyncio
import logging
import logging.config

# Load variables from .env into the environment so this standalone script
# reuses the same config as `lightrag-server` (LLM_BINDING, EMBEDDING_BINDING, ...).
# Requires the `python-dotenv` package (already a LightRAG dependency).
from dotenv import load_dotenv

load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import logger, set_verbose_debug, wrap_embedding_func_with_attrs

WORKING_DIR = "./dickens"

# ---------------------------------------------------------------------------
# Provider config — read straight from .env (same keys the API server uses).
# DeepSeek and Aliyun DashScope both speak the OpenAI-compatible protocol,
# so we reuse the generic openai_complete_if_cache / openai_embed adapters
# and just point them at the right base_url with the right api_key + model.
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("LLM_BINDING_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

DASHSCOPE_BASE_URL = os.getenv(
    "EMBEDDING_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY", "")
DASHSCOPE_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

# text-embedding-v4 default dimension is 1024.
DASHSCOPE_EMBED_DIM = 1024


async def deepseek_complete(prompt, system_prompt=None, history_messages=None,
                            keyword_extraction=False, entity_extraction=False, **kwargs):
    """LLM completion via DeepSeek (OpenAI-compatible)."""
    if history_messages is None:
        history_messages = []
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    return await openai_complete_if_cache(
        DEEPSEEK_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=DASHSCOPE_EMBED_DIM,
    max_token_size=8192,
)
async def dashscope_embed(texts: list[str]) -> "np.ndarray":
    """Embedding via Aliyun DashScope (OpenAI-compatible).

    NOTE: `openai_embed` is itself already wrapped by
    @wrap_embedding_func_with_attrs(embedding_dim=1536) — calling it directly
    would trigger its built-in 1536-dim validation and reject DashScope's
    1024-dim output. We call the unwrapped underlying via `.func` so only our
    own 1024-dim decorator validates the result (the @retry layer is preserved).
    """
    return await openai_embed.func(
        texts,
        model=DASHSCOPE_EMBED_MODEL,
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
        embedding_dim=DASHSCOPE_EMBED_DIM,
    )


def configure_logging():
    """Configure logging for the application"""

    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=dashscope_embed,
        llm_model_func=deepseek_complete,
        # Tell LightRAG which LLM model name is in use (shown in logs / cache keys).
        # Embedding dimension is declared via the @wrap_embedding_func_with_attrs
        # decorator on dashscope_embed above — no need to pass it here.
        llm_model_name=DEEPSEEK_MODEL,
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status

    return rag


async def main():
    # Sanity check: keys must be present in .env
    if not DEEPSEEK_API_KEY:
        print("Error: LLM_BINDING_API_KEY (DeepSeek) is not set. Check your .env.")
        return
    if not DASHSCOPE_API_KEY:
        print("Error: EMBEDDING_BINDING_API_KEY (DashScope) is not set. Check your .env.")
        return

    rag = None
    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function (DashScope)")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
