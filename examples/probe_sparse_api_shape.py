"""Focused probe: find where DashScope v4 actually returns sparse vectors.

Tries multiple endpoints / param placements and dumps the raw response keys so
we can see which one yields a sparse field.

Run:  python examples/probe_sparse_api_shape.py
"""

import os
import json
import asyncio

from dotenv import load_dotenv

load_dotenv()

import httpx

EMBED_API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
COMPAT_BASE = os.getenv(
    "EMBEDDING_BINDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
NATIVE_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

TEXT = "LightRAG supports local global hybrid and naive query modes."


def _find_sparse_keys(data_item: dict) -> list[str]:
    return [k for k in data_item if k != "embedding" and "sparse" in k.lower()]


async def try_compat(client: httpx.AsyncClient, output_type: str):
    """OpenAI-compatible endpoint with extra output_type."""
    url = COMPAT_BASE.rstrip("/") + "/embeddings"
    body = {
        "model": EMBED_MODEL,
        "input": [TEXT],
        "encoding_format": "float",
        "output_type": output_type,
    }
    r = await client.post(url, json=body, headers=_headers())
    print(f"\n[compat] output_type={output_type!r} -> HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"  body: {r.text[:400]}")
        return
    j = r.json()
    d0 = j.get("data", [{}])[0]
    print(f"  top keys: {list(j.keys())}")
    print(f"  data[0] keys: {list(d0.keys())}")
    sp = _find_sparse_keys(d0)
    print(f"  sparse keys: {sp}")
    if sp:
        print(f"  sample sparse: {str(d0[sp[0]])[:200]}")


async def try_native(client: httpx.AsyncClient, output_type: str, dimension: int | None):
    """DashScope native text-embedding API with parameters.output_type."""
    params = {"output_type": output_type}
    if dimension is not None:
        params["dimension"] = dimension
    body = {
        "model": EMBED_MODEL,
        "input": {"texts": [TEXT]},
        "parameters": params,
    }
    r = await client.post(NATIVE_URL, json=body, headers=_headers())
    print(f"\n[native] output_type={output_type!r} dimension={dimension} -> HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"  body: {r.text[:400]}")
        return
    j = r.json()
    print(f"  top keys: {list(j.keys())}")
    out = j.get("output", {})
    embs = out.get("embeddings", [])
    print(f"  output keys: {list(out.keys())}")
    if embs:
        d0 = embs[0]
        print(f"  embeddings[0] keys: {list(d0.keys())}")
        sp = _find_sparse_keys(d0)
        print(f"  sparse keys: {sp}")
        if sp:
            print(f"  sample sparse: {str(d0[sp[0]])[:200]}")
        if "embedding" in d0:
            print(f"  dense len: {len(d0['embedding'])}")


def _headers():
    return {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json",
    }


async def main():
    if not EMBED_API_KEY:
        print("EMBEDDING_BINDING_API_KEY not set"); raise SystemExit(1)
    print(f"model={EMBED_MODEL}")
    async with httpx.AsyncClient(timeout=60.0) as c:
        # OpenAI-compatible attempts
        await try_compat(c, "sparse")
        await try_compat(c, "dense&sparse")
        # Native API attempts
        await try_native(c, "sparse", None)
        await try_native(c, "dense&sparse", None)
        await try_native(c, "dense&sparse", 1024)


if __name__ == "__main__":
    asyncio.run(main())
