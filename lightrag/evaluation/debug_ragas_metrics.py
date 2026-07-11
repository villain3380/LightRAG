#!/usr/bin/env python3
"""
Step-by-step debug: isolate why answer_correctness / context_precision
return NaN with deepseek-v4-flash + RAGAS 0.4.3.

Each test runs a SINGLE metric in isolation on a tiny 1-row dataset.

Usage:
    uv run python lightrag/evaluation/debug_ragas_metrics.py
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── monkey-patch: RAGAS 0.4.3 needs embed_text / aembed_text ──────────────
if not hasattr(OpenAIEmbeddings, "embed_text"):
    OpenAIEmbeddings.embed_text = lambda self, text: self.embed_query(text)
if not hasattr(OpenAIEmbeddings, "embed_texts"):
    OpenAIEmbeddings.embed_texts = lambda self, texts: self.embed_documents(texts)
if not hasattr(OpenAIEmbeddings, "aembed_text"):
    OpenAIEmbeddings.aembed_text = lambda self, text: self.aembed_query(text)
if not hasattr(OpenAIEmbeddings, "aembed_texts"):
    OpenAIEmbeddings.aembed_texts = lambda self, texts: self.aembed_documents(texts)

# ── shared config ──────────────────────────────────────────────────────────
MODEL      = os.getenv("EVAL_LLM_MODEL", "deepseek-v4-flash")
LLM_KEY    = os.getenv("EVAL_LLM_BINDING_API_KEY", "")
LLM_BASE   = os.getenv("EVAL_LLM_BINDING_HOST", "")
TIMEOUT    = int(os.getenv("EVAL_LLM_TIMEOUT", "600"))
EMB_MODEL  = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-v4")
EMB_KEY    = os.getenv("EVAL_EMBEDDING_BINDING_API_KEY", "")
EMB_BASE   = os.getenv("EVAL_EMBEDDING_BINDING_HOST", "")

# tiny 1-row dataset
DS = Dataset.from_dict({
    "question":      ["Haikuo Shao有哪些论文"],
    "answer":        ["APT-LLM和ASTRA的作者包括Haikuo Shao。"],
    "contexts":      [["APT-LLM作者包括Haikuo Shao等人。", "ASTRA作者包括Haikuo Shao。", "AccLLM第一作者H. Shao。"]],
    "ground_truth":  ["AccLLM, APT-LLM, ASTRA, Trio-ViT等。"],
})


def make_llm():
    base = ChatOpenAI(
        model=MODEL, api_key=LLM_KEY, base_url=LLM_BASE,
        temperature=0, request_timeout=TIMEOUT, max_retries=1,
    )
    return LangchainLLMWrapper(langchain_llm=base, bypass_n=True)


def make_openai_emb():
    """OpenAIEmbeddings via dashscope compatible-mode (known working baseline)."""
    return OpenAIEmbeddings(
        model=EMB_MODEL, api_key=EMB_KEY, base_url=EMB_BASE,
        check_embedding_ctx_length=False,
    )


def make_dashscope_emb():
    """Our custom _DashScopeEmbeddings (async via thread bridge)."""
    from lightrag.evaluation.eval_woo import _DashScopeEmbeddings
    return _DashScopeEmbeddings(api_key=EMB_KEY, model=EMB_MODEL)


def run_one(metric, llm, emb, label: str) -> float:
    """Run a single metric on the 1-row dataset, return its float score."""
    print(f"\n  [{label}] ...", end="", flush=True)
    try:
        result = evaluate(
            dataset=DS,
            metrics=[metric],
            llm=llm,
            embeddings=emb,
        )
        df = result.to_pandas()
        score = float(df.iloc[0].iloc[-1])  # last column = the metric
        tag = f"{score:.4f}" if score == score else "NaN"
        print(f" => {tag}")
        return score
    except Exception as e:
        print(f" => ERROR: {type(e).__name__}: {e}")
        return float("nan")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  LLM: {MODEL} @ {LLM_BASE}")
    print(f"  EMB: {EMB_MODEL}")
    print("=" * 60)

    llm = make_llm()

    # ── Step 1: baseline with OpenAIEmbeddings ──
    print("\n── Step 1: OpenAIEmbeddings baseline ──")
    emb_oai = make_openai_emb()

    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision,
    )
    from ragas.metrics._answer_correctness import AnswerCorrectness

    run_one(Faithfulness(llm=llm), llm, emb_oai, "Faithfulness")
    run_one(AnswerRelevancy(llm=llm, embeddings=emb_oai), llm, emb_oai, "AnswerRelevancy")
    run_one(ContextRecall(llm=llm), llm, emb_oai, "ContextRecall")
    run_one(ContextPrecision(llm=llm), llm, emb_oai, "ContextPrecision")
    # answer_correctness needs BOTH llm + embeddings
    s5 = run_one(AnswerCorrectness(llm=llm, embeddings=emb_oai), llm, emb_oai, "AnswerCorrectness")

    if s5 != s5:  # NaN
        print("\n── Step 1b: AnswerCorrectness fails with OpenAIEmbeddings too ──")
        print("  This means the problem is NOT in _DashScopeEmbeddings.")
        print("  It's a RAGAS 0.4.3 + deepseek compatibility issue.")
        print("  Let's try without LangchainLLMWrapper...")

        # Try with raw ChatOpenAI directly
        base_llm = ChatOpenAI(
            model=MODEL, api_key=LLM_KEY, base_url=LLM_BASE,
            temperature=0, request_timeout=TIMEOUT, max_retries=1,
        )
        from ragas.llms import llm_factory as ragas_llm_factory

        # RAGAS 0.4.3 new API: use llm_factory with an openai client
        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE)
            new_llm = ragas_llm_factory(MODEL, client=client)
            print(f"  new_llm type: {type(new_llm).__name__}")

            run_one(
                AnswerCorrectness(llm=new_llm, embeddings=emb_oai),
                new_llm, emb_oai,
                "AnswerCorrectness (llm_factory + OpenAIEmbeddings)",
            )
        except Exception as e:
            print(f"  llm_factory failed: {e}")

        # Try answer_similarity instead (embeddings-only, no LLM needed)
        print("\n── Step 1c: answer_similarity (embeddings-only, no LLM) ──")
        from ragas.metrics import AnswerSimilarity
        run_one(
            AnswerSimilarity(embeddings=emb_oai),
            llm, emb_oai, "AnswerSimilarity",
        )

    # ── Step 2: with _DashScopeEmbeddings ──
    print("\n── Step 2: _DashScopeEmbeddings ──")
    emb_ds = make_dashscope_emb()
    print(f"  isinstance(Embeddings): {_is_lc_emb(emb_ds)}")
    run_one(Faithfulness(llm=llm), llm, emb_ds, "Faithfulness")
    run_one(ContextRecall(llm=llm), llm, emb_ds, "ContextRecall")
    run_one(ContextPrecision(llm=llm), llm, emb_ds, "ContextPrecision")
    run_one(AnswerCorrectness(llm=llm, embeddings=emb_ds), llm, emb_ds, "AnswerCorrectness")

    # ── Step 3: manual LLM prompt test ──
    print("\n── Step 3: raw LLM prompt (bypass RAGAS) ──")
    base = ChatOpenAI(
        model=MODEL, api_key=LLM_KEY, base_url=LLM_BASE,
        temperature=0, request_timeout=TIMEOUT,
    )
    prompt = (
        "Compare the student answer to the ground truth and rate correctness "
        "from 0 to 1.\n\n"
        f"Question: Haikuo Shao有哪些论文\n"
        f"Student Answer: APT-LLM和ASTRA的作者包括Haikuo Shao。\n"
        f"Ground Truth: AccLLM, APT-LLM, ASTRA, Trio-ViT等。\n\n"
        "Reply with ONLY a JSON object: {\"score\": <0-1>, \"reason\": \"...\"}"
    )
    try:
        resp = base.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        obj = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        print(f"  score={obj['score']}, reason={obj['reason'][:120]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


def _is_lc_emb(obj) -> bool:
    try:
        from langchain_core.embeddings import Embeddings
        return isinstance(obj, Embeddings)
    except ImportError:
        return False


if __name__ == "__main__":
    main()
