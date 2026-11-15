"""
Module B Demo (Langfuse): Evaluation Dataset & A/B Experiment
-------------------------------------------------------------
Mirrors module_b_evaluation/demo.py using Langfuse Datasets & Experiments:
  - Creates/updates the 'fintech-demo-hill-climb' dataset in Langfuse (8 examples)
  - Runs Experiment A (v1-baseline: chunk_size=100, top_k=1)
  - Runs Experiment B (v2-improved: chunk_size=1500, top_k=1)
  - Evaluates both with routing_accuracy and keyword_correctness
"""

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Evaluation, get_client

from eval_dataset import DEMO_HC_DATASET_NAME, HILL_CLIMB_EXAMPLES

load_dotenv()

# chromadb's posthog telemetry client is incompatible with posthog>=3 — silence it
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import ask, build_support_agent


def routing_evaluator(*, output, expected_output, **kwargs) -> Evaluation:
    """Check if the supervisor routed to the correct agent."""
    predicted = (output or {}).get("intent", "")
    expected = (expected_output or {}).get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return Evaluation(name="routing_accuracy", value=score)


def keyword_correctness(*, output, expected_output, **kwargs) -> Evaluation:
    """Check if key numbers/dollar amounts from expected_output appear in output."""
    actual = (output or {}).get("answer", "").lower()
    expected = (expected_output or {}).get("answer", "").lower()

    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+", expected)
    if not key_terms:
        return Evaluation(name="keyword_correctness", value=0.5)

    matches = sum(1 for term in key_terms if term in actual)
    return Evaluation(name="keyword_correctness", value=round(matches / len(key_terms), 4))


def _ensure_dataset(langfuse):
    """Create the Langfuse dataset and populate items if empty."""
    langfuse.create_dataset(
        name=DEMO_HC_DATASET_NAME,
        description=(
            "Policy-focused evaluation examples for hill climbing experiments. "
            "All 8 questions require precise factual answers sensitive to retrieval quality."
        ),
    )
    dataset = langfuse.get_dataset(DEMO_HC_DATASET_NAME)
    if len(dataset.items) == 0:
        for idx, ex in enumerate(HILL_CLIMB_EXAMPLES):
            langfuse.create_dataset_item(
                dataset_name=DEMO_HC_DATASET_NAME,
                id=f"{DEMO_HC_DATASET_NAME}-{idx}",
                input=ex["inputs"],
                expected_output=ex["outputs"],
            )
        dataset = langfuse.get_dataset(DEMO_HC_DATASET_NAME)
    return dataset


def _make_task(app):
    """Return a Langfuse experiment task function wrapping the compiled graph."""
    def task(*, item, **kwargs):
        result = ask(app, item.input["question"])
        return {
            "answer": result["response"],
            "intent": result["intent"],
            "retrieved_sources": result["retrieved_sources"],
            "context": result["context"],
        }
    return task


def main() -> None:
    langfuse = get_client()
    dataset = _ensure_dataset(langfuse)
    print(f"Langfuse dataset '{DEMO_HC_DATASET_NAME}' ready with {len(dataset.items)} items.\n")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print("Building FinTech support agent (v1 — chunk_size=100, top_k=1)...")
    agent_v1 = build_support_agent(
        collection_name="langfuse_eval_v1", chunk_size=100, chunk_overlap=0, top_k=1
    )
    print("Running Langfuse Experiment A (v1-baseline)...")
    res_a = dataset.run_experiment(
        name="demo-v1-baseline",
        run_name=f"demo-v1-baseline-{stamp}",
        description="Baseline: chunk_size=100, top_k=1",
        task=_make_task(agent_v1["app"]),
        evaluators=[routing_evaluator, keyword_correctness],
        max_concurrency=4,
        metadata={"model": "gpt-4o-mini", "version": "baseline", "chunk_size": 100, "top_k": 1},
    )
    print(res_a.format(include_item_results=False))

    print("\nBuilding improved agent (v2 — chunk_size=1500, top_k=1)...")
    agent_v2 = build_support_agent(
        collection_name="langfuse_eval_v2", chunk_size=1500, chunk_overlap=0, top_k=1
    )
    print("Running Langfuse Experiment B (v2-improved)...")
    res_b = dataset.run_experiment(
        name="demo-v2-improved",
        run_name=f"demo-v2-improved-{stamp}",
        description="Improved chunking: chunk_size=1500, top_k=1",
        task=_make_task(agent_v2["app"]),
        evaluators=[routing_evaluator, keyword_correctness],
        max_concurrency=4,
        metadata={"model": "gpt-4o-mini", "version": "improved-chunking", "chunk_size": 1500, "top_k": 1},
    )
    print(res_b.format(include_item_results=False))

    langfuse.flush()


if __name__ == "__main__":
    main()
