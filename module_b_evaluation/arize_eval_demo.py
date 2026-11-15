"""
Module B Demo (Arize AX): Evaluation Dataset & A/B Experiment
-------------------------------------------------------------
Mirrors module_b_evaluation/demo.py using Arize AX Datasets & Experiments:
  - Creates/reuses the 'fintech-demo-hill-climb' dataset in Arize AX (8 examples)
  - Runs Experiment A (v1-baseline: chunk_size=100, top_k=1)
  - Runs Experiment B (v2-improved: chunk_size=1500, top_k=1)
  - Evaluates both with routing_accuracy and keyword_correctness
"""

import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from arize import ArizeClient
from arize.experiments import EvaluationResult
from dotenv import load_dotenv

from eval_dataset import DEMO_HC_DATASET_NAME, HILL_CLIMB_EXAMPLES

load_dotenv()

# chromadb's posthog telemetry client is incompatible with posthog>=3 — silence it
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import ask, build_support_agent


def routing_accuracy(output: dict, dataset_row: dict) -> EvaluationResult:
    """Check if the supervisor routed to the correct agent."""
    predicted = (output or {}).get("intent", "")
    expected = dataset_row.get("expected_intent", "")
    score = 1.0 if predicted == expected else 0.0
    return EvaluationResult(
        score=score,
        label="correct" if score == 1.0 else "incorrect",
        explanation=f"predicted={predicted}, expected={expected}",
    )


def keyword_correctness(output: dict, dataset_row: dict) -> EvaluationResult:
    """Check if key numbers/dollar amounts from expected_answer appear in output."""
    actual = (output or {}).get("answer", "").lower()
    expected = dataset_row.get("expected_answer", "").lower()

    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+", expected)
    if not key_terms:
        return EvaluationResult(score=0.5, label="partial")

    matches = sum(1 for term in key_terms if term in actual)
    score = round(matches / len(key_terms), 4)
    return EvaluationResult(
        score=score,
        label=f"{matches}/{len(key_terms)}",
        explanation=f"Matched {matches} of {len(key_terms)} key terms: {key_terms}",
    )


def _ensure_dataset(client: ArizeClient, space_id: str) -> str:
    """Create the Arize AX dataset if it does not already exist."""
    existing = client.datasets.list(space=space_id, name=DEMO_HC_DATASET_NAME)
    if existing.datasets:
        ds = existing.datasets[0]
        print(f"Using existing Arize AX dataset '{ds.name}' (id={ds.id}).")
        return ds.name

    examples = [
        {
            "question": ex["inputs"]["question"],
            "expected_answer": ex["outputs"]["answer"],
            "expected_intent": ex["outputs"]["intent"],
        }
        for ex in HILL_CLIMB_EXAMPLES
    ]
    ds = client.datasets.create(
        name=DEMO_HC_DATASET_NAME,
        space=space_id,
        examples=examples,
        force_http=True,
    )
    print(f"Created Arize AX dataset '{ds.name}' with {len(examples)} examples.")
    return ds.name


def _make_task(app):
    """Return an Arize experiment task function wrapping the compiled graph."""
    def task(dataset_row: dict) -> dict:
        result = ask(app, dataset_row["question"])
        return {
            "answer": result["response"],
            "intent": result["intent"],
            "retrieved_sources": result["retrieved_sources"],
        }
    return task


def main() -> None:
    api_key = os.environ["PHOENIX_API_KEY"]
    space_id = os.environ["ARIZE_SPACE_ID"]
    client = ArizeClient(api_key=api_key)

    dataset_name = _ensure_dataset(client, space_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print("\nBuilding FinTech support agent (v1 — chunk_size=100, top_k=1)...")
    agent_v1 = build_support_agent(
        collection_name="arize_eval_v1", chunk_size=100, chunk_overlap=0, top_k=1
    )
    print("Running Arize AX Experiment A (v1-baseline)...")
    exp_a, df_a = client.experiments.run(
        name=f"demo-v1-baseline-{stamp}",
        dataset=dataset_name,
        space=space_id,
        task=_make_task(agent_v1["app"]),
        evaluators=[routing_accuracy, keyword_correctness],
        concurrency=3,
        set_global_tracer_provider=False,
        force_http=True,
        metadata={"model": "gpt-4o-mini", "version": "baseline", "chunk_size": "100", "top_k": "1"},
    )
    print(f"Experiment A complete: {exp_a.name if exp_a else 'done'} ({len(df_a)} rows)")

    print("\nBuilding improved agent (v2 — chunk_size=1500, top_k=1)...")
    agent_v2 = build_support_agent(
        collection_name="arize_eval_v2", chunk_size=1500, chunk_overlap=0, top_k=1
    )
    print("Running Arize AX Experiment B (v2-improved)...")
    exp_b, df_b = client.experiments.run(
        name=f"demo-v2-improved-{stamp}",
        dataset=dataset_name,
        space=space_id,
        task=_make_task(agent_v2["app"]),
        evaluators=[routing_accuracy, keyword_correctness],
        concurrency=3,
        set_global_tracer_provider=False,
        force_http=True,
        metadata={"model": "gpt-4o-mini", "version": "improved-chunking", "chunk_size": "1500", "top_k": "1"},
    )
    print(f"Experiment B complete: {exp_b.name if exp_b else 'done'} ({len(df_b)} rows)")


if __name__ == "__main__":
    main()
