"""
CI/CD Offline Evaluation Gate
-----------------------------
Runs the 8 policy evaluation examples against the FinTech support agent
and exits non-zero if quality drops below configured thresholds.

Thresholds (override via env vars):
  CI_MIN_ROUTING_ACCURACY    (default: 0.90)
  CI_MIN_KEYWORD_CORRECTNESS (default: 0.60)
"""

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval_dataset import HILL_CLIMB_EXAMPLES

load_dotenv()

# Silence noisy chromadb posthog telemetry warning
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import ask, build_support_agent

MIN_ROUTING = float(os.environ.get("CI_MIN_ROUTING_ACCURACY", "0.90"))
MIN_KEYWORD = float(os.environ.get("CI_MIN_KEYWORD_CORRECTNESS", "0.60"))


def _score_keyword_correctness(actual: str, expected: str) -> float:
    """Return fraction of key numbers/dollar amounts from expected present in actual."""
    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+", expected.lower())
    if not key_terms:
        return 0.5
    actual_lower = actual.lower()
    matches = sum(1 for term in key_terms if term in actual_lower)
    return matches / len(key_terms)


def main() -> int:
    print("Building FinTech support agent for CI offline evaluation...")
    agent = build_support_agent(
        collection_name="ci_eval_gate", chunk_size=1500, chunk_overlap=0, top_k=1
    )
    app = agent["app"]

    routing_scores: list[float] = []
    keyword_scores: list[float] = []

    for idx, ex in enumerate(HILL_CLIMB_EXAMPLES, start=1):
        question = ex["inputs"]["question"]
        expected_intent = ex["outputs"]["intent"]
        expected_answer = ex["outputs"]["answer"]

        result = ask(app, question)
        r_score = 1.0 if result["intent"] == expected_intent else 0.0
        k_score = _score_keyword_correctness(result["response"], expected_answer)

        routing_scores.append(r_score)
        keyword_scores.append(k_score)
        print(
            f"  [{idx}/{len(HILL_CLIMB_EXAMPLES)}] "
            f"routing={r_score:.2f}  keyword={k_score:.2f}  q={question[:55]}"
        )

    avg_routing = sum(routing_scores) / len(routing_scores)
    avg_keyword = sum(keyword_scores) / len(keyword_scores)

    print("\n" + "=" * 55)
    print(f"OFFLINE EVAL GATE SUMMARY (Run ID: {agent['run_id']})")
    print("=" * 55)
    print(f"  routing_accuracy    : {avg_routing:.3f} (min required: {MIN_ROUTING:.3f})")
    print(f"  keyword_correctness : {avg_keyword:.3f} (min required: {MIN_KEYWORD:.3f})")

    passed = avg_routing >= MIN_ROUTING and avg_keyword >= MIN_KEYWORD
    if passed:
        print("RESULT: PASSED")
        return 0

    print("RESULT: FAILED — quality below CI threshold!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
