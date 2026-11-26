"""Measure how accurately the agent extracts posting details.

Runs the agent on fictional sample postings with known answers and reports
field-level accuracy, how often the evidence check passed on the first try,
and how many postings needed human review.

Usage (from the repo root):
    python evals/run_evals.py --resume resume.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import DEFAULT_MODEL, run_agent  # noqa: E402
from tools import normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
FIELDS = ["company", "title", "salary", "deadline", "best_resume"]


def field_matches(expected: str, actual: str) -> bool:
    """Lenient match: one normalized value contains the other."""
    e, a = normalize(str(expected)), normalize(str(actual))
    return bool(e) and (e in a or a in e)


def score_results(expected: dict, results: dict) -> dict:
    """Compare agent results with expected answers and return summary metrics."""
    correct = {f: 0 for f in FIELDS}
    review = 0
    for name, truth in expected.items():
        result = results.get(name, {})
        if result.get("status") != "ok":
            review += 1
            continue
        for f in FIELDS:
            if field_matches(truth[f], result.get(f, "")):
                correct[f] += 1
    n = len(expected)
    return {
        "postings": n,
        "needs_human_review": review,
        "field_accuracy": {f: round(correct[f] / n, 2) for f in FIELDS},
        "overall_accuracy": round(sum(correct.values()) / (n * len(FIELDS)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    resume = Path(args.resume).read_text(encoding="utf-8")
    expected = json.loads((HERE / "expected.json").read_text())

    results = {}
    for name in expected:
        text = (HERE / "postings" / name).read_text(encoding="utf-8")
        print(f"Running {name} ...")
        results[name] = run_agent(client, resume, name, posting_text=text, model=args.model)

    summary = score_results(expected, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
