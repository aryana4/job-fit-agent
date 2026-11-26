"""Unit tests that run without an API key, using a scripted fake model."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import run_agent, validate, write_csv  # noqa: E402
from evals.run_evals import score_results  # noqa: E402

POSTING = (
    "Lumen AI - AI Response Evaluator\nPay: $35/hour\n"
    "Review and rank responses from large language models for accuracy."
)

GOOD = {
    "company": "Lumen AI", "title": "AI Response Evaluator", "location": "Remote",
    "salary": "$35/hour", "deadline": "Not listed", "fit_score": 8,
    "best_resume": "UX", "strengths": ["research judgment"], "gaps": [],
    "evidence": ["Review and rank responses from large language models"],
}


def tool_call(name, data, call_id="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=data, id=call_id)


class FakeClient:
    """Returns pre-scripted model responses, one per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.responses.pop(0))


def test_valid_assessment_passes():
    assert validate(GOOD, POSTING) == []


def test_hallucinated_quote_is_caught():
    bad = {**GOOD, "evidence": ["Requires a PhD in machine learning"]}
    problems = validate(bad, POSTING)
    assert any("not found in posting" in p for p in problems)


def test_missing_field_is_caught():
    bad = {k: v for k, v in GOOD.items() if k != "salary"}
    assert validate(bad, POSTING) == ["Missing field: salary"]


def test_agent_retries_after_failed_evidence_check():
    bad = {**GOOD, "evidence": ["made-up requirement"]}
    client = FakeClient([
        [tool_call("submit_assessment", bad, "t1")],
        [tool_call("submit_assessment", GOOD, "t2")],
    ])
    result = run_agent(client, "resume", "sample.txt", posting_text=POSTING)
    assert result["status"] == "ok"
    assert client.calls == 2


def test_agent_hands_off_to_human_when_it_keeps_failing():
    bad = {**GOOD, "evidence": ["made-up requirement"]}
    client = FakeClient([[tool_call("submit_assessment", bad, f"t{i}")] for i in range(6)])
    result = run_agent(client, "resume", "sample.txt", posting_text=POSTING)
    assert result["status"] == "needs human review"


def test_csv_written(tmp_path):
    out = tmp_path / "results.csv"
    write_csv([{"source": "sample.txt", **GOOD, "status": "ok"}], str(out))
    text = out.read_text()
    assert "Lumen AI" in text and "fit_score" in text


def test_eval_scoring():
    expected = {"a.txt": {"company": "Lumen AI", "title": "AI Response Evaluator",
                          "salary": "$35/hour", "deadline": "Not listed", "best_resume": "UX"}}
    summary = score_results(expected, {"a.txt": {**GOOD, "status": "ok"}})
    assert summary["overall_accuracy"] == 1.0
