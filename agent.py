"""Job-fit agent: reads a job posting, scores it against a resume, and logs the result.

The model works in a tool-use loop. Two verification checkpoints run in code,
not in the prompt, so the agent cannot talk its way past them:

1. Schema check  - every required field is present and well-typed.
2. Evidence check - every quote in `evidence` must appear in the posting text.
   This catches hallucinated requirements before they reach the spreadsheet.

If a check fails, the error goes back to the model as a tool result and it
retries. After MAX_TURNS the agent gives up and marks the posting for human
review instead of guessing. It never applies to jobs or contacts anyone.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tools import RESUME_VERSIONS, TOOL_SCHEMAS, fetch_posting, normalize

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TURNS = 6

SYSTEM_PROMPT = """You are a careful job-search assistant.
You receive a candidate's resume and one job posting (either as text or a URL
you must fetch). Assess how well the candidate fits.

Rules:
- Only use facts stated in the posting. If a detail is missing, write "Not listed".
- Score fit from 1 (poor) to 10 (excellent) based on the posting's stated
  requirements versus the resume. Be honest about gaps; do not inflate scores.
- Pick the best resume version: Research (research/coordinator roles),
  Tech (developer/data/implementation roles), UX (UX research and AI evaluation
  roles), or Other.
- Support your score with 2-4 short quotes copied exactly from the posting.
- Finish by calling submit_assessment exactly once."""

CSV_FIELDS = [
    "source", "company", "title", "location", "salary", "deadline",
    "fit_score", "best_resume", "strengths", "gaps", "evidence", "status",
]


def validate(assessment: dict, posting_text: str) -> list[str]:
    """Return a list of problems with an assessment (empty list = it passed)."""
    problems = []
    for field in TOOL_SCHEMAS[1]["input_schema"]["required"]:
        if field not in assessment:
            problems.append(f"Missing field: {field}")
    if problems:
        return problems

    score = assessment["fit_score"]
    if not isinstance(score, int) or not 1 <= score <= 10:
        problems.append("fit_score must be an integer from 1 to 10")
    if assessment["best_resume"] not in RESUME_VERSIONS:
        problems.append(f"best_resume must be one of {RESUME_VERSIONS}")
    if not assessment["evidence"]:
        problems.append("evidence must contain at least one quote")

    haystack = normalize(posting_text)
    for quote in assessment["evidence"]:
        if normalize(quote) not in haystack:
            problems.append(f"Evidence quote not found in posting: {quote!r}")
    return problems


def run_agent(client, resume: str, source: str, posting_text: str | None = None,
              model: str = DEFAULT_MODEL) -> dict:
    """Assess one posting. Pass posting_text for local files, or a URL as source."""
    if posting_text is None:
        task = f"Fetch and assess this posting: {source}"
    else:
        task = f"Assess this posting.\n\n<posting>\n{posting_text}\n</posting>"

    messages = [{
        "role": "user",
        "content": f"<resume>\n{resume}\n</resume>\n\n{task}",
    }]
    fetched_text = posting_text or ""

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model, max_tokens=2000, system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            messages.append({"role": "user", "content": "Please finish by calling submit_assessment."})
            continue

        results = []
        for call in tool_calls:
            if call.name == "fetch_posting":
                try:
                    fetched_text = fetch_posting(call.input["url"])
                    content, is_error = fetched_text, False
                except Exception as exc:  # network errors, 404s, etc.
                    content, is_error = f"Could not fetch posting: {exc}", True
            elif call.name == "submit_assessment":
                problems = validate(call.input, fetched_text)
                if not problems:
                    return {"source": source, **call.input, "status": "ok"}
                content, is_error = "Fix these problems and resubmit:\n- " + "\n- ".join(problems), True
            else:
                content, is_error = f"Unknown tool: {call.name}", True
            results.append({
                "type": "tool_result", "tool_use_id": call.id,
                "content": content, "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})

    return {"source": source, "status": "needs human review"}


def write_csv(rows: list[dict], path: str) -> None:
    """Append assessment rows to a CSV, writing the header if the file is new."""
    is_new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            flat = {k: ("; ".join(v) if isinstance(v, list) else v) for k, v in row.items()}
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score job postings against your resume.")
    parser.add_argument("--resume", required=True, help="Path to your resume as text/markdown")
    parser.add_argument("--urls", help="Text file with one posting URL per line")
    parser.add_argument("--postings", nargs="*", default=[], help="Local posting text files")
    parser.add_argument("--out", default="results.csv", help="CSV file to append results to")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable first.")

    import anthropic
    client = anthropic.Anthropic()
    resume = Path(args.resume).read_text(encoding="utf-8")

    rows = []
    if args.urls:
        for url in Path(args.urls).read_text().split():
            print(f"Assessing {url} ...")
            rows.append(run_agent(client, resume, url, model=args.model))
    for path in args.postings:
        print(f"Assessing {path} ...")
        text = Path(path).read_text(encoding="utf-8")
        rows.append(run_agent(client, resume, path, posting_text=text, model=args.model))

    write_csv(rows, args.out)
    for r in sorted(rows, key=lambda r: r.get("fit_score", 0), reverse=True):
        print(f"{r.get('fit_score', '-'):>3}  {r.get('company', '?')} - {r.get('title', r['source'])}  [{r['status']}]")
    print(f"\nSaved {len(rows)} result(s) to {args.out}")


if __name__ == "__main__":
    main()
