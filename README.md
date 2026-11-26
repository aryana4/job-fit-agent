# Job-Fit Agent

An AI agent that reads job postings, scores how well they fit a resume, and logs the results to a spreadsheet, with built-in checks that stop it from making things up.

I built this while job hunting. Using AI agents for my search, I kept running into the same problems: they invented requirements, used stale information, and sounded confident when they were wrong. This project is my attempt to design an agent that is useful **and** verifiable.

## What it does

For each job posting (a URL or a text file), the agent:

1. Reads the posting (fetching it from the web if needed).
2. Extracts the company, title, location, salary and deadline.
3. Scores fit from 1–10 against your resume, listing strengths and gaps.
4. Recommends which version of your resume to send (Research, Tech, UX or Other).
5. Backs up its score with quotes copied exactly from the posting.
6. Appends everything to `results.csv`.

## How it stays honest

The agent works in a tool-use loop with Claude. Two verification checkpoints run **in code**, not in the prompt, so the model can't talk its way past them:

| Checkpoint | What it catches |
| --- | --- |
| **Schema check** | Missing fields, scores outside 1–10, invalid resume choices |
| **Evidence check** | Every supporting quote must actually appear in the posting text. Invented requirements are rejected. |

If a check fails, the specific error goes back to the model and it retries. After six attempts, it stops and marks the posting **"needs human review"** instead of guessing. It never applies to jobs or contacts anyone.

```
posting ──► model reads & assesses ──► submit_assessment
                 ▲                            │
                 │                    schema + evidence checks
                 │                            │
                 └──── error, retry ◄── fail ─┤
                                              │ pass
                                              ▼
                                        results.csv
```

## Evaluation

`evals/` contains three fictional postings with known answers. The eval runner measures field-level extraction accuracy and how many postings needed human review:

```bash
python evals/run_evals.py --resume resume.md
```

It prints a summary in this format (values are from 0 to 1):

```json
{
  "postings": 3,
  "needs_human_review": <count>,
  "field_accuracy": {"company": <0-1>, "title": <0-1>, "salary": <0-1>, "deadline": <0-1>, "best_resume": <0-1>},
  "overall_accuracy": <0-1>
}
```

Unit tests use a scripted fake model, so they run without an API key:

```bash
python -m pytest
```

They cover the evidence check catching a hallucinated quote, the agent retrying after a failed check, and the agent handing off to a human when it keeps failing.

## Setup

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/<your-username>/job-fit-agent.git
cd job-fit-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"   # Windows: set ANTHROPIC_API_KEY=your-key-here
```

Copy `resume.example.md` to `resume.md` and paste in your own resume as plain text.

## Usage

```bash
# Local posting files (most reliable)
python agent.py --resume resume.md --postings evals/postings/*.txt

# URLs, one per line in urls.txt
python agent.py --resume resume.md --urls urls.txt
```

Note: many job sites load postings with JavaScript, so fetching by URL can return little text. When that happens, the agent says so and the posting goes to human review. Saving the posting text to a `.txt` file always works.

## Project structure

```
agent.py          Agent loop, verification checks, CSV output, CLI
tools.py          Tool schemas, web fetching, text normalization
evals/            Fictional postings, expected answers, eval runner
tests/            Unit tests with a fake model (no API key needed)
```
