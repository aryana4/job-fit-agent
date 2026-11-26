"""Tools the agent can call, plus the schemas that describe them to the model."""

import re

import requests
from bs4 import BeautifulSoup

MAX_CHARS = 15000  # keep postings small enough for the model's context

RESUME_VERSIONS = ["Research", "Tech", "UX", "Other"]

TOOL_SCHEMAS = [
    {
        "name": "fetch_posting",
        "description": (
            "Download a job posting from a URL and return its visible text. "
            "Some job sites need JavaScript and return little text; if so, say "
            "the posting could not be read rather than guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Posting URL"}},
            "required": ["url"],
        },
    },
    {
        "name": "submit_assessment",
        "description": (
            "Submit the final structured assessment of the posting. Every value "
            "must come from the posting text; use 'Not listed' when a detail is "
            "missing. 'evidence' must contain short quotes copied exactly from "
            "the posting that support the fit score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "title": {"type": "string"},
                "location": {"type": "string"},
                "salary": {"type": "string"},
                "deadline": {"type": "string"},
                "fit_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "best_resume": {"type": "string", "enum": RESUME_VERSIONS},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "gaps": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "required": [
                "company", "title", "location", "salary", "deadline",
                "fit_score", "best_resume", "strengths", "gaps", "evidence",
            ],
        },
    },
]


def fetch_posting(url: str, timeout: int = 15) -> str:
    """Return the readable text of a web page, trimmed to MAX_CHARS."""
    response = requests.get(
        url, timeout=timeout, headers={"User-Agent": "job-fit-agent/1.0"}
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    return text[:MAX_CHARS]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace and quote styles for fuzzy matching."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()
