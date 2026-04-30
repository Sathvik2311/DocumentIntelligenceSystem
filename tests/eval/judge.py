"""LLM-judge faithfulness scorer.

Asks the active LLM to grade whether an answer is grounded in its citations.
Uses the same provider abstraction as `generate_answer`, so by default this is
free (Ollama) — no extra cost to run the eval.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable

from backend.services.generation import _get_provider

logger = logging.getLogger(__name__)


_JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a question, a candidate answer, and the "
    "source passages used to generate it, score how faithful the answer is to the "
    "passages on a 0-to-1 scale. Faithful means every factual claim in the answer "
    "is directly supported by at least one passage; small paraphrases are fine. "
    "Hallucinations, unsupported claims, or use of outside knowledge bring the "
    "score down. Return ONLY a single JSON object on one line with keys "
    '`score` (float in [0,1]) and `reasoning` (one short sentence). No prose '
    "before or after the JSON."
)


@dataclass(frozen=True)
class JudgeResult:
    score: float
    reasoning: str


def judge_faithfulness(
    question: str,
    answer: str,
    citations: Iterable[dict],
) -> JudgeResult:
    """Run the LLM judge and return a parsed score + reasoning."""
    blocks: list[str] = []
    for c in citations:
        head = (
            f"[{c.get('rank')}] {c.get('filename')} (page {c.get('page_number')}, "
            f"chunk {c.get('chunk_index')})"
        )
        text = (c.get("text") or "").strip()
        if not text:
            text = "(text not available)"
        blocks.append(f"{head}\n{text}")
    passages = "\n\n---\n\n".join(blocks) if blocks else "(no citations)"

    user = (
        f"Question: {question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Source passages:\n{passages}"
    )

    try:
        result = _get_provider().complete(_JUDGE_SYSTEM, [{"role": "user", "content": user}])
    except RuntimeError as exc:
        logger.warning("Judge provider failed: %s", exc)
        return JudgeResult(score=0.0, reasoning=f"judge unavailable: {exc}")

    return _parse(result.text)


def _parse(text: str) -> JudgeResult:
    """Extract `{score, reasoning}` from the model's reply, robust to wrappers."""
    text = (text or "").strip()
    # Try strict JSON first.
    candidate = text
    if not candidate.startswith("{"):
        # Pull the first {...} block out of any wrapper text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = match.group(0) if match else ""
    try:
        body = json.loads(candidate)
        score = float(body.get("score", 0.0))
        reasoning = str(body.get("reasoning", ""))
        score = max(0.0, min(1.0, score))
        return JudgeResult(score=score, reasoning=reasoning)
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse judge reply: %s", text[:200])
        return JudgeResult(score=0.0, reasoning=f"parse error: {exc}")
