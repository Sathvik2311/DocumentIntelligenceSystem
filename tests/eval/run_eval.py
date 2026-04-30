"""Eval harness CLI.

Reads `tests/eval/golden.jsonl` (or a custom path), runs the configured
retrieval pipeline against each entry, and reports Hit@k, MRR, and (optionally)
LLM-judge faithfulness. The `--ablate` flag runs all four configurations and
prints a side-by-side comparison table.

Usage:
    python -m tests.eval.run_eval                     # current settings, no LLM judge
    python -m tests.eval.run_eval --with-llm          # include faithfulness scoring
    python -m tests.eval.run_eval --ablate            # compare cosine vs hybrid vs hybrid+rerank
    python -m tests.eval.run_eval --ablate --with-llm # full matrix
    python -m tests.eval.run_eval --golden path/to/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Make the project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.services.generation import generate_answer
from backend.services.retrieval import retrieve

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "eval" / "golden.jsonl"
RESULTS_DIR = PROJECT_ROOT / "tests" / "eval"


@dataclass
class GoldenEntry:
    question: str
    filename: str | None
    must_contain_any: list[str]
    tags: list[str]
    expected_negative: bool = False


@dataclass
class Mode:
    name: str
    use_hybrid: bool
    use_reranker: bool


def load_golden(path: Path) -> list[GoldenEntry]:
    entries: list[GoldenEntry] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            entries.append(
                GoldenEntry(
                    question=row["question"],
                    filename=row.get("filename"),
                    must_contain_any=row.get("must_contain_any") or [],
                    tags=row.get("tags") or [],
                    expected_negative=bool(row.get("expected_negative", False)),
                )
            )
    return entries


def _hits_for(chunks: list[Any], entry: GoldenEntry, k: int) -> tuple[bool, int | None]:
    """Returns (hit_in_top_k, first_hit_rank).

    A "hit" is a chunk whose filename matches AND whose text contains any of the
    expected keywords (case-insensitive). For entries with `expected_negative`,
    a "hit" inverts to "no chunks returned at all" (or no matching chunks).
    """
    keywords_lower = [kw.lower() for kw in entry.must_contain_any]

    def matches(ch: Any) -> bool:
        if entry.filename and ch.filename != entry.filename:
            return False
        if not keywords_lower:
            return True
        text = (ch.text or "").lower()
        return any(kw in text for kw in keywords_lower)

    if entry.expected_negative:
        # Negative case: success = no matching chunks, regardless of k.
        any_match = any(matches(c) for c in chunks[:k])
        return (not any_match, None if any_match else 1)

    for rank, ch in enumerate(chunks[:k], start=1):
        if matches(ch):
            return True, rank
    return False, None


def evaluate(
    entries: list[GoldenEntry],
    mode: Mode,
    k: int,
    top_k: int,
    with_llm: bool,
) -> dict[str, float]:
    """Run one mode across all entries; return aggregate metrics."""
    hits = 0
    rr_total = 0.0
    judge_scores: list[float] = []

    for entry in entries:
        chunks = retrieve(
            entry.question,
            top_k=top_k,
            filename=entry.filename if not entry.expected_negative else None,
            use_hybrid=mode.use_hybrid,
            use_reranker=mode.use_reranker,
        )

        hit, first_rank = _hits_for(chunks, entry, k=k)
        if hit:
            hits += 1
            if first_rank:
                rr_total += 1.0 / first_rank

        if with_llm and not entry.expected_negative and chunks:
            from tests.eval.judge import judge_faithfulness

            answer = generate_answer(entry.question, chunks)
            citations = [
                {
                    "rank": i + 1,
                    "filename": ch.filename,
                    "page_number": ch.page_number,
                    "chunk_index": ch.chunk_index,
                    "text": ch.text,
                }
                for i, ch in enumerate(chunks)
            ]
            judge = judge_faithfulness(entry.question, answer.text, citations)
            judge_scores.append(judge.score)

    n = len(entries)
    return {
        "hit@k": hits / n if n else 0.0,
        "mrr": rr_total / n if n else 0.0,
        "faithfulness": (sum(judge_scores) / len(judge_scores)) if judge_scores else float("nan"),
        "n": n,
    }


def render_table(rows: list[tuple[str, dict[str, float]]], k: int) -> str:
    out = [
        f"| mode                | hit@{k:<2} | mrr   | faithfulness | n  |",
        "|---------------------|---------|-------|--------------|----|",
    ]
    for name, m in rows:
        f_score = m["faithfulness"]
        f_str = f"{f_score:.3f}" if f_score == f_score else "  —  "  # NaN check
        out.append(
            f"| {name:<19} | {m['hit@k']:.3f}   | {m['mrr']:.3f} | "
            f"{f_str:<12} | {int(m['n']):<2} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=5, help="Chunks retrieved per query.")
    parser.add_argument("--k", type=int, default=5, help="Hit@k threshold.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also score answer faithfulness via the LLM judge.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Compare cosine vs hybrid vs hybrid+rerank (overrides current settings).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't write a JSON results file.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s")

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}", file=sys.stderr)
        return 2

    entries = load_golden(args.golden)
    if not entries:
        print("Golden file is empty.", file=sys.stderr)
        return 2

    if args.ablate:
        modes = [
            Mode("cosine", use_hybrid=False, use_reranker=False),
            Mode("hybrid", use_hybrid=True, use_reranker=False),
            Mode("hybrid+rerank", use_hybrid=True, use_reranker=True),
        ]
    else:
        s = get_settings()
        modes = [
            Mode(
                f"current ({'+'.join(filter(None, ['hybrid' if s.enable_hybrid_search else None, 'rerank' if s.enable_reranker else None])) or 'cosine'})",
                use_hybrid=s.enable_hybrid_search,
                use_reranker=s.enable_reranker,
            )
        ]

    rows: list[tuple[str, dict[str, float]]] = []
    print(f"Running eval over {len(entries)} entries × {len(modes)} mode(s)…\n")
    for mode in modes:
        metrics = evaluate(entries, mode, k=args.k, top_k=args.top_k, with_llm=args.with_llm)
        rows.append((mode.name, metrics))

    table = render_table(rows, k=args.k)
    print(table)

    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = RESULTS_DIR / f"results-{ts}.json"
        out_path.write_text(
            json.dumps(
                {
                    "timestamp": ts,
                    "k": args.k,
                    "top_k": args.top_k,
                    "with_llm": args.with_llm,
                    "ablate": args.ablate,
                    "n": len(entries),
                    "results": [{"mode": name, **metrics} for name, metrics in rows],
                },
                indent=2,
            )
        )
        print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
