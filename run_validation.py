#!/usr/bin/env python3
"""Validation script for the RAG pipeline.

Runs a predefined test matrix against the current ChromaDB index and prints
a pass/fail results grid.  Doubles as a demo artifact for reviewers.

Usage:
    python run_validation.py
    python run_validation.py --debug      # show retrieval debug output
    python run_validation.py --no-color   # plain ASCII table (no ANSI)
"""

import argparse
import logging
import sys
import traceback
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

from rag_pipeline.retriever.retrieve import query_chunks
from rag_pipeline.llm.generator import generate_answer_with_gate

# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

CATEGORY_A: List[Tuple[str, List[str]]] = [
    # (query, expected_signals) — at least one signal must appear in the answer
    ("Where is Tesla headquartered?",
     ["Austin", "Texas"]),
    ("What does Tesla do?",
     ["electric vehicle", "energy", "automotive", "EV"]),
    ("Who is Tesla's auditor?",
     ["PricewaterhouseCoopers", "PwC"]),
    ("What are the main risk factors?",
     ["ITEM 1A", "risk", "competition", "supply", "regulatory"]),
    ("Where is Tesla incorporated?",
     ["Delaware"]),
]

CATEGORY_B: List[str] = [
    # queries that should trigger abstention
    "What is Apple's revenue?",
    "Who is the CEO of Tesla?",
    "What was Tesla's stock price on March 5th?",
]

CATEGORY_C: List[str] = [
    # edge cases — must not crash
    "",
    "asdfghjkl",
    "revenue revenue revenue revenue",
    "x" * 250,  # very long query (250 chars)
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_found(answer: str, signals: List[str]) -> bool:
    answer_lower = answer.lower()
    return any(s.lower() in answer_lower for s in signals)


def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def _run_query(query: str, debug: bool) -> Tuple[str, bool]:
    """Run a single query through retrieve → generate_answer_with_gate."""
    chunks = query_chunks(query, top_k=5, debug=debug)
    answer, abstained = generate_answer_with_gate(query, chunks)
    return answer, abstained

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(debug: bool, use_color: bool) -> int:
    """Execute all test categories and return the count of failures."""

    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    RESET = "\033[0m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""

    DIM = "\033[2m" if use_color else ""

    COL_CAT = 5
    COL_STATUS = 6
    COL_QUERY = 40
    COL_DETAIL = 60

    sep = "+" + "-" * (COL_CAT + 2) + "+" + "-" * (COL_STATUS + 2) + "+" + "-" * (COL_QUERY + 2) + "+" + "-" * (COL_DETAIL + 2) + "+"

    def row(cat: str, status: str, query: str, detail: str):
        return f"| {cat:<{COL_CAT}} | {status:<{COL_STATUS + (len(status) - len(status.replace(GREEN,'').replace(RED,'').replace(RESET,'').replace(BOLD,'')))}} | {_truncate(query, COL_QUERY):<{COL_QUERY}} | {_truncate(detail, COL_DETAIL):<{COL_DETAIL}} |"

    # -- header --
    print()
    print(f"{BOLD}RAG Pipeline Validation{RESET}")
    print(f"{'=' * 40}")
    print()
    print(sep)
    print(f"| {'Cat':<{COL_CAT}} | {'Result':<{COL_STATUS}} | {'Query':<{COL_QUERY}} | {'Detail':<{COL_DETAIL}} |")
    print(sep)

    failures = 0
    total = 0

    # ---- Category A: Known-answer ----
    for query, signals in CATEGORY_A:
        total += 1
        try:
            answer, abstained = _run_query(query, debug)
            if abstained:
                status = f"{RED}FAIL{RESET}"
                detail = "Abstained (expected an answer)"
                failures += 1
            elif _signal_found(answer, signals):
                status = f"{GREEN}PASS{RESET}"
                detail = f"Answer contains expected signal"
            else:
                status = f"{RED}FAIL{RESET}"
                detail = f"Missing signals: {signals}"
                failures += 1
        except Exception as exc:
            status = f"{RED}FAIL{RESET}"
            detail = f"Exception: {exc}"
            answer = None
            failures += 1
        print(row("A", status, query, detail))
        if answer is not None:
            print(f"  {DIM}Response: {_truncate(answer, 120)}{RESET}")

    print(sep)

    # ---- Category B: Unanswerable ----
    for query in CATEGORY_B:
        total += 1
        try:
            answer, abstained = _run_query(query, debug)
            if abstained:
                status = f"{GREEN}PASS{RESET}"
                detail = "Correctly abstained"
            else:
                status = f"{RED}FAIL{RESET}"
                detail = f"Should have abstained: {_truncate(answer, 45)}"
                failures += 1
        except Exception as exc:
            status = f"{RED}FAIL{RESET}"
            detail = f"Exception: {exc}"
            answer = None
            failures += 1
        print(row("B", status, query, detail))
        if answer is not None:
            print(f"  {DIM}Response: {_truncate(answer, 120)}{RESET}")

    print(sep)

    # ---- Category C: Edge cases ----
    for query in CATEGORY_C:
        total += 1
        display_query = query if query else "(empty string)"
        if len(query) > 50:
            display_query = f"(long query, {len(query)} chars)"
        try:
            answer, abstained = _run_query(query, debug)
            status = f"{GREEN}PASS{RESET}"
            detail = "Abstained" if abstained else f"Answered ({len(answer)} chars)"
        except Exception as exc:
            status = f"{RED}FAIL{RESET}"
            detail = f"Crashed: {exc}"
            failures += 1
            answer = None
            if debug:
                traceback.print_exc()
        print(row("C", status, display_query, detail))
        if answer is not None:
            print(f"  {DIM}Response: {_truncate(answer, 120)}{RESET}")

    print(sep)

    # -- summary --
    passed = total - failures
    summary_color = GREEN if failures == 0 else RED
    print()
    print(f"{BOLD}Summary:{RESET} {summary_color}{passed}/{total} passed{RESET}")
    if failures > 0:
        print(f"{RED}{failures} failure(s){RESET}")
    print()
    return failures


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline validation test matrix")
    parser.add_argument("--debug", action="store_true", help="Enable retrieval debug logging")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s - %(message)s")

    failures = run_tests(debug=args.debug, use_color=not args.no_color)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
