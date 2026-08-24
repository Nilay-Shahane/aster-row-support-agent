"""
evaluator.py

Runs evaluation/visible-cases.json (and, if present, evaluation/cases_custom.json)
against the LangGraph agent defined in graph.py, using flexible/normalized
matching instead of brittle literal substring checks.

Usage:
    python evaluator.py
    python evaluator.py --visible-only
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

# Assuming these exist in your environment
from graph import graph
from schemas.graph_state import GraphState

EVAL_DIR = Path(__file__).parent / "evaluation"
VISIBLE_CASES_PATH = EVAL_DIR / "visible-cases.json"
CUSTOM_CASES_PATH = EVAL_DIR / "custom-cases.json"


# ==========================================
# NORMALIZATION HELPERS
# ==========================================

_HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2015"
_SPACES = "\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

NEGATION_CUES = [
    "no ", "not ", "n/a", "unavailable", "cannot", "can't", "won't", "unable to",
    "isn't available", "is not available", "wasn't found", "was not found",
    "don't have", "do not have", "does not have", "doesn't have",
]

def _normalize(s: str) -> str:
    s = (s or "").lower()
    for h in _HYPHENS:
        s = s.replace(h, "-")
    for sp in _SPACES:
        s = s.replace(sp, " ")
    for smart, plain in _QUOTES.items():
        s = s.replace(smart, plain)
    s = s.replace("**", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _flexible_contains(haystack: str, needle: str) -> bool:
    h, n = _normalize(haystack), _normalize(needle)
    if n in h: return True
    h2, n2 = h.replace("-", " "), n.replace("-", " ")
    h2, n2 = re.sub(r"\s+", " ", h2), re.sub(r"\s+", " ", n2)
    if n2 in h2: return True
    words = n2.split()
    if words:
        last = words[-1]
        alt_last = last[:-1] if last.endswith("s") else last + "s"
        if " ".join(words[:-1] + [alt_last]) in h2: return True
        VERB_FORM_VARIANTS = {"delivery": ["delivered", "deliver"], "delivered": ["delivery", "deliver"]}
        for variant in VERB_FORM_VARIANTS.get(last, []):
            if " ".join(words[:-1] + [variant]) in h2: return True
    return False

def _contains(haystack: str, needle: str) -> bool:
    return _flexible_contains(haystack, needle)

def _looks_like_date(text: str) -> bool:
    return any(m in text.lower() for m in MONTHS)

def _flexible_date_match(answer: str, expected: str) -> bool:
    ans_l = _normalize(answer)
    exp_l = _normalize(expected)
    month = next((m for m in MONTHS if m in exp_l), None)
    day_match = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\b", exp_l)
    year_match = re.search(r"\b(20\d\d)\b", exp_l)
    if month and month not in ans_l: return False
    if day_match:
        day = day_match.group(1)
        if not re.search(rf"\b0?{day}\b", ans_l): return False
    if year_match and year_match.group(1) not in ans_l: return False
    return True

def _check_must_include(answer: str, items: list) -> list:
    failures = []
    for text in items:
        if _looks_like_date(text):
            if not _flexible_date_match(answer, text): failures.append(f"missing required date: '{text}'")
        elif not _contains(answer, text):
            failures.append(f"missing required text: '{text}'")
    return failures

def _invented_check(answer: str, forbidden_terms: list, label: str) -> list:
    failures = []
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    for term in forbidden_terms:
        term_l = _normalize(term)
        for sent in sentences:
            sent_l = _normalize(sent)
            if term_l in sent_l and not any(cue in sent_l for cue in NEGATION_CUES):
                failures.append(f"{label} present without negation: '{term}'")
                break
    return failures

def _sources_match(sources_cited: list, filename: str) -> bool:
    return any(_normalize(s).startswith(_normalize(filename)) for s in sources_cited)


# ==========================================
# CASE RUNNER & MODELS
# ==========================================

@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    failures: list = field(default_factory=list)
    query: str = ""
    tools: list = field(default_factory=list)
    handoff: bool = False
    required_sources: list = field(default_factory=list)


def evaluate_case(case: dict) -> CaseResult:
    case_id = case["id"]
    category = case["category"]
    expect = case["expect"]
    messages = case["messages"]
    
    # Store first query for reporting
    query_str = messages[0]["content"] if messages else ""

    thread_config = {"configurable": {"thread_id": f"eval_session_{case_id}"}}
    final_state = None
    
    for msg in messages:
        if msg["role"] != "user": continue
        initial_state = {"messages": [HumanMessage(content=msg["content"])], "query": msg["content"]}
        final_state = graph.invoke(initial_state, config=thread_config)

    if final_state is None:
        return CaseResult(case_id, category, False, ["no user messages in case"], query_str, [], False, [])

    answer = str(final_state.get("answer", "") or "")
    sources = final_state.get("sources_used") or []
    handoff = final_state.get("handoff", False)
    route = final_state.get("route", "")
    tool_calls = final_state.get("tool_calls") or []
    tool_called = bool(tool_calls)
    tool_names = [tc.get("tool") for tc in tool_calls if tc.get("tool")]

    failures: list = []

    # Assertions mapped from JSON
    tool_expect = expect.get("tool")
    if tool_expect == "not_called" and (route == "order" or tool_called):
        failures.append(f"expected no tool call, but tool was called.")
    elif tool_expect == "not_called_without_id" and tool_called:
        failures.append("expected tool NOT called without an order ID.")
    elif tool_expect == "order_lookup" and not any(tc.get("tool") == "order_lookup" for tc in tool_calls):
        failures.append("expected order_lookup to be called.")

    if "handoff" in expect and expect["handoff"] is not None:
        if handoff != expect["handoff"]: failures.append(f"handoff mismatch.")

    failures.extend(_check_must_include(answer, expect.get("must_include", [])))
    for text in expect.get("must_not_include", []):
        if _contains(answer, text): failures.append(f"forbidden text present: '{text}'")

    failures.extend(_invented_check(answer, expect.get("must_not_invent", []), "invented content"))
    failures.extend(_invented_check(answer, expect.get("must_refuse_to_disclose", []), "undisclosed field"))

    # Extracted required sources for return and evaluation
    req_sources = expect.get("required_sources", [])
    for filename in req_sources:
        if not _sources_match(sources, filename): failures.append(f"source not cited: '{filename}'")

    for filename in expect.get("forbidden_sources_as_authority", []):
        if _sources_match(sources, filename): failures.append(f"forbidden source cited: '{filename}'")

    return CaseResult(
        case_id=case_id, 
        category=category, 
        passed=len(failures) == 0, 
        failures=failures, 
        query=query_str, 
        tools=tool_names, 
        handoff=handoff,
        required_sources=req_sources
    )


def load_cases(visible_only: bool = False) -> tuple[list, list]:
    with open(VISIBLE_CASES_PATH, "r", encoding="utf-8") as f:
        visible = json.load(f)["cases"]
    if visible_only or not CUSTOM_CASES_PATH.exists():
        return visible, []
    with open(CUSTOM_CASES_PATH, "r", encoding="utf-8") as f:
        custom = json.load(f)["cases"]
    return visible, custom


def run(visible_only: bool = False):
    visible_cases, custom_cases = load_cases(visible_only=visible_only)
    cases = visible_cases + custom_cases
    results = []

    print("\nStarting Test Execution Suite...\n")
    print("-" * 60)

    for case in cases:
        try:
            result = evaluate_case(case)
        except Exception as e:
            result = CaseResult(case["id"], case.get("category", "unknown"), False, [f"Exception: {e}"])
        
        results.append(result)

        # Professional clean printing formatted for screen-recording
        status = "[Pass]" if result.passed else "[Fail]"
        print(f"{status} {result.case_id}")
        print(f"    * Category      : {result.category.replace('-', ' ').title()}")
        print(f"    * Test Query    : \"{result.query}\"")
        print(f"    * Tool Execution: {', '.join(result.tools) if result.tools else 'None'}")
        print(f"    * Agent Handoff : {'Required' if result.handoff else 'Not Required'}")
        
        # New required sources output
        formatted_sources = ', '.join(result.required_sources) if result.required_sources else 'None'
        print(f"    * Req. Sources  : {formatted_sources}")
        
        # Only print failure reasons, keeping it clean for successful runs
        if not result.passed:
            for f in result.failures:
                print(f"    * Failure       : {f}")
        print()

    # Calculate final scores
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    total_passed = sum(1 for r in results if r.passed)
    total_cases = len(results)

    # Clean End Summary Block
    print("=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    for category, cat_results in sorted(by_category.items()):
        passed = sum(1 for r in cat_results if r.passed)
        total = len(cat_results)
        cat_name = category.replace('-', ' ').title()
        print(f"{cat_name:30s} : {passed} / {total} passed")

    print("-" * 60)
    
    # --- NEW: Visible vs Custom Breakdown ---
    visible_total = len(visible_cases)
    visible_results = results[:visible_total]
    visible_passed = sum(1 for r in visible_results if r.passed)
    print(f"{'Visible Cases (Base)':30s} : {visible_passed} / {visible_total} passed")
    
    custom_total = len(custom_cases)
    if custom_total > 0:
        custom_results = results[visible_total:]
        custom_passed = sum(1 for r in custom_results if r.passed)
        print(f"{'Custom Cases (Extra)':30s} : {custom_passed} / {custom_total} passed")
        
    print("-" * 60)
    print(f"{'FINAL SCORE':30s} : {total_passed} / {total_cases} PASSED")
    print("=" * 60)
    print("\nRun completed.")

    return total_passed == total_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-only", action="store_true", help="Skip cases_custom.json even if present")
    args = parser.parse_args()

    all_passed = run(visible_only=args.visible_only)
    sys.exit(0 if all_passed else 1)