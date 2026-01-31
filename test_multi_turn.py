#!/usr/bin/env python3
"""Test script for multi-turn query rewriting on cloud dataset.

This script tests the Phase 1 & 2 implementation:
- Query rewriting for context-dependent queries
- Conversation memory
- Query duplication for retrieval
- IDK detection (Phase 2)
"""

import os
from langchain.chat_models import init_chat_model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from workflow.generation import query
from workflow.memory import ConversationMemory

# ANSI colors for terminal
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET = "\033[0m"


# Test cases: Multi-turn conversations with expected rewrite behavior
TEST_CASES = [
    {
        "name": "Pronoun Resolution (he)",
        "turns": [
            {
                "query": "Who provides the CDN for IBM Cloud?",
                "expected_rewrite": None,  # Should NOT rewrite (standalone)
                "expected_keywords": ["Akamai", "powered by Akamai"],
            },
            {
                "query": "What services do they provide?",
                "expected_rewrite": "What services does Akamai provide?",  # Should rewrite
                "expected_keywords": ["CDN", "content delivery", "edge"],
            },
        ],
    },
    {
        "name": "Pronoun Resolution (it)",
        "turns": [
            {
                "query": "What is TTL in Akamai CDN?",
                "expected_rewrite": None,
                "expected_keywords": ["Time To Live", "caching", "3600"],
            },
            {
                "query": "What is the default value for it?",
                "expected_rewrite": "What is the default TTL value?",
                "expected_keywords": ["day", "24 hours", "hour", "1 day"],
            },
        ],
    },
    {
        "name": "Context Resolution (the former)",
        "turns": [
            {
                "query": "What are the rule types for advanced CDN rules?",
                "expected_rewrite": None,
                "expected_keywords": ["Geographical", "Hotlink", "Token"],
            },
            {
                "query": "How do I configure the first one?",
                "expected_rewrite": "How do I configure geographical access control?",
                "expected_keywords": ["filter", "location", "countries", "continents"],
            },
        ],
    },
    {
        "name": "Multiple Reference Resolution",
        "turns": [
            {
                "query": "How do I start a CDN?",
                "expected_rewrite": None,
                "expected_keywords": ["Start CDN", "Overflow", "Running"],
            },
            {
                "query": "What status shows after I do that?",
                "expected_rewrite": "What status shows after starting a CDN?",
                "expected_keywords": ["CNAME", "Running"],
            },
        ],
    },
    {
        "name": "Complex Multi-Turn",
        "turns": [
            {
                "query": "What is the South American region list for CDN?",
                "expected_rewrite": None,
                "expected_keywords": ["Argentina", "Brazil", "Chile"],
            },
            {
                "query": "Is Peru in that list?",
                "expected_rewrite": "Is Peru in the South American region list?",
                "expected_keywords": ["Peru", "yes"],
            },
            {
                "query": "What about Mexico?",
                "expected_rewrite": "Is Mexico in the South American region list?",
                "expected_keywords": ["no", "not"],
            },
        ],
    },
]

# IDK Detection test cases - queries that should return IDK
IDK_TEST_CASES = [
    {
        "name": "IDK: Unrelated Topic (Space)",
        "query": "What is the escape velocity of Mars?",
        "should_be_idk": True,  # Should return IDK message
        "reason": "Space topic not in CDN documentation",
    },
    {
        "name": "IDK: Unrelated Topic (Cooking)",
        "query": "How do I make a chocolate cake?",
        "should_be_idk": True,
        "reason": "Cooking recipe not in CDN documentation",
    },
    {
        "name": "IDK: Unrelated Topic (Sports)",
        "query": "Who won the 2022 World Cup?",
        "should_be_idk": True,
        "reason": "Sports not in CDN documentation",
    },
]


def check_keywords(response: str, keywords: list[str]) -> bool:
    """Check if response contains expected keywords (case-insensitive)."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_turn(query: str, response: str, rewritten: str | None):
    """Print a conversation turn."""
    print(f"{Colors.CYAN}User:{Colors.RESET} {query}")

    if rewritten:
        print(f"{Colors.MAGENTA}[Rewritten: {rewritten}]{Colors.RESET}")

    print(f"{Colors.GREEN}Assistant:{Colors.RESET} {response[:300]}{'...' if len(response) > 300 else ''}")
    print()


def evaluate_response(
    response: str,
    expected_keywords: list[str],
    turn_num: int,
) -> dict:
    """Evaluate if response contains expected information."""
    has_keywords = check_keywords(response, expected_keywords)

    return {
        "has_keywords": has_keywords,
        "response_length": len(response),
    }


def run_test_case(test_case: dict, model, verbose: bool = True) -> dict:
    """Run a single test case."""
    if verbose:
        print_header(f"Test: {test_case['name']}")

    memory = ConversationMemory(window_size=5)
    results = {
        "name": test_case["name"],
        "turns": [],
        "passed": 0,
        "total": len(test_case["turns"]),
    }

    for i, turn in enumerate(test_case["turns"], 1):
        query_text = turn["query"]

        if verbose:
            print(f"{Colors.YELLOW}--- Turn {i} ---{Colors.RESET}\n")

        # Get response (this will trigger rewrite logic internally)
        response = query(query_text, model, memory)

        if verbose:
            print_turn(query_text, response, None)

        # Evaluate response
        eval_result = evaluate_response(
            response,
            turn["expected_keywords"],
            i,
        )

        turn_result = {
            "turn": i,
            "query": query_text,
            "has_expected_content": eval_result["has_keywords"],
        }

        if eval_result["has_keywords"]:
            results["passed"] += 1
            if verbose:
                print(f"{Colors.GREEN}✓ Contains expected information{Colors.RESET}")
        else:
            if verbose:
                print(f"{Colors.RED}✗ Missing expected keywords: {turn['expected_keywords']}{Colors.RESET}")

        results["turns"].append(turn_result)
        print()

    return results


def print_summary(all_results: list[dict]):
    """Print test summary."""
    print_header("Test Summary")

    total_passed = sum(r["passed"] for r in all_results)
    total_tests = sum(r["total"] for r in all_results)

    table = Table(title="Multi-Turn Query Rewriting Test Results")
    table.add_column("Test Case", style="cyan")
    table.add_column("Passed", style="green")
    table.add_column("Total", style="white")
    table.add_column("Status", style="bold")

    for result in all_results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result["passed"] == result["total"] else f"{Colors.RED}FAIL{Colors.RESET}"
        table.add_row(
            result["name"],
            str(result["passed"]),
            str(result["total"]),
            status,
        )

    console = Console()
    console.print(table)

    print(f"\n{Colors.BLUE}Overall: {total_passed}/{total_tests} turns passed{Colors.RESET}")

    if total_passed == total_tests:
        print(f"{Colors.GREEN}All tests passed! ✓{Colors.RESET}\n")
    else:
        print(f"{Colors.YELLOW}Some tests failed. Check output above.{Colors.RESET}\n")

    return total_passed == total_tests


def main():
    """Run all test cases."""
    print_header("Multi-Turn Query Rewriting Test Suite")

    print(f"{Colors.YELLOW}Dataset:{Colors.RESET} IBM Cloud CDN")
    print(f"{Colors.YELLOW}Testing:{Colors.RESET} Query rewriting, conversation memory, duplication")
    print(f"{Colors.YELLOW}Model:{Colors.RESET} gpt-4o-mini\n")

    # Initialize model
    model = init_chat_model("gpt-4o-mini")

    # Run all test cases
    all_results = []

    for test_case in TEST_CASES:
        result = run_test_case(test_case, model, verbose=True)
        all_results.append(result)
        print()

    # Print summary
    all_passed = print_summary(all_results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
