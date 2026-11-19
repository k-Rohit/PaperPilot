"""Decision evals: does the agent's guardrail and router make the right call?

60 hand-labelled cases (see decision_cases.json) call `score_user_query` and `select_tool`
directly. No judge model is involved; each case costs one or two cheap gpt-4o-mini calls.

Labels encode what we WANT, not what the prompts do today, so a failing hard case is a finding.
"""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.services.agents.config import GraphConfig
from src.services.agents.nodes.guardrail_node import score_user_query
from src.services.agents.nodes.router_node import select_tool

logger = logging.getLogger(__name__)

CASES_FILE = Path(__file__).parent / "decision_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def _tool_is_correct(query: str, expected: str, actual: str) -> bool:
    """Score one routing decision with DeepEval's ToolCorrectnessMetric (no LLM judge needed)."""
    test_case = LLMTestCase(
        input=query,
        actual_output="",
        tools_called=[ToolCall(name=actual)],
        expected_tools=[ToolCall(name=expected)],
    )
    metric = ToolCorrectnessMetric(async_mode=False, include_reason=False, should_exact_match=True)
    metric.measure(test_case, _show_indicator=False)
    return metric.score == 1.0


async def _run_case(case: dict, runtime, semaphore: asyncio.Semaphore) -> dict:
    state = {"messages": [HumanMessage(content=case["input"])]}
    threshold = runtime.context.graph_config.guardrail_threshold

    async with semaphore:
        guardrail_out = await score_user_query(state, runtime)
        # Route only the cases we expect to be in scope, whatever the guardrail said,
        # so routing accuracy is measured independently of the guardrail.
        router_out = await select_tool(state, runtime) if case["guardrail"] == "in" else None

    scoring = guardrail_out["guardrail_result"]
    actual_guardrail = "in" if scoring.score >= threshold else "out"
    actual_tool = router_out["tool_selection"] if router_out else None
    tool_correct = _tool_is_correct(case["input"], case["tool"], actual_tool) if router_out else None

    return {
        "id": case["id"],
        "input": case["input"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "expected_guardrail": case["guardrail"],
        "actual_guardrail": actual_guardrail,
        "guardrail_score": scoring.score,
        "guardrail_reason": scoring.reason,
        "expected_tool": case["tool"],
        "actual_tool": actual_tool,
        "guardrail_correct": actual_guardrail == case["guardrail"],
        "tool_correct": tool_correct,
    }


def _accuracy(rows: list[dict], key: str) -> float:
    """Share of rows where `key` is True. Rows with key None (not applicable) are skipped."""
    scored = [r[key] for r in rows if r[key] is not None]
    return sum(scored) / len(scored) if scored else 0.0


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    hard = [r for r in rows if r["difficulty"] == "hard"]
    return {
        "Guardrail Accuracy": _accuracy(rows, "guardrail_correct"),
        "Tool Routing Accuracy": _accuracy(rows, "tool_correct"),
        "Guardrail Hard Accuracy": _accuracy(hard, "guardrail_correct"),
        "Routing Hard Accuracy": _accuracy(hard, "tool_correct"),
    }


async def arun_decision_evals() -> tuple[dict[str, float], list[dict]]:
    cases = load_cases()
    # The nodes only read runtime.context.graph_config, so a stand-in avoids building the full Context.
    runtime = SimpleNamespace(context=SimpleNamespace(graph_config=GraphConfig()))
    semaphore = asyncio.Semaphore(get_settings().deepeval.max_concurrent)

    rows = await asyncio.gather(*(_run_case(c, runtime, semaphore) for c in cases))
    rows = list(rows)
    return compute_metrics(rows), rows


def run_decision_evals() -> tuple[dict[str, float], list[dict]]:
    return asyncio.run(arun_decision_evals())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    metrics, rows = run_decision_evals()
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    failed = [r for r in rows if not r["guardrail_correct"] or r["tool_correct"] is False]
    print(f"\n{len(failed)} failed case(s):")
    for r in failed:
        print(
            f"- [{r['id']}] ({r['category']}) {r['input']!r}\n"
            f"    guardrail: expected={r['expected_guardrail']} actual={r['actual_guardrail']} "
            f"score={r['guardrail_score']} reason={r['guardrail_reason']!r}\n"
            f"    tool: expected={r['expected_tool']} actual={r['actual_tool']}"
        )
