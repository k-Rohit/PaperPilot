import os
from collections import Counter
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test")

import pytest  # noqa: E402

from evals.agent_evals import decision_evals  # noqa: E402

VALID_TOOLS = {"retrieve", "fetch_live_papers"}


@pytest.fixture(scope="module")
def cases():
    return decision_evals.load_cases()


def test_case_counts(cases):
    assert len(cases) == 60
    easy = [c for c in cases if c["difficulty"] == "easy"]
    hard = [c for c in cases if c["difficulty"] == "hard"]
    assert len(easy) == 36
    assert len(hard) == 24
    assert Counter((c["guardrail"], c["tool"]) for c in easy) == {
        ("out", None): 12,
        ("in", "retrieve"): 12,
        ("in", "fetch_live_papers"): 12,
    }
    assert Counter(c["category"] for c in hard).most_common()[-1][1] == 2
    assert len({c["category"] for c in hard}) == 12


def test_labels_valid(cases):
    assert len({c["id"] for c in cases}) == 60
    for c in cases:
        assert set(c) == {"id", "input", "guardrail", "tool", "difficulty", "category"}
        assert c["input"].strip()
        assert c["guardrail"] in {"in", "out"}
        assert c["difficulty"] in {"easy", "hard"}
        if c["guardrail"] == "in":
            assert c["tool"] in VALID_TOOLS
        else:
            assert c["tool"] is None


def _install_fakes(monkeypatch, cases, wrong_guardrail=(), wrong_tool=()):
    """Fake nodes: correct for every case except the ids listed in wrong_*."""
    by_input = {c["input"]: c for c in cases}
    calls = {"router": []}

    async def fake_score(state, runtime):
        case = by_input[state["messages"][-1].content]
        correct_in = case["guardrail"] == "in"
        is_in = (not correct_in) if case["id"] in wrong_guardrail else correct_in
        return {"guardrail_result": SimpleNamespace(score=90 if is_in else 10, reason="fake")}

    async def fake_select(state, runtime):
        case = by_input[state["messages"][-1].content]
        calls["router"].append(case["id"])
        tool = case["tool"]
        if case["id"] in wrong_tool:
            tool = "retrieve" if tool == "fetch_live_papers" else "fetch_live_papers"
        return {"tool_selection": tool, "target_topic": None}

    monkeypatch.setattr(decision_evals, "score_user_query", fake_score)
    monkeypatch.setattr(decision_evals, "select_tool", fake_select)
    return calls


def test_perfect_run(monkeypatch, cases):
    calls = _install_fakes(monkeypatch, cases)
    metrics, rows = decision_evals.run_decision_evals()

    assert set(metrics) == {
        "Guardrail Accuracy",
        "Tool Routing Accuracy",
        "Guardrail Hard Accuracy",
        "Routing Hard Accuracy",
    }
    assert all(v == 1.0 for v in metrics.values())
    assert len(rows) == 60
    # Router runs only for `in` cases.
    assert len(calls["router"]) == sum(c["guardrail"] == "in" for c in cases)


def test_row_fields(monkeypatch, cases):
    _install_fakes(monkeypatch, cases)
    _, rows = decision_evals.run_decision_evals()
    expected_keys = {
        "id", "input", "category", "difficulty", "expected_guardrail", "actual_guardrail",
        "guardrail_score", "guardrail_reason", "expected_tool", "actual_tool",
        "guardrail_correct", "tool_correct",
    }  # fmt: skip
    for r in rows:
        assert set(r) == expected_keys
    out_row = next(r for r in rows if r["expected_guardrail"] == "out")
    assert out_row["actual_tool"] is None
    assert out_row["tool_correct"] is None
    assert out_row["guardrail_score"] == 10
    in_row = next(r for r in rows if r["expected_guardrail"] == "in")
    assert in_row["actual_tool"] == in_row["expected_tool"]
    assert in_row["tool_correct"] is True


def test_scoring_with_mistakes(monkeypatch, cases):
    n_in = sum(c["guardrail"] == "in" for c in cases)
    n_hard_in = sum(c["guardrail"] == "in" and c["difficulty"] == "hard" for c in cases)
    # One easy guardrail miss, one hard guardrail miss, one easy routing miss, one hard routing miss.
    _install_fakes(
        monkeypatch,
        cases,
        wrong_guardrail={"easy-out-01", "hard-pop-01"},
        wrong_tool={"easy-fetch-01", "hard-rnew-01"},
    )
    metrics, rows = decision_evals.run_decision_evals()

    assert metrics["Guardrail Accuracy"] == pytest.approx(58 / 60)
    assert metrics["Guardrail Hard Accuracy"] == pytest.approx(23 / 24)
    assert metrics["Tool Routing Accuracy"] == pytest.approx((n_in - 2) / n_in)
    assert metrics["Routing Hard Accuracy"] == pytest.approx((n_hard_in - 1) / n_hard_in)

    by_id = {r["id"]: r for r in rows}
    assert by_id["easy-out-01"]["guardrail_correct"] is False
    assert by_id["easy-out-01"]["actual_guardrail"] == "in"
    assert by_id["easy-fetch-01"]["tool_correct"] is False
    assert by_id["easy-fetch-01"]["actual_tool"] == "retrieve"


def test_threshold_boundary(monkeypatch, cases):
    """A score equal to the threshold counts as `in`."""
    threshold = decision_evals.GraphConfig().guardrail_threshold

    async def fake_score(state, runtime):
        return {"guardrail_result": SimpleNamespace(score=threshold, reason="edge")}

    async def fake_select(state, runtime):
        return {"tool_selection": "retrieve", "target_topic": None}

    monkeypatch.setattr(decision_evals, "score_user_query", fake_score)
    monkeypatch.setattr(decision_evals, "select_tool", fake_select)
    _, rows = decision_evals.run_decision_evals()
    assert all(r["actual_guardrail"] == "in" for r in rows)
