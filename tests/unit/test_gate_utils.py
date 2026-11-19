"""Unit tests for the gate helpers: tolerance, noise thresholds/floors, promote/reject tag moves.

No DagsHub, no OpenAI: MLflow and dagshub are mocked.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from scripts import create_thresholds, promote

from evals import mlflow_utils
from evals.mlflow_utils import PIPELINES, tolerance


def make_run(metrics: dict, run_id: str = "id", name: str = "run"):
    return SimpleNamespace(data=SimpleNamespace(metrics=metrics), info=SimpleNamespace(run_id=run_id, run_name=name))


# --- tolerance ---------------------------------------------------------------


def test_tolerance_without_floor_is_two_times_noise():
    assert tolerance("m", {"m": 0.01}, {}) == pytest.approx(0.02)


def test_tolerance_floor_wins_when_noise_is_zero():
    assert tolerance("m", {"m": 0.0}, {"m": 1 / 60}) == pytest.approx(1 / 60)


def test_tolerance_noise_wins_when_larger_than_floor():
    assert tolerance("m", {"m": 0.1}, {"m": 0.01}) == pytest.approx(0.2)


def test_pipelines_map():
    assert PIPELINES["rag"][0] == "paperpilot-rag"
    assert PIPELINES["agent"][0] == "paperpilot-agent"
    assert PIPELINES["agent"][1].name == "thresholds_agent.json"


# --- thresholds --------------------------------------------------------------


def test_ungated_metrics_are_excluded():
    runs = [
        make_run({"Faithfulness": 0.8, "pass_rate": 0.5, "Live Fetch Rate": 0.1}),
        make_run({"Faithfulness": 0.9, "pass_rate": 0.9, "Live Fetch Rate": 0.3}),
    ]
    result = create_thresholds.compute_noise_thresholds(runs, create_thresholds.UNGATED_METRICS["agent"])
    assert set(result) == {"Faithfulness"}
    assert result["Faithfulness"] == pytest.approx(0.0707, abs=1e-3)


def test_rag_only_excludes_pass_rate():
    runs = [make_run({"a": 1.0, "pass_rate": 0.1}), make_run({"a": 0.0, "pass_rate": 0.9})]
    assert set(create_thresholds.compute_noise_thresholds(runs, create_thresholds.UNGATED_METRICS["rag"])) == {"a"}


def test_agent_ungated_set_matches_plan():
    assert create_thresholds.UNGATED_METRICS["agent"] == {
        "pass_rate",
        "Guardrail Hard Accuracy",
        "Routing Hard Accuracy",
        "Avg Retrieval Attempts",
        "Live Fetch Rate",
    }


def test_needs_two_runs():
    with pytest.raises(ValueError):
        create_thresholds.compute_noise_thresholds([make_run({"a": 1.0})], set())


def test_noise_floors_from_cases_file(tmp_path):
    cases = (
        [{"guardrail": "out"}] * 20
        + [{"guardrail": "in", "tool": "retrieve"}] * 30
        + [{"guardrail": "in", "tool": "fetch_live_papers"}] * 10
    )
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    floors = create_thresholds.compute_noise_floors(path)
    assert floors["Guardrail Accuracy"] == pytest.approx(1 / 60)
    assert floors["Tool Routing Accuracy"] == pytest.approx(1 / 40)


def test_noise_floors_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Decision cases file not found"):
        create_thresholds.compute_noise_floors(tmp_path / "nope.json")


# --- promote / reject --------------------------------------------------------


@pytest.fixture
def mocked_mlflow(monkeypatch):
    client = MagicMock()
    champion = make_run({}, "champ-id", "champ")
    challenger = make_run({}, "chal-id", "chal")
    runs = {"champion": champion, "challenger": challenger}
    monkeypatch.setattr(promote.dagshub, "init", MagicMock())
    monkeypatch.setattr(promote.mlflow, "get_experiment_by_name", lambda name: SimpleNamespace(experiment_id="1"))
    monkeypatch.setattr(promote.mlflow, "MlflowClient", lambda: client)
    monkeypatch.setattr(promote, "get_run_by_stage", lambda exp_id, stage: runs[stage])
    return client


def test_promote_archives_champion_then_promotes_challenger(mocked_mlflow):
    promote.main("promote", "agent")
    assert [c.args for c in mocked_mlflow.set_tag.call_args_list] == [
        ("champ-id", "stage", "archived"),
        ("chal-id", "stage", "champion"),
    ]


def test_reject_only_tags_challenger(mocked_mlflow):
    promote.main("reject")
    mocked_mlflow.set_tag.assert_called_once_with("chal-id", "stage", "rejected")


def test_promote_uses_pipeline_experiment(mocked_mlflow, monkeypatch):
    seen = []
    monkeypatch.setattr(
        promote.mlflow, "get_experiment_by_name", lambda name: seen.append(name) or SimpleNamespace(experiment_id="1")
    )
    promote.main("reject", "agent")
    promote.main("reject", "rag")
    assert seen == ["paperpilot-agent", "paperpilot-rag"]


def test_promote_rejects_bad_args(mocked_mlflow):
    with pytest.raises(ValueError):
        promote.main("maybe")
    with pytest.raises(ValueError):
        promote.main("promote", "nope")


def test_load_noise_floors_optional(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"noise_thresholds": {"a": 0.1}}))
    assert mlflow_utils.load_noise_floors(path) == {}
    path.write_text(json.dumps({"noise_thresholds": {"a": 0.1}, "noise_floors": {"a": 0.05}}))
    assert mlflow_utils.load_noise_floors(path) == {"a": 0.05}
