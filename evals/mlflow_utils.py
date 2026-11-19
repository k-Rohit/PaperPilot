"""Shared MLflow helpers for the champion/challenger `stage` tag scheme.

Invariants: exactly one champion, at most one challenger.
"""

import json
from pathlib import Path

import dagshub
import mlflow

EXPERIMENT_NAME = "paperpilot-rag"
AGENT_EXPERIMENT_NAME = "paperpilot-agent"

EVALS_DIR = Path(__file__).parent
NOISE_MULTIPLIER = 2
EPSILON = 1e-9  # see tolerance()

# pipeline -> (MLflow experiment, thresholds file)
PIPELINES = {
    "rag": (EXPERIMENT_NAME, EVALS_DIR / "thresholds.json"),
    "agent": (AGENT_EXPERIMENT_NAME, EVALS_DIR / "thresholds_agent.json"),
}


def find_runs_by_stage(experiment_id: str, stage: str) -> list:
    """Every run currently tagged stage=<stage> (may be 0, 1 or many)."""
    return mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.stage = '{stage}'",
        output_format="list",
    )


def get_run_by_stage(experiment_id: str, stage: str):
    """The single run tagged stage=<stage>; raises unless exactly one exists."""
    runs = find_runs_by_stage(experiment_id, stage)
    if len(runs) != 1:
        raise RuntimeError(f"Expected exactly 1 run with stage={stage!r}, found {len(runs)}")
    return runs[0]


def demote_stale_challengers(experiment_id: str) -> int:
    """Retag any leftover challenger as 'demoted' so the next run is the only one. Returns count."""
    client = mlflow.MlflowClient()
    stale = find_runs_by_stage(experiment_id, "challenger")
    for run in stale:
        client.set_tag(run.info.run_id, "stage", "demoted")
    return len(stale)


def tolerance(metric: str, noise: dict[str, float], floors: dict[str, float]) -> float:
    """How far a metric may move before it counts as real: 2 x noise, but never below its floor.

    The tiny EPSILON stops float rounding from deciding exact-boundary cases: with a floor of
    1/60, a one-case change (k/60 - (k-1)/60) must count as noise, never as a real gain or drop.
    """
    return max(NOISE_MULTIPLIER * noise[metric], floors.get(metric, 0)) + EPSILON


def load_noise_floors(thresholds_path: Path) -> dict[str, float]:
    """Optional `noise_floors` section of a thresholds file ({} when absent, e.g. RAG)."""
    return json.loads(Path(thresholds_path).read_text()).get("noise_floors", {})


def load_gate_inputs(experiment_name: str, thresholds_path: Path) -> tuple[dict, dict, dict]:
    """(champion_metrics, challenger_metrics, noise) for the gate tests. Reads DagsHub."""
    thresholds = json.loads(Path(thresholds_path).read_text())
    if thresholds.get("num_runs", 0) < 2:
        raise RuntimeError(
            f"{thresholds_path} has no measured noise (num_runs={thresholds.get('num_runs', 0)}). "
            "Do the baseline + noise runs and run scripts.create_thresholds first."
        )
    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")
    champion = get_run_by_stage(experiment.experiment_id, "champion").data.metrics
    challenger = get_run_by_stage(experiment.experiment_id, "challenger").data.metrics
    noise = json.loads(Path(thresholds_path).read_text())["noise_thresholds"]
    return champion, challenger, noise
