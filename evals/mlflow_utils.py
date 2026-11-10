"""Shared MLflow helpers for the champion/challenger `stage` tag scheme.

Invariants: exactly one champion, at most one challenger.
"""

import mlflow

EXPERIMENT_NAME = "paperpilot-rag"


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
