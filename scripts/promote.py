"""Act on the regression/promotion gate's verdict by moving the `stage` tags in MLflow.

Run from the repo root:
    uv run python -m scripts.promote promote [rag|agent]
    uv run python -m scripts.promote reject [rag|agent]
The pipeline defaults to `rag`.
"""

import sys

import dagshub
import mlflow

from evals.mlflow_utils import PIPELINES, get_run_by_stage

DECISIONS = {"promote", "reject"}


def main(decision: str, pipeline: str = "rag") -> None:
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}, use 'promote' or 'reject'")
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}, use one of {sorted(PIPELINES)}")
    experiment_name, _ = PIPELINES[pipeline]

    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")

    client = mlflow.MlflowClient()
    challenger = get_run_by_stage(experiment.experiment_id, "challenger")

    if decision == "promote":
        champion = get_run_by_stage(experiment.experiment_id, "champion")
        # Archive first: a crash between the two calls leaves 0 champions (loud), not 2 (silent).
        client.set_tag(champion.info.run_id, "stage", "archived")
        client.set_tag(challenger.info.run_id, "stage", "champion")
        print(f"Promoted {challenger.info.run_name}; archived {champion.info.run_name}")
    else:
        client.set_tag(challenger.info.run_id, "stage", "rejected")
        print(f"Rejected {challenger.info.run_name}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python -m scripts.promote {promote|reject} [rag|agent]")
    main(*sys.argv[1:])
