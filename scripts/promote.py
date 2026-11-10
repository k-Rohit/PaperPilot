"""Act on the regression/promotion gate's verdict by moving the `stage` tags in MLflow.

Run from the repo root:
    uv run python -m scripts.promote promote
    uv run python -m scripts.promote reject
"""

import sys

import dagshub
import mlflow

from evals.mlflow_utils import EXPERIMENT_NAME, get_run_by_stage

DECISIONS = {"promote", "reject"}


def main(decision: str) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}, use 'promote' or 'reject'")

    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {EXPERIMENT_NAME!r}")

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
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.promote {promote|reject}")
    main(sys.argv[1])
