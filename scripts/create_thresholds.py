"""Measure how much each eval metric wobbles by chance, and save it to evals/thresholds.json.

Reads every same-config run (tagged run_type=baseline or noise) from MLflow on DagsHub and
takes the standard deviation of each metric across them. The regression gate later allows a
challenger to drop by up to 2 x this noise before calling it a real regression.

Run from the repo root:
    uv run python -m scripts.create_thresholds
"""

import json
from pathlib import Path
from statistics import stdev

import dagshub
import mlflow

EXPERIMENT_NAME = "paperpilot-rag"
NOISE_RUN_TYPES = ["baseline", "noise"]  # runs with the same config; only chance differs between them
UNGATED_METRICS = {"pass_rate"}
THRESHOLDS_PATH = Path(__file__).parent.parent / "evals" / "thresholds.json"


def fetch_noise_runs(experiment_id: str) -> list:
    # DagsHub's MLflow server doesn't support `IN` on tags, so query one run_type at a time.
    runs = []
    for run_type in NOISE_RUN_TYPES:
        runs += mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.run_type = '{run_type}'",
            output_format="list",
        )
    return runs


def compute_noise_thresholds(runs: list) -> dict[str, float]:
    """{metric: std across runs}, e.g. {"Faithfulness": 0.019, ...}."""
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 same-config runs to measure noise, found {len(runs)}")

    metric_names = set(runs[0].data.metrics) - UNGATED_METRICS
    return {name: stdev(run.data.metrics[name] for run in runs) for name in sorted(metric_names)}


def main() -> None:
    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {EXPERIMENT_NAME!r}")

    runs = fetch_noise_runs(experiment.experiment_id)
    thresholds = {
        "num_runs": len(runs),
        "run_names": [run.info.run_name for run in runs],
        "noise_thresholds": compute_noise_thresholds(runs),
    }

    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=4) + "\n")
    print(json.dumps(thresholds, indent=4))
    print(f"Saved to {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
