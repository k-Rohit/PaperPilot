"""Measure how much each eval metric wobbles by chance, and save it to the pipeline's thresholds file.

Reads every same-config run (tagged run_type=baseline or noise) from MLflow on DagsHub and
takes the standard deviation of each metric across them. The regression gate later allows a
challenger to drop by up to 2 x this noise before calling it a real regression.

Run from the repo root:
    uv run python -m scripts.create_thresholds [rag|agent]     # default: rag

For the agent, the decision metrics run at temperature 0 and may show zero noise, so the file
also gets `noise_floors` (1 / number of cases behind the metric): one flipped case is the
smallest change that can ever be seen.
"""

import json
import sys
from pathlib import Path
from statistics import stdev

import dagshub
import mlflow

from evals.mlflow_utils import PIPELINES

NOISE_RUN_TYPES = ["baseline", "noise"]  # runs with the same config; only chance differs between them
UNGATED_METRICS = {
    "rag": {"pass_rate"},
    "agent": {
        "pass_rate",
        "Guardrail Hard Accuracy",
        "Routing Hard Accuracy",  # 24 cases, too noisy
        "Avg Retrieval Attempts",
        "Live Fetch Rate",  # lower-is-better / not a quality score
    },
}
DECISION_CASES_PATH = Path(__file__).parent.parent / "evals" / "agent_evals" / "decision_cases.json"


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


def compute_noise_thresholds(runs: list, ungated: set[str]) -> dict[str, float]:
    """{metric: std across runs}, e.g. {"Faithfulness": 0.019, ...}."""
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 same-config runs to measure noise, found {len(runs)}")

    metric_names = set(runs[0].data.metrics) - ungated
    return {name: stdev(run.data.metrics[name] for run in runs) for name in sorted(metric_names)}


def compute_noise_floors(cases_path: Path = DECISION_CASES_PATH) -> dict[str, float]:
    """One flipped case is the smallest visible change: 1/n for each decision metric."""
    if not Path(cases_path).exists():
        raise FileNotFoundError(f"Decision cases file not found: {cases_path} (needed for the agent noise floors)")
    cases = json.loads(Path(cases_path).read_text())
    if isinstance(cases, dict):  # tolerate {"cases": [...]}
        cases = cases["cases"]
    n_guardrail = len(cases)
    n_in = sum(1 for case in cases if case.get("guardrail") == "in")
    if n_guardrail == 0 or n_in == 0:
        raise ValueError(f"No usable decision cases in {cases_path}")
    return {"Guardrail Accuracy": 1 / n_guardrail, "Tool Routing Accuracy": 1 / n_in}


def main(pipeline: str = "rag") -> None:
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}, use one of {sorted(PIPELINES)}")
    experiment_name, thresholds_path = PIPELINES[pipeline]

    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r}")

    runs = fetch_noise_runs(experiment.experiment_id)
    thresholds = {
        "num_runs": len(runs),
        "run_names": [run.info.run_name for run in runs],
        "noise_thresholds": compute_noise_thresholds(runs, UNGATED_METRICS[pipeline]),
    }
    if pipeline == "agent":
        thresholds["noise_floors"] = compute_noise_floors()

    thresholds_path.write_text(json.dumps(thresholds, indent=4) + "\n")
    print(json.dumps(thresholds, indent=4))
    print(f"Saved to {thresholds_path}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python -m scripts.create_thresholds [rag|agent]")
    main(*sys.argv[1:])
