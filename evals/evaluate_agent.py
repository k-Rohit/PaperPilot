"""Run the agent eval suite (decision evals + end-to-end DeepEval) and log it to MLflow on DagsHub.

Run from the repo root:
    uv run python -m evals.evaluate_agent --run-type experiment   # PAID (OpenAI)
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import dagshub
import mlflow
from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig

from evals.agent_evals.agent_runner import (
    build_agent_test_cases,
    generate_agent_dataset,
    latest_agent_dataset_file,
    trajectory_metrics,
)
from evals.agent_evals.decision_evals import arun_decision_evals
from evals.component_evals.generator_metrics import generator_metrics
from evals.component_evals.retriever_metrics import retriever_metrics
from evals.eval_utils import collect_params, metrics_from_result
from evals.mlflow_utils import AGENT_EXPERIMENT_NAME, demote_stale_challengers

logger = logging.getLogger(__name__)

AGENT_REPORTS_DIR = Path(__file__).parent / "reports" / "agent"


def latest_agent_report() -> Path:
    files = list(AGENT_REPORTS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No reports in {AGENT_REPORTS_DIR}")
    return max(files, key=lambda p: p.stat().st_mtime)


def evaluate_agent_app():
    """DeepEval's 5 RAG metrics over the agent's end-to-end answers."""
    metrics = retriever_metrics() + generator_metrics()
    test_cases = build_agent_test_cases()
    AGENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating agent: {len(test_cases)} test cases x {len(metrics)} metrics")
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        identifier="agent_all",
        display_config=DisplayConfig(file_type="md", file_output_dir=str(AGENT_REPORTS_DIR)),
    )


async def _run_agent_parts() -> tuple[dict[str, float], list[dict], Path]:
    decision_metrics, decision_rows = await arun_decision_evals()
    dataset_path = await generate_agent_dataset()
    return decision_metrics, decision_rows, dataset_path


def track_agent_experiment(run_type: str) -> None:
    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)

    experiment = mlflow.set_experiment(AGENT_EXPERIMENT_NAME)

    if run_type == "experiment":
        demoted = demote_stale_challengers(experiment.experiment_id)
        logger.info(f"Demoted {demoted} stale challenger(s)")

    with mlflow.start_run():
        mlflow.set_tags({"run_type": run_type, "pipeline": "agent"})

        # Cheap decision evals first, so a crash there doesn't waste the paid 40-golden run.
        # Both share ONE event loop: langchain-openai caches its async HTTP client, and a
        # second asyncio.run() would hand it a closed loop.
        decision_metrics, decision_rows, dataset_path = asyncio.run(_run_agent_parts())
        logger.info("Decision evals done, agent dataset generated")

        mlflow.log_params(collect_params(pipeline="agent"))
        logger.info("Parameters logged")

        result = evaluate_agent_app()
        rows = json.loads(Path(dataset_path).read_text())

        metrics = {**decision_metrics, **metrics_from_result(result), **trajectory_metrics(rows)}
        mlflow.log_metrics(metrics)
        logger.info("Metrics logged")

        decision_rows_path = AGENT_REPORTS_DIR / f"{Path(dataset_path).stem}_decisions.json"
        decision_rows_path.write_text(json.dumps(decision_rows, indent=2, default=str))

        mlflow.log_artifact(str(latest_agent_dataset_file()))
        mlflow.log_artifact(str(decision_rows_path))
        mlflow.log_artifact(str(latest_agent_report()))
        logger.info("Artifacts logged")

        if run_type == "experiment":
            mlflow.set_tag("stage", "challenger")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)  # only this file's logger
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-type", default="experiment", choices=["experiment", "baseline", "noise"])
    track_agent_experiment(run_type=parser.parse_args().run_type)
