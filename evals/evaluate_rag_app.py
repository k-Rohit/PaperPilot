"""Run the RAG eval suite against the latest evaluation dataset and write a Markdown report."""

import asyncio
import logging
from pathlib import Path
from typing import Literal

import dagshub
import mlflow
from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig

from evals.component_evals.generator_metrics import generator_metrics
from evals.component_evals.retriever_metrics import retriever_metrics
from evals.data.generate_evaluation_dataset import generate_evaluation_dataset
from evals.eval_utils import (
    build_test_cases,
    collect_params,
    latest_eval_dataset_file,
    latest_report,
    metrics_from_result,
)
from src.config import get_settings

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"

Component = Literal["all", "retriever", "generator"]


def evaluate_rag_app(component: Component = "all"):
    if component == "retriever":
        metrics = retriever_metrics()
    elif component == "generator":
        metrics = generator_metrics()
    elif component == "all":
        metrics = retriever_metrics() + generator_metrics()
    else:
        raise ValueError(f"Unknown component {component!r} — use 'all', 'retriever' or 'generator'")

    settings = get_settings().deepeval
    test_cases = build_test_cases(settings.retriever)
    print(f"Evaluating {component} on {settings.retriever}: {len(test_cases)} test cases × {len(metrics)} metrics")

    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        identifier=f"{settings.retriever}_{component}",
        hyperparameters={"retriever": settings.retriever, "top_k": get_settings().retrieval.top_k},
        display_config=DisplayConfig(file_type="md", file_output_dir=str(REPORTS_DIR)),
    )


def track_experiment(run_type: str) -> None:

    dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)

    mlflow.set_experiment("paperpilot-rag")

    with mlflow.start_run():
        mlflow.set_tags({"run_type": run_type, "pipeline": "direct"})

        asyncio.run(generate_evaluation_dataset())
        logger.info("Evaluation dataset generated")

        mlflow.log_params(collect_params())
        logger.info("Parameters logged")

        result = evaluate_rag_app()

        mlflow.log_metrics(metrics_from_result(result))
        logger.info("Metrics logged")

        mlflow.log_artifact(str(latest_eval_dataset_file("pinecone")))
        mlflow.log_artifact(str(latest_report()))
        logger.info("Artifacts logged")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)  # only this file's logger
    track_experiment(run_type="noise")
