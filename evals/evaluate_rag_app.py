"""Run the RAG eval suite against the latest evaluation dataset and write a Markdown report.

Run from the repo root:
    uv run python -m evals.evaluate_rag_app              # retriever + generator
    uv run python -m evals.evaluate_rag_app retriever    # just one component
"""

import sys
from pathlib import Path
from typing import Literal

from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig

from evals.component_evals.generator_metrics import generator_metrics
from evals.component_evals.retriever_metrics import retriever_metrics
from evals.eval_utils import build_test_cases
from src.config import get_settings

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


if __name__ == "__main__":
    evaluate_rag_app(sys.argv[1] if len(sys.argv) > 1 else "all")
