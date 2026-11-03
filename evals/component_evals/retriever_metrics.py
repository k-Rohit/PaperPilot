"""Retriever metrics: did hybrid search bring back the right chunks?"""

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric

from src.config import get_settings


def retriever_metrics() -> list:
    settings = get_settings()
    return [
        ContextualRecallMetric(threshold=settings.deepeval.threshold, model=settings.deepeval.judge_model, include_reason=True),
        ContextualPrecisionMetric(
            threshold=settings.deepeval.threshold, model=settings.deepeval.judge_model, include_reason=True
        ),
        ContextualRelevancyMetric(
            threshold=settings.deepeval.threshold, model=settings.deepeval.judge_model, include_reason=True
        ),
    ]
