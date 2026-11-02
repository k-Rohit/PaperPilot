"""Retriever metrics: did hybrid search bring back the right chunks?"""

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric

from src.config import get_settings

settings = get_settings()


def retriever_metrics(
    model: str = "gpt-4.1-mini",
) -> list:
    return [
        ContextualRecallMetric(threshold=settings.deepeval.threshold, model=model, include_reason=True),
        ContextualPrecisionMetric(threshold=settings.deepeval.threshold, model=model, include_reason=True),
        ContextualRelevancyMetric(threshold=settings.deepeval.threshold, model=model, include_reason=True),
    ]
