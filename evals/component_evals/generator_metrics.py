"""Generator metrics: is the LLM's answer grounded in the chunks and on-topic?"""

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from src.config import get_settings


def generator_metrics() -> list:
    settings = get_settings()
    return [
        FaithfulnessMetric(threshold=settings.deepeval.threshold, model=settings.deepeval.judge_model, include_reason=True),
        AnswerRelevancyMetric(threshold=settings.deepeval.threshold, model=settings.deepeval.judge_model, include_reason=True),
    ]
