"""Generator metrics: is the LLM's answer grounded in the chunks and on-topic?"""

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from src.config import get_settings

settings = get_settings()


def generator_metrics(model: str = "gpt-4.1-mini") -> list:
    return [
        FaithfulnessMetric(threshold=settings.deepeval.threshold, model=model, include_reason=True),
        AnswerRelevancyMetric(threshold=settings.deepeval.threshold, model=model, include_reason=True),
    ]
