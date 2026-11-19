"""Run every golden through the full LangGraph agent and save a dataset DeepEval can grade.

The agent is built with no side effects: no checkpointer, and a live fetch that always
finds "0 papers", so evals never write to Supabase or Pinecone and never call arXiv.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from deepeval.test_case import LLMTestCase

from evals.eval_utils import EVALUATION_DIR, latest_goldens_file
from src.config import get_settings
from src.services.agents.agentic_rag import AgenticRag
from src.services.agents.config import GraphConfig
from src.services.embeddings.factory import make_openai_embeddings_client
from src.services.openai_.factory import make_openai_client
from src.services.pinecone_.factory import make_pinecone_client

logger = logging.getLogger(__name__)

AGENT_DATASET_PREFIX = "agent"


class NoopMetadataFetcher:
    """Stands in for MetadataFetcher during evals: a live fetch always finds 0 papers.

    With no new arxiv_ids the live-fetch tool returns before it touches the DB session,
    the paper repository or the indexer, so nothing is written anywhere.
    """

    async def fetch_and_process_by_topic(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "papers_fetched": 0,
            "papers_skipped_existing": 0,
            "papers_stored": 0,
            "new_arxiv_ids": [],
            "errors": [],
        }


def build_eval_agent() -> AgenticRag:
    """Full agent graph with side-effect-free live fetch and no checkpointer."""
    return AgenticRag(
        pinecone=make_pinecone_client(),
        openaiembeddings=make_openai_embeddings_client(),
        openai_=make_openai_client(),
        graph_config=GraphConfig(),
        checkpointer=None,
        metadata_fetcher=NoopMetadataFetcher(),
        hybrid_indexing_service=None,  # never reached: the no-op fetcher returns 0 new papers
        db_session_factory=contextlib.nullcontext,
    )


def is_answered(result: dict, guardrail_threshold: int) -> bool:
    """True if the agent produced a real answer.

    Not answered when either:
      - the guardrail refused (score below threshold -> out-of-scope template), or
      - no retrieval attempt was graded relevant (retrieval-exhausted fallback message).
    Uses state signals, not answer text.
    """
    score = result.get("guardrail_score")
    if score is None or score < guardrail_threshold:
        return False
    return any(g.is_relevant for g in result.get("grading_results", []))


async def _run_one(agent: AgenticRag, golden: dict, guardrail_threshold: int, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        result = await agent.ask(query=golden["input"], thread_id=str(uuid.uuid4()))
    return {
        "input": golden["input"],
        "actual_output": result["answer"],
        "expected_output": golden.get("expected_output", ""),
        "retrieval_context": result["retrieval_context"],
        "tool_selection": result["tool_selection"],
        "retrieval_attempts": result["retrieval_attempts"],
        "live_fetch_attempted": result["live_fetch_attempted"],
        "answered": is_answered(result, guardrail_threshold),
    }


async def generate_agent_dataset(limit: int | None = None) -> Path:
    """Run goldens through the agent and save evals/data/evaluation/agent_<timestamp>.json."""
    with open(latest_goldens_file()) as f:
        goldens = json.load(f)
    if limit is not None:
        goldens = goldens[:limit]
    logger.info(f"Running {len(goldens)} goldens through the agent")

    agent = build_eval_agent()
    threshold = agent.graph_config.guardrail_threshold
    semaphore = asyncio.Semaphore(get_settings().deepeval.max_concurrent)
    rows = await asyncio.gather(*(_run_one(agent, g, threshold, semaphore) for g in goldens))

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIR / f"{AGENT_DATASET_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(list(rows), indent=2))
    logger.info(f"Saved agent dataset to {path}")
    return path


def trajectory_metrics(rows: list[dict]) -> dict[str, float]:
    """Answered Rate, Avg Retrieval Attempts, Live Fetch Rate. Raises ValueError on empty rows."""
    if not rows:
        raise ValueError("Cannot compute trajectory metrics on an empty dataset")
    return {
        "Answered Rate": mean(float(bool(r["answered"])) for r in rows),
        "Avg Retrieval Attempts": mean(r["retrieval_attempts"] for r in rows),
        "Live Fetch Rate": mean(float(bool(r["live_fetch_attempted"])) for r in rows),
    }


def latest_agent_dataset_file() -> Path:
    files = list(EVALUATION_DIR.glob(f"{AGENT_DATASET_PREFIX}_*.json"))
    if not files:
        raise FileNotFoundError(f"No agent dataset in {EVALUATION_DIR} — run agent_runner first")
    return max(files)


def build_agent_test_cases(path: Path | None = None) -> list[LLMTestCase]:
    rows = json.loads((path or latest_agent_dataset_file()).read_text())
    return [
        LLMTestCase(
            input=r["input"],
            actual_output=r["actual_output"],
            expected_output=r["expected_output"],
            retrieval_context=r["retrieval_context"],
        )
        for r in rows
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Run goldens through the full agent")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N goldens")
    args = parser.parse_args()

    dataset_path = asyncio.run(generate_agent_dataset(limit=args.limit))
    print(f"Saved {dataset_path}")
    for name, value in trajectory_metrics(json.loads(dataset_path.read_text())).items():
        print(f"{name}: {value:.3f}")
