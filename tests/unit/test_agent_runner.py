import os

for _k, _v in {
    "OPENAI_API_KEY": "test",
    "PINECONE_API_KEY": "test",
    "POSTGRES_DATABASE_URL": "postgresql://u:p@localhost:5432/db",
}.items():
    os.environ.setdefault(_k, _v)

from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from evals.agent_evals.agent_runner import NoopMetadataFetcher, is_answered, trajectory_metrics  # noqa: E402
from src.services.agents.agentic_rag import AgenticRag  # noqa: E402
from src.services.agents.config import GraphConfig  # noqa: E402
from src.services.embeddings.openai_client import OpenAIEmbeddingsClient  # noqa: E402
from src.services.openai_ import OpenAIClient  # noqa: E402
from src.services.pinecone_.client import PineconeClient  # noqa: E402


def _row(answered, attempts, live):
    return {"answered": answered, "retrieval_attempts": attempts, "live_fetch_attempted": live}


def test_trajectory_metrics_math():
    rows = [_row(True, 1, False), _row(True, 2, True), _row(False, 3, False), _row(True, 2, False)]
    m = trajectory_metrics(rows)
    assert m == {"Answered Rate": 0.75, "Avg Retrieval Attempts": 2.0, "Live Fetch Rate": 0.25}


def test_trajectory_metrics_empty_raises():
    with pytest.raises(ValueError):
        trajectory_metrics([])


def _fake_agent(final_state: dict) -> AgenticRag:
    agent = AgenticRag.__new__(AgenticRag)
    agent.pinecone = MagicMock(spec=PineconeClient)
    agent.openaiembeddings = MagicMock(spec=OpenAIEmbeddingsClient)
    agent.openai_ = MagicMock(spec=OpenAIClient)
    agent.graph_config = GraphConfig()
    agent.graph = SimpleNamespace(ainvoke=AsyncMock(return_value=final_state))
    return agent


async def test_ask_extracts_eval_fields():
    state = {
        "messages": [
            HumanMessage(content="q"),
            ToolMessage(
                content="x",
                name="retrieve_papers",
                tool_call_id="1",
                artifact=[Document(page_content="chunk A"), Document(page_content="chunk B")],
            ),
            ToolMessage(content="y", name="retrieve_papers", tool_call_id="2", artifact=[Document(page_content="chunk A")]),
            ToolMessage(content="z", name="fetch_live_papers", tool_call_id="3", artifact={"papers_fetched": 0}),
            AIMessage(content="final"),
        ],
        "tool_selection": "retrieve",
        "guardrail_result": SimpleNamespace(score=85),
        "live_fetch_attempted": True,
        "retrieval_attempts": 2,
    }
    out = await _fake_agent(state).ask("q", thread_id="t")
    assert out["retrieval_context"] == ["chunk A", "chunk B"]
    assert out["tool_selection"] == "retrieve"
    assert out["guardrail_score"] == 85
    assert out["live_fetch_attempted"] is True
    assert out["answer"] == "final"
    assert out["retrieval_attempts"] == 2


async def test_ask_defaults_when_state_is_sparse():
    out = await _fake_agent({"messages": [AIMessage(content="no")]}).ask("q", thread_id="t")
    assert out["retrieval_context"] == []
    assert out["tool_selection"] is None
    assert out["guardrail_score"] is None
    assert out["live_fetch_attempted"] is False


def test_is_answered():
    rel, irrel = SimpleNamespace(is_relevant=True), SimpleNamespace(is_relevant=False)
    assert is_answered({"guardrail_score": 80, "grading_results": [irrel, rel]}, 60)
    assert not is_answered({"guardrail_score": 30, "grading_results": []}, 60)  # out of scope
    assert not is_answered({"guardrail_score": 80, "grading_results": [irrel, irrel]}, 60)  # fallback
    assert not is_answered({"guardrail_score": None, "grading_results": []}, 60)


async def test_noop_fetcher_returns_zero_paper_stats():
    stats = await NoopMetadataFetcher().fetch_and_process_by_topic(
        search_query="x", max_results=5, process_pdfs=True, store_to_db=True, db_session=None
    )
    assert stats["papers_fetched"] == 0
    assert stats["new_arxiv_ids"] == []
