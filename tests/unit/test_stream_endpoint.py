"""Unit tests for POST /api/v1/agentic_ask/stream and AgenticRag.stream().

Everything is faked: no OpenAI, Pinecone, Supabase or real graph is touched.
"""

import json
import os

for _k, _v in {
    "OPENAI_API_KEY": "test",
    "PINECONE_API_KEY": "test",
    "POSTGRES_DATABASE_URL": "postgresql://u:p@localhost:5432/db",
}.items():
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402

from src.dependencies import get_agentic_rag  # noqa: E402
from src.routers import agentic_ask  # noqa: E402
from src.services.agents import agentic_rag as agentic_rag_module  # noqa: E402
from src.services.agents.agentic_rag import AgenticRag  # noqa: E402
from src.services.agents.models import GradingResult, GuardrailScoring, SourceItem  # noqa: E402

BODY = {"query": "What is attention?", "thread_id": "t-1"}


def parse_sse(text: str) -> list[dict]:
    frames = [f for f in text.split("\n\n") if f.strip()]
    events = []
    for frame in frames:
        assert frame.startswith("data: "), frame
        events.append(json.loads(frame[len("data: ") :]))
    return events


def make_client(agent) -> TestClient:
    app = FastAPI()
    app.include_router(agentic_ask.router, prefix="/api/v1")
    app.dependency_overrides[get_agentic_rag] = lambda: agent
    return TestClient(app)


class FakeAgent:
    async def stream(self, query, user_id="api_user", thread_id=None):
        yield {"type": "step", "text": "Validated query scope (score: 85/100)"}
        yield {"type": "step", "text": "Retrieved documents (1 attempt(s))"}
        yield {"type": "token", "text": "Attention "}
        yield {"type": "token", "text": "is all you need."}
        yield {
            "type": "done",
            "query": query,
            "answer": "Attention is all you need.",
            "sources": ["https://arxiv.org/pdf/1706.03762"],
            "sources_detailed": [{"url": "https://arxiv.org/pdf/1706.03762"}],
            "reasoning_steps": ["Validated query scope (score: 85/100)"],
            "chunks_used": 1,
            "search_mode": "agentic",
        }


class RaisingAgent:
    async def stream(self, query, user_id="api_user", thread_id=None):
        yield {"type": "step", "text": "Validated query scope (score: 85/100)"}
        raise RuntimeError("boom")


# ─── endpoint ────────────────────────────────────────────────────────────


def test_stream_event_order_and_headers():
    resp = make_client(FakeAgent()).post("/api/v1/agentic_ask/stream", json=BODY)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"

    events = parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert types == ["step", "step", "token", "token", "done"]
    assert events[-1]["answer"] == "".join(e["text"] for e in events if e["type"] == "token")
    assert events[-1]["search_mode"] == "agentic"


def test_stream_error_event_on_exception():
    resp = make_client(RaisingAgent()).post("/api/v1/agentic_ask/stream", json=BODY)
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    assert [e["type"] for e in events] == ["step", "error"]
    assert "boom" in events[-1]["detail"]


def test_stream_validates_before_streaming():
    client = make_client(FakeAgent())
    assert client.post("/api/v1/agentic_ask/stream", json={"query": "hi"}).status_code == 400
    assert client.post("/api/v1/agentic_ask/stream", json={"query": "   ", "thread_id": "t"}).status_code == 400


# ─── AgenticRag.stream() ─────────────────────────────────────────────────


class FakeGraph:
    def __init__(self, items):
        self.items = items
        self.kwargs = None

    async def astream(self, state_input, **kwargs):
        self.kwargs = kwargs
        for item in self.items:
            yield item


def make_rag(items) -> AgenticRag:
    rag = AgenticRag.__new__(AgenticRag)  # bypass __init__: no real services
    rag.pinecone = rag.openaiembeddings = rag.openai_ = rag.graph_config = None
    rag.graph = FakeGraph(items)
    return rag


@pytest.fixture(autouse=True)
def _fake_context(monkeypatch):
    monkeypatch.setattr(agentic_rag_module, "Context", lambda **kw: kw)


async def collect(rag, query="What is attention?"):
    return [e async for e in rag.stream(query, thread_id="t-1")]


@pytest.mark.asyncio
async def test_rag_stream_only_streams_generate_node_tokens():
    source = SourceItem(arxiv_id="1706.03762", title="Attn", authors=["A"], url="http://x/1", relevance_score=0.9)
    items = [
        ("updates", {"guardrail_node": {"guardrail_result": GuardrailScoring(score=85, reason="ok")}}),
        # a token from another node's LLM call must be ignored
        ("messages", (AIMessageChunk(content="GRADER"), {"langgraph_node": "grade_document_node"})),
        ("updates", {"retrieve_node": {"retrieval_attempts": 1, "messages": []}}),
        (
            "updates",
            {
                "grade_document_node": {
                    "grading_results": [GradingResult(document_id="d", is_relevant=True, reasoning="r")],
                    "relevant_sources": [source],
                }
            },
        ),
        ("messages", (AIMessageChunk(content="Hello "), {"langgraph_node": "generate_answer_node"})),
        ("messages", (AIMessageChunk(content="world"), {"langgraph_node": "generate_answer_node"})),
        # LangGraph also emits the node's final full AIMessage in messages mode; must not be re-sent
        ("messages", (AIMessage(content="Hello world"), {"langgraph_node": "generate_answer_node"})),
        ("updates", {"generate_answer_node": {"messages": [AIMessage(content="Hello world")]}}),
    ]
    events = await collect(make_rag(items))

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Hello ", "world"]
    steps = [e["text"] for e in events if e["type"] == "step"]
    assert steps == [
        "Validated query scope (score: 85/100)",
        "Retrieved documents (1 attempt(s))",
        "Graded documents (1 relevant)",
        "Generated answer from context",
    ]
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Hello world"
    assert done["sources"] == ["http://x/1"]
    assert done["chunks_used"] == 1
    assert done["search_mode"] == "agentic"
    assert [e["type"] for e in events].count("done") == 1


@pytest.mark.asyncio
async def test_rag_stream_sends_refusal_as_single_token_when_no_llm_stream():
    items = [
        ("updates", {"guardrail_node": {"guardrail_result": GuardrailScoring(score=10, reason="off")}}),
        ("updates", {"out_of_scope_node": {"messages": [AIMessage(content="I can only help with papers.")]}}),
    ]
    events = await collect(make_rag(items))
    assert [e["type"] for e in events] == ["step", "step", "token", "done"]
    assert events[2]["text"] == "I can only help with papers."
    assert events[-1]["answer"] == "I can only help with papers."


@pytest.mark.asyncio
async def test_rag_stream_requires_query_and_thread_id():
    rag = make_rag([])
    with pytest.raises(ValueError):
        await collect(rag, query="  ")
    with pytest.raises(ValueError):
        _ = [e async for e in rag.stream("q", thread_id=None)]
