from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..prompts import GUARDRAIL_PROMPT
from ..state import AgentState
from .utils import get_latest_query

load_dotenv()


async def score_user_query(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, GuardrailScoring]:
    """Score the query 0-100 for on-topic-ness."""
    graph_config = runtime.context.graph_config
    query = get_latest_query(state["messages"])
    prompt = GUARDRAIL_PROMPT.format(question=query)

    model = ChatOpenAI(model=graph_config.model, temperature=0.0)
    scorer_llm = model.with_structured_output(GuardrailScoring)

    response = await scorer_llm.ainvoke(prompt)
    return {"guardrail_result": response}


def route(state: AgentState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Route based on the guardrail score."""
    threshold = runtime.context.graph_config.guardrail_threshold
    score = state["guardrail_result"].score
    return "continue" if score >= threshold else "out_of_scope"
