import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from ..context import Context
from ..models import CondensedQuery
from ..prompts import CONDENSE_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)
load_dotenv()


async def ainvoke_condense_followup(state: AgentState, runtime: Runtime[Context]) -> dict:
    """
    This will be the first node in the graph, at the first request it performs
    no operation but subsequent requests it will condense the followup query into a standalone query.
    """
    messages = state["messages"]
    if _count_human_messages(messages) <= 1:
        return {}

    query = get_latest_query(messages)
    history = _format_history(messages)
    if not history:
        return {}
    prompt = CONDENSE_PROMPT.format(history=history, question=query)
    try:
        model = ChatOpenAI(model=runtime.context.graph_config.model, temperature=runtime.context.graph_config.temperature)
        condenser = model.with_structured_output(CondensedQuery)
        result: CondensedQuery = await condenser.ainvoke(prompt)
        standalone = result.standalone_query.strip()
    except Exception as e:
        logger.warning(f"Failed to condense followup query: {e}")
        return {}

    if not standalone or standalone == query:
        return {}

    logger.info(f"Condensed followup query: {standalone}")
    return {"messages": [HumanMessage(content=standalone)]}


def _count_human_messages(messages: list) -> int:
    """Count the number of HumanMessage instances in the message list"""
    return sum(1 for msg in messages if isinstance(msg, HumanMessage))


def _format_history(messages: list) -> str:
    """Render every turn EXCEPT the latest HumanMessage (the new follow-up itself)."""
    lines = []
    for m in messages[:-1]:  # Exclude the last message
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)
