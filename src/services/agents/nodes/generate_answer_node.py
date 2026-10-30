import logging
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)
load_dotenv()


async def ainvoke_generate_answer(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, list[AIMessage]]:
    """Generate the final answer from the retrieved chunks."""
    logger.info("NODE: generate_answer")
    start_time = time.time()

    graph_config = runtime.context.graph_config
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    if not context:
        context = "No relevant documents found."

    prompt = GENERATE_ANSWER_PROMPT.format(context=context, question=question)

    llm = ChatOpenAI(model=graph_config.model, temperature=graph_config.temperature)
    response = await llm.ainvoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    logger.info(f"answer_len={len(answer)} took={time.time() - start_time:.2f}s")

    return {"messages": [AIMessage(content=answer)]}
