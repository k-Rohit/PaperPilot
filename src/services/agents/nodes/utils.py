import logging

from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


def get_latest_query(messages: list):
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: list):
    """
    Get the latest context message (mainly for ToolMessage)
    :param messages: List of messages
    :returns: Latest context which it contains
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else ""
    return ""


def get_latest_tool_artifact(messages: list) -> list:
    """Get the structured Document list attached to the latest ToolMessage.

    The retriever tool returns (content, documents) via response_format=
    "content_and_artifact" — `documents` lands on ToolMessage.artifact,
    unflattened, so callers don't have to re-parse it out of the text content.

    :param messages: List of messages
    :returns: List of langchain_core.documents.Document, or [] if none found
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "artifact", None):
            return msg.artifact
    return []
