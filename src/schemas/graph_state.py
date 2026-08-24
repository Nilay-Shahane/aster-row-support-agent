from typing import Annotated, Optional, Literal, TypedDict , Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    # Using TypedDict is required for LangGraph to properly merge state updates
    messages: Annotated[list[BaseMessage], add_messages]
    query: Optional[str]
    order_id: Optional[str]
    order_context: Optional[str]
    tool_calls: Optional[list[dict]]
    rag_context: Optional[list[dict[str, Any]]]
    route: Optional[str]
    answer: Optional[str]
    sources_used: Optional[list[str]]
    confidence: Optional[Literal["grounded", "insufficient", "conflicting"]]
    handoff: Optional[bool]
    injection_detected: Optional[bool]