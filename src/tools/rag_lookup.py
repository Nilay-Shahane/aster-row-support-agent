import json
from schemas.graph_state import GraphState
from rag.vectorstore import VectorStore

vector_store = VectorStore()

def rag_tool_node(state: GraphState):
    query = state.get("query")

    if not query:
        return {"rag_context": "No query provided."}

    results = vector_store.query(
        query_text=query,
        k=8
    )

    return {
        "rag_context": results
    }