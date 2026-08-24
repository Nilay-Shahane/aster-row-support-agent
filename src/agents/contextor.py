from schemas.graph_state import GraphState
from schemas.contextor_output import ContextorOutput
from llm import llm

def contextor_node(state: GraphState):

    messages = state.get("messages", [])
    if not messages:
        return {}
        
    current_order_id = state.get("order_id")
    structured_llm = llm.with_structured_output(ContextorOutput)

    system_prompt = f"""You are a conversational context analyzer for a support system. 
    Your ONLY job is to rewrite the latest user message so it is fully self-contained and understandable without the conversation history.

    CRITICAL RULES:
    1. Do NOT answer the user's question. Only rewrite their query.
    2. If the user mentions a specific Order ID (usually formatted like ORD-XXXX), extract it EXACTLY as written.
    3. If no Order ID is explicitly mentioned in the current turn, check if one was being actively discussed in the immediate history. If so, include it in your output.
    4. Do not invent or guess an Order ID under any circumstances.
    5. Query Expansion (CRITICAL): If a user mentions a damaged, broken, defective, or incorrect item, you MUST append terms like "reporting window timeframe deadline requirements" to the rewritten query to ensure time-limit policies are retrieved by the search system.
    
    Current active order ID in memory: {current_order_id}"""

    messages_to_pass = [("system", system_prompt)] + messages
    parsed_output = structured_llm.invoke(messages_to_pass)

    print(f"Rewritten Query: {parsed_output.query}")
    print(f"Extracted Order ID: {parsed_output.order_id}")

    final_order_id = parsed_output.order_id if parsed_output.order_id else current_order_id

    return {
        "query": parsed_output.query,
        "order_id": final_order_id
    }