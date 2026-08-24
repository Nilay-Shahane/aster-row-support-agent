import json
from schemas.graph_state import GraphState
from schemas.safe_order import SafeOrder

def load_orders_db():
    try:
        with open("../ai-agent-intern-test/data/orders.json", "r") as file:
            return json.load(file)["orders"]
    except FileNotFoundError:
        print('Error in loading , knew it')
        return []
        

def order_tool_node(state: GraphState):
    print("--- EXECUTING ORDER TOOL NODE ---")
    order_id = state.get("order_id")

    if not order_id:
        return {"order_context": "Error: No order ID provided by user."}
        # correctly no tool_calls — no lookup was attempted

    clean_id = str(order_id).strip().upper()
    orders_db = load_orders_db()

    raw_order_data = next(
        (order for order in orders_db if order["order_id"] == clean_id),
        None
    )

    if not raw_order_data:
        return {
            "order_context": f"Error: Order {clean_id} not found",
            "tool_calls": [{
                "tool": "order_lookup",
                "args": {"order_id": clean_id},
                "hit": False,
            }],
        }

    if raw_order_data["status"] in ["cancelled", "returned"]:
        raw_order_data["estimated_delivery"] = None

    sanitized_order = SafeOrder(**raw_order_data)

    return {
        "order_context": sanitized_order.model_dump_json(exclude_none=True),
        "tool_calls": [{
            "tool": "order_lookup",
            "args": {"order_id": clean_id},
            "hit": True,
        }],
    }