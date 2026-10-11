import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
SYSTEM = (
    "You are ShopAssist, an online store's support assistant. "
    "Use the tools to look up orders and to start returns. "
    "The start_return tool checks the return policy; do not "
    "judge it yourself. Answer in one or two sentences."
)

ORDERS = {
    "ORD-12345678": {
        "item": "jacket",
        "status": "delivered",
        "delivered_days_ago": 20,
    },
    "ORD-87654321": {
        "item": "lamp",
        "status": "delivered",
        "delivered_days_ago": 45,
    },
}

TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order by its ID. Returns the "
                       "item, the status and how many days ago "
                       "it was delivered.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "start_return",
        "description": "Start a return for a delivered order that "
                       "is inside the 30-day window. Returns the "
                       "return label number.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


def lookup_order(order_id):
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "message": "No order with this ID."}
    return {"found": True, "order": order}


def start_return(order_id):
    order = ORDERS.get(order_id)
    if order["delivered_days_ago"] > 30:
        return {
            "isError": True,
            "errorCategory": "business",
            "isRetryable": False,
            "customerMessage": "This order is past the 30-day "
                               "return window.",
            "developerMessage": "delivered_days_ago="
                                f"{order['delivered_days_ago']} > 30",
        }
    return {"return_label": "RET-2048", "order_id": order_id}


TOOL_FUNCTIONS = {"lookup_order": lookup_order,
                  "start_return": start_return}


def handle(text):
    messages = [{"role": "user", "content": text}]
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            output_config={"effort": "low"},
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant",
                         "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = TOOL_FUNCTIONS[block.name](**block.input)
                print(f"  {block.name} -> {output}")
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(output),
                                "is_error": bool(output.get("isError"))})
        messages.append({"role": "user", "content": results})
    for block in response.content:
        if block.type == "text":
            print("ShopAssist:", block.text)


if __name__ == "__main__":
    handle("Can I return my order ORD-99999999?")
    handle("I want to return the lamp from order ORD-87654321.")
