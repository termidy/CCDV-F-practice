import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
SYSTEM = (
    "You are ShopAssist, an online store's support assistant. "
    "Answer in two sentences."
)

CUSTOMERS = {"alex@example.com": "CUS-1001"}
ORDERS = {
    "ORD-87654321": {
        "item": "lamp",
        "status": "delivered",
        "delivered_days_ago": 45,
        "customer_id": "CUS-1001",
    },
}

TOOLS = [
    {
        "name": "get_customer",
        "description": "Verify a customer by email. Returns the "
                       "customer ID.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "name": "lookup_order",
        "description": "Look up an order by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "start_return",
        "description": "Start a return for a delivered order inside "
                       "the 30-day window.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate",
        "description": "Hand the case to a human agent with a short "
                       "reason. Use it when a tool refused and the "
                       "customer still needs help.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


def get_customer(email):
    customer_id = CUSTOMERS.get(email)
    if customer_id is None:
        return {"found": False}
    case["customer_id"] = customer_id
    return {"found": True, "customer_id": customer_id}


def lookup_order(order_id):
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "message": "No order with this ID."}
    case["order_id"] = order_id
    return {"found": True, "order": order}


def start_return(order_id):
    order = ORDERS[order_id]
    if order["delivered_days_ago"] > 30:
        return {"isError": True, "errorCategory": "business",
                "isRetryable": False,
                "customerMessage": "Past the 30-day return window."}
    return {"return_label": "RET-2048"}


def escalate(reason):
    case["escalated"] = True
    return {"ticket": "ESC-507", "reason": reason}


TOOL_FUNCTIONS = {"get_customer": get_customer,
                  "lookup_order": lookup_order,
                  "start_return": start_return,
                  "escalate": escalate}


def run(text):
    messages = [{"role": "user", "content": text}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=600,
            output_config={"effort": "low"},
            system=SYSTEM, tools=TOOLS, messages=messages,
        )
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant",
                         "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = TOOL_FUNCTIONS[block.name](**block.input)
                print(f"  {block.name}({block.input}) -> {output}")
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(output),
                                "is_error": bool(output.get("isError"))})
        messages.append({"role": "user", "content": results})
    for block in response.content:
        if block.type == "text":
            print("ShopAssist:", block.text)
    print("case:", case)

