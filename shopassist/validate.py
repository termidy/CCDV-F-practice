import json
import re
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": ["string", "null"]},
        "item": {"type": "string"},
        "reason": {
            "type": "string",
            "enum": ["too_small", "too_big", "damaged", "other"],
        },
    },
    "required": ["order_id", "item", "reason"],
    "additionalProperties": False,
}

INSTRUCTION = (
    "Extract the return request from this customer email. "
    "Order IDs look like ORD-12345678. If the email gives no "
    "order number, set order_id to null. Never invent one."
)


def validate(data):
    order_id = data["order_id"]
    if order_id is not None and not re.fullmatch(r"ORD-\d{8}", order_id):
        return f"order_id {order_id!r} is not a valid order ID"
    return None


def extract(email):
    messages = [{"role": "user", "content": INSTRUCTION + "\n\n" + email}]
    for attempt in range(3):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=messages,
        )
        for block in response.content:
            if block.type == "text":
                text = block.text
        data = json.loads(text)
        error = validate(data)
        if error is None:
            return data
        print("  attempt", attempt + 1, "rejected:", error)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         error + ". Fix it, or use null."})
    return None


EMAILS = [
    "Hi, I want to return the blue jacket from order "
    "ORD-12345678. It arrived last week but it is too small. Alex",
    "Hello, the shoes I got are damaged. Please tell me how "
    "to send them back. Thanks, Sam",
    "The headphones from order ORD-7788 stopped working after "
    "two days. I want a return. Kim",
]

if __name__ == "__main__":
    for email in EMAILS:
        data = extract(email)
        print(data)
