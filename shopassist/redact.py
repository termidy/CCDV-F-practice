import re
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"

message = (
    "Hi, I am alex@example.com, phone +353 85 123 4567. "
    "I paid with card 4111 1111 1111 1111 for order "
    "ORD-12345678 and want to return the jacket."
)

PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "CARD": r"\b(?:\d[ -]?){12,15}\d\b",
    "PHONE": r"\+?\d[\d -]{8,}\d",
}


def redact(text):
    for name, pattern in PATTERNS.items():
        text = re.sub(pattern, f"<{name}>", text)
    return text


SYSTEM = (
    "You are ShopAssist, an online store's support assistant. "
    "Returns are accepted within 30 days of delivery. "
    "Personal data arrives as placeholders such as <EMAIL>; "
    "never ask the customer to send it. Answer in two sentences."
)

safe = redact(message)
print("sent to Claude:", safe)

response = client.messages.create(
    model=MODEL,
    max_tokens=200,
    output_config={"effort": "low"},
    system=SYSTEM,
    messages=[{"role": "user", "content": safe}],
)
for block in response.content:
    if block.type == "text":
        print(block.text)
