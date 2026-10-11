import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
SYSTEM = (
    "You are ShopAssist, an online store's support assistant. "
    "Returns are accepted within 30 days of delivery."
)
QUESTION = (
    "Explain in five short sentences how I return a jacket "
    "that arrived two weeks ago."
)

t0 = time.time()
response = client.messages.create(
    model=MODEL,
    max_tokens=300,
    output_config={"effort": "low"},
    system=SYSTEM,
    messages=[{"role": "user", "content": QUESTION}],
)
for block in response.content:
    if block.type == "text":
        print(block.text)
print(f"\n[whole reply after {time.time() - t0:.1f} s]")
