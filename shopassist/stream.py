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
first = None
with client.messages.stream(
    model=MODEL,
    max_tokens=300,
    output_config={"effort": "low"},
    system=SYSTEM,
    messages=[{"role": "user", "content": QUESTION}],
) as stream:
    for text in stream.text_stream:
        if first is None:
            first = time.time() - t0
        print(text, end="", flush=True)
    message = stream.get_final_message()
print(f"\n[first words after {first:.1f} s, "
      f"whole reply after {time.time() - t0:.1f} s]")
print("stop_reason:", message.stop_reason)
print("output tokens:", message.usage.output_tokens)
