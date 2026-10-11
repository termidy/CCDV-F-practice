import base64
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"

with open("label.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode()

response = client.messages.create(
    model=MODEL,
    max_tokens=300,
    output_config={"effort": "low"},
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": "This is a photo of a shipping label. "
                        "What is the order number, and which "
                        "city was it shipped to?",
            },
        ],
    }],
)
for block in response.content:
    if block.type == "text":
        print(block.text)
print("input tokens:", response.usage.input_tokens)
