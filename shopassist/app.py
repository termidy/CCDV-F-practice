from dotenv import load_dotenv
from anthropic import Anthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
client = Anthropic()
app = FastAPI()

MODEL = "claude-sonnet-5-5"
SYSTEM = (
    "You are ShopAssist, an online store's support assistant. "
    "Returns are accepted within 30 days of delivery. "
    "Answer in one or two sentences."
)
history = []


class ChatIn(BaseModel):
    text: str


@app.post("/chat")
def chat(body: ChatIn):
    history.append({"role": "user", "content": body.text})
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=history,
    )
    for block in response.content:
        if block.type == "text":
            reply = block.text
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply, "input_tokens": response.usage.input_tokens}


app.mount("/", StaticFiles(directory="static", html=True))
