import anthropic
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(max_retries=2, timeout=30)

MODEL = "claude-sonnet-5-5"
QUESTION = "Can I return a jacket after twenty days?"


def ask(client, **extra):
    return client.messages.create(
        model=MODEL,
        max_tokens=100,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": QUESTION}],
        **extra,
    )


def attempt(label, fn):
    try:
        fn()
        print(f"{label}: ok")
    except anthropic.APIStatusError as e:
        print(f"{label}: {type(e).__name__} {e.status_code}")
        print("   ", e.body["error"]["message"][:66])
    except anthropic.APIConnectionError as e:
        print(f"{label}: {type(e).__name__}")


attempt("good request", lambda: ask(client))

bad_key = Anthropic(api_key="sk-ant-wrong-key")
attempt("wrong key", lambda: ask(bad_key))

attempt("old parameter",
        lambda: ask(client, extra_body={"temperature": 0.2}))

slow = client.with_options(timeout=0.05, max_retries=0)
attempt("timeout 50 ms", lambda: ask(slow))
