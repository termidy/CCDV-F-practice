from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
POLICY = open("policy.txt").read()


def ask(question):
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        output_config={"effort": "low"},
        system=[
            {"type": "text", "text": "You are ShopAssist. Answer "
             "from the policy below in one sentence."},
            {"type": "text", "text": POLICY,
             "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": question}],
    )
    u = response.usage
    print(f"input {u.input_tokens:5} | cache write "
          f"{u.cache_creation_input_tokens:5} | cache read "
          f"{u.cache_read_input_tokens:5}")
    for block in response.content:
        if block.type == "text":
            print("  ", block.text[:70])


ask("Can I return earrings?")
ask("How long does a card refund take?")
ask("Who approves a refund of 250?")
