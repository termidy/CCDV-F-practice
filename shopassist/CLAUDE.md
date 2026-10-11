# ShopAssist

A support assistant for an online store, built on the Claude API.

## Conventions
- Tool functions return dicts and never raise. A failure is a structured
  error: isError, errorCategory, isRetryable, customerMessage, developerMessage.
- "Not found" is data, not an error: return found: false with a message.
- Order IDs look like ORD-12345678 (ORD- and eight digits).
- Run scripts with `python <file>.py` from this folder; the venv is on PATH.
