"""Observer for the practice recordings: logs every HTTP exchange of the Anthropic client to a JSONL file.
Loaded automatically through PYTHONPATH; the lesson scripts stay unchanged."""
import json, os, time
# the Anthropic SDK 1.x vendors httpx as httpx2; older SDKs use httpx itself - hook whichever exists
_clients = []
for _name in ("httpx2", "httpx"):
    try:
        _clients.append(__import__(_name))
    except ImportError:
        pass
httpx = _clients[0] if _clients else None

TRACE = os.environ.get("TRACE_FILE", os.path.join(os.path.dirname(__file__), "trace.jsonl"))

def _write(rec):
    rec["t"] = time.time()
    with open(TRACE, "a") as f:
        f.write(json.dumps(rec) + "\n")

def _hook(mod):
    _send = mod.Client.send

    def send(self, request, **kw):
        body = None
        try:
            body = json.loads(request.content.decode()) if request.content else None
        except Exception:
            body = request.content.decode(errors="replace")[:2000]
        _write({"event": "request", "method": request.method, "url": str(request.url), "body": body})
        t0 = time.time()
        resp = _send(self, request, **kw)
        if isinstance(body, dict) and body.get("stream") and resp.status_code < 400:
            # a streamed reply must stay a stream: reading it here would hand the script the whole text at once
            rbody = {"streamed": True, "stop_reason": "streamed"}
        else:
            try:
                resp.read()
                rbody = resp.json()
            except Exception:
                rbody = resp.text[:2000] if hasattr(resp, "text") else None
        _write({"event": "response", "status": resp.status_code, "ms": round((time.time() - t0) * 1000), "body": rbody})
        return resp

    mod.Client.send = send


for _mod in _clients:
    _hook(_mod)
if _clients:
    import sys
    _write({"event": "start", "argv": os.environ.get("TRACE_LABEL") or os.path.basename(sys.argv[0] if sys.argv and sys.argv[0] else "")})
