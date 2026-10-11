import json
from pathlib import Path

from validate import extract

requests = []
for path in sorted(Path("emails").glob("*.txt")):
    data = extract(path.read_text())
    data["email"] = path.name
    data["status"] = "ok" if data["order_id"] else "needs review"
    requests.append(data)
    print(f"{path.name}  {str(data['order_id']):14} "
          f"{data['item']:12} {data['reason']:10} {data['status']}")

Path("return_requests.json").write_text(json.dumps(requests, indent=2))
print(len(requests), "requests saved,",
      sum(r["status"] != "ok" for r in requests), "need review")
