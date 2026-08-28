#!/bin/bash
set -e
python - <<'PY'
import json, os, time, urllib.request

broker = os.environ.get("BROKER_URL", "http://broker:8000")

def call(path, body=None):
    req = urllib.request.Request(
        broker + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    return json.load(urllib.request.urlopen(req, timeout=600))

for _ in range(150):
    try:
        call("/healthz"); break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("broker never became reachable")

markets = call("/reset", {})

# Reference policy: quote back the market's own YES price.
predictions = [
    {"symbol": m["symbol"],
     "predicted_yes_probability": float(m.get("current_price") or 0.5)}
    for m in markets["markets"]
]
print(json.dumps(call("/step", {"predictions": predictions}))[:200])
PY
