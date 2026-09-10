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

obs = call("/reset", {})

# Reference policy: a random walk forecast, i.e. predict the current price.
# The task never reports done, so the agent decides when to stop.
targets = [s for s in obs if isinstance(obs[s], dict) and "price" in obs[s]]
for _ in range(10):
    symbol = targets[0]
    action = {"symbol": symbol, "predicted_price": float(obs[symbol]["price"])}
    result = call("/step", action)
    if result["done"]:
        break
    obs = result["observation"]
PY
