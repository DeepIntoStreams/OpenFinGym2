#!/bin/bash
set -e
python - <<'PY'
import os, time, urllib.request, json

broker = os.environ.get("BROKER_URL", "http://broker:8000")

def call(path, body=None):
    req = urllib.request.Request(
        broker + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    return json.load(urllib.request.urlopen(req, timeout=600))

for _ in range(60):
    try:
        call("/healthz"); break
    except Exception:
        time.sleep(2)

obs = call("/reset", {})

# Reference policy: take a position in the first target symbol on the opening
# bar, hold it, and flatten on the last one.
symbol = obs["portfolio"] and next(iter(obs["symbols"]))
price = obs["symbols"][symbol]["price"]
quantity = int(obs["portfolio"]["cash"] * 0.2 / price)

while True:
    if obs["step"] == 0 and quantity > 0:
        action = {"action": "buy", "symbol": symbol, "quantity": quantity}
    elif obs["steps_remaining"] == 1 and obs["portfolio"]["positions"].get(symbol):
        action = {"action": "sell", "symbol": symbol,
                  "quantity": obs["portfolio"]["positions"][symbol]}
    else:
        action = {"action": "hold"}
    result = call("/step", action)
    if result["done"]:
        break
    obs = result["observation"]
PY
