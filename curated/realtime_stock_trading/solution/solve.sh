#!/bin/bash
set -e
uv run --with requests python - <<'PY'
import os, time, requests

broker = os.environ.get("BROKER_URL", "http://broker:8000")
for _ in range(60):
    try:
        requests.get(f"{broker}/healthz", timeout=2).raise_for_status()
        break
    except requests.RequestException:
        time.sleep(2)

obs = requests.post(f"{broker}/reset", json={}, timeout=120).json()

# Reference policy: hold the largest affordable position in the first target
# symbol for the episode, then flatten.
symbol = obs["target_symbols"][0]
price = obs["symbols"][symbol]["quote"]["ask"]
quantity = int(obs["portfolio"]["buying_power"] * 0.2 / price)

while True:
    action = {"action": "hold"}
    if obs["step"] == 0 and quantity > 0:
        action = {"action": "buy", "symbol": symbol, "quantity": quantity}
    elif obs["steps_remaining"] == 1 and obs["portfolio"]["positions"].get(symbol):
        action = {"action": "sell", "symbol": symbol, "quantity": obs["portfolio"]["positions"][symbol]}
    result = requests.post(f"{broker}/step", json=action, timeout=600).json()
    print(f"step {obs['step']} {action['action']} reward={result['reward']:+.4f}", flush=True)
    if result["done"]:
        break
    obs = result["observation"]
PY
