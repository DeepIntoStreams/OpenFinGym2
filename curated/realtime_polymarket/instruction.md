# Realtime Prediction Market Forecasting

## Problem

Probability forecasting on live Polymarket prediction markets.

Markets are discovered at trial setup rather than read from a fixed symbol list,
because each prediction market resolves once and is never reused.

## Protocol

The broker is reachable at `$BROKER_URL` (`http://broker:8000`).

```python
import os, time, requests

broker = os.environ["BROKER_URL"]
for _ in range(60):
    try:
        requests.get(f"{broker}/healthz", timeout=2).raise_for_status()
        break
    except requests.RequestException:
        time.sleep(2)
```

```python
obs = requests.post(f"{broker}/reset", json={}).json()
while True:
    result = requests.post(f"{broker}/step", json=forecast(obs)).json()
    if result["done"]:
        break
    obs = result["observation"]
```

## Action

A probability for the market's outcome:

```python
{"market_id": str, "probability": float}
```

## Scoring

Headline metric is `brier_score`, lower being better.
