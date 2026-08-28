# Realtime Prediction Market Forecasting

## Problem

Estimate the probability that each of a set of live Polymarket prediction
markets resolves YES. Markets are discovered at trial setup rather than read
from a fixed list, because each market resolves once and is never reused. Only
binary markets resolving within the next few hours are included.

This is a single-shot task: you receive the whole market universe, submit one
batch of probabilities, and the episode ends. Scoring is deferred until the
markets actually resolve.

## Protocol

The broker is reachable at `$BROKER_URL` (`http://broker:8000`).

```python
import os, time, requests

broker = os.environ["BROKER_URL"]
for _ in range(150):
    try:
        requests.get(f"{broker}/healthz", timeout=2).raise_for_status()
        break
    except requests.RequestException:
        time.sleep(2)
else:
    raise SystemExit("broker never became reachable")

universe = requests.post(f"{broker}/reset", json={}).json()
predictions = [
    {"symbol": m["symbol"], "predicted_yes_probability": estimate(m)}
    for m in universe["markets"]
]
requests.post(f"{broker}/step", json={"predictions": predictions}).json()
```

The single `step` ends the episode and returns `done: true`.

## Observation

```python
{"markets": [{"symbol", "question", "description", "categories",
              "current_price", "best_bid", "best_ask", "outcomes",
              "outcome_prices", "liquidity", "resolution_at",
              "active", "closed"}]}
```

`current_price` is the market's own YES price, and `resolution_at` is when it
settles. Around twenty markets are discovered per trial, all binary and all
resolving within a few hours.

## Action

One probability per market:

```python
{"symbol": str, "predicted_yes_probability": float}  # in [0, 1]
```

Markets outside the discovered universe are ignored.

## Scoring

Headline metric is `brier_score`, lower being better, computed once each market
resolves. Until then the episode reports `status_deferred`. Markets that resolve
at exactly 0.5 are dropped from scoring.
