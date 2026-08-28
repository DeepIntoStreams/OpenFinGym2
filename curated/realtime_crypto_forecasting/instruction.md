# Realtime Crypto Forecasting

## Problem

Forecasting on live crypto market data. Each step submits one independent prediction of the price a fixed horizon ahead.

Predictions are scored once their horizon elapses, so `step` returns a reward of
zero and a `prediction_id`. The episode score stays deferred until the ground
truth exists.

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
    result = requests.post(f"{broker}/step", json=predict(obs)).json()
    if result["done"]:
        break
    obs = result["observation"]
```

## Observation

Keyed by symbol:

```python
{"BTCUSDT": {"symbol", "price", "timestamp", "history_available",
          "recent_bars": [...]}}
```

## Action

One prediction per step:

```python
{"symbol": "BTCUSDT", "predicted_price": float}
```

`predicted_price` is the absolute price you expect at the horizon; direction is
derived from its sign against the entry price. A `direction` of `"long"` or
`"short"` may be supplied instead, but if both are given they must agree.
Predictions for symbols outside `target_symbols` are dropped with a warning and
the step still advances.

## Scoring

Headline metric is `price_mape`. The episode reports `status_deferred` and
`n_predictions` until every horizon has elapsed.
