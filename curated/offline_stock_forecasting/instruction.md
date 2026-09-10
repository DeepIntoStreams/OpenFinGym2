# Offline Stock Forecasting

## Problem

Batch forecasting over historical hourly bars for US equities.

This is a batch task: you receive the whole training set and the test features
in one call, and submit every prediction at once.

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
```

```python
data = requests.get(f"{broker}/features").json()
# data["train_features"], data["train_ground_truth"], data["features"]

predictions = fit_and_predict(data)
score = requests.post(f"{broker}/predict",
                      json={"predictions": predictions}).json()
```

Predictions are keyed by symbol, one series per target symbol, aligned with the
test features you were given.

## Scoring

Headline metric is `mape`. Also reported per symbol and in aggregate: `mse`,
`rmse`, `mae`, `r2`, `pearson`, `directional_accuracy`.
