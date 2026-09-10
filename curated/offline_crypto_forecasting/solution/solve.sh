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
    return json.load(urllib.request.urlopen(req, timeout=900))

for _ in range(150):
    try:
        call("/healthz"); break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("broker never became reachable")

data = call("/features")

# Reference policy: predict the mean of the training targets. The features are
# derived quantities rather than prices, so carrying one of them forward would
# be off by orders of magnitude against an absolute-price target.
targets = data["train_ground_truth"]
features = data["features"]
predictions = {}
for symbol, series in features.items():
    y = [float(v) for v in targets[symbol]]
    predictions[symbol] = [sum(y) / len(y)] * len(series)

print(json.dumps(call("/predict", {"predictions": predictions})))
PY
