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

for _ in range(60):
    try:
        call("/healthz"); break
    except Exception:
        time.sleep(2)

data = call("/features")

# Reference policy: a random walk forecast, i.e. carry the last observed value.
features = data["features"]
predictions = {
    symbol: [row[-1] if isinstance(row, list) else row for row in series]
    for symbol, series in features.items()
}
print(json.dumps(call("/predict", {"predictions": predictions})))
PY
