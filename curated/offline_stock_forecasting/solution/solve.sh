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

# Reference policy: reference * (1 + drift), the numpy baseline the task
# instructions name. The target is an absolute price while every feature is
# scale-free, so the reference close is what fixes the level; drift is the
# mean per-bar move over the training split. Carrying the reference across
# unchanged would score well on price error but leaves the direction
# undefined, since sign(predicted - reference) would be zero everywhere.
ref_train = data["reference_train"]
ref_test = data["reference_test"]
train_target = data["train_ground_truth"]

predictions = {}
for symbol, test_ref in ref_test.items():
    moves = [
        float(t) / float(r) - 1.0
        for t, r in zip(train_target[symbol], ref_train[symbol])
        if float(r) != 0.0
    ]
    drift = sum(moves) / len(moves) if moves else 0.0
    predictions[symbol] = [float(r) * (1.0 + drift) for r in test_ref]

print(json.dumps(call("/predict", {"predictions": predictions})))
PY
