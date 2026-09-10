#!/bin/bash
set -e
mkdir -p /logs/verifier
python -c "
import json, os, urllib.request
url = os.environ.get('BROKER_URL', 'http://broker:8000') + '/score'
with urllib.request.urlopen(url, timeout=120) as r:
    json.dump(json.load(r), open('/logs/verifier/reward.json', 'w'), indent=2)
"
