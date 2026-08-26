#!/bin/bash
set -e
mkdir -p /logs/verifier
uv run --with requests python -c "
import json, os, requests
url = os.environ.get('BROKER_URL', 'http://broker:8000') + '/score'
json.dump(requests.get(url, timeout=30).json(), open('/logs/verifier/reward.json', 'w'), indent=2)
"
