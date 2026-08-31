# Realtime Crypto Trading

## Problem

Trading on live crypto market data from Binance. Orders are filled in-process against the current market snapshot with configured slippage.

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
obs = requests.post(f"{broker}/reset", json={}).json()
while True:
    result = requests.post(f"{broker}/step", json=decide(obs)).json()
    if result["done"]:
        break
    obs = result["observation"]
```

## Observation

```python
{
  "step": int, "steps_remaining": int,
  "symbols": {"BTCUSDT": {"symbol", "price", "open", "high", "low", "close",
                    "volume", "timestamp", "recent_bars": [...]}},
  "portfolio": {"cash", "reserved_cash", "positions", "pnl", "value",
                "pending_orders"},
}
```

`positions` only lists symbols you currently hold, so use
`positions.get(symbol, 0.0)`.

## Action

One order intent per step:

```python
{"action": "buy" | "sell" | "hold" | "cancel",
 "symbol": "BTCUSDT", "quantity": 10,
 "order_type": "market" | "limit" | "stop" | "stop_limit",
 "limit_price": float, "stop_price": float,
 "tif": "ioc" | "gtc", "order_id": str}
```

Only `action` is always required; `symbol` and `quantity` are required for
`buy` and `sell`, and `order_id` for `cancel`. Submit several orders in one
step with `{"orders": [...]}`. Shorting is allowed and orders are capped by
buying power. Limit and stop orders queue across bars under `gtc` or expire on
the same bar under `ioc`, and fill pessimistically at the limit or stop price.

## Scoring

Headline metric is `pnl`. Also reported: `sharpe_ratio`, `max_drawdown`,
`win_rate`, `total_return`, `num_trades`. Only trades on `target_symbols` count
toward the metrics; trades on the other symbols are allowed for hedging but are
not scored.
