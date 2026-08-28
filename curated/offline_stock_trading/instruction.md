# Offline Stock Trading

## Problem

Sequential trading over historical hourly bars for `SPY`, `QQQ` and `IWM`.
A broker service holds the market data and the account, and replays it one bar
at a time. You only ever see bars up to the current step, so no look-ahead is
possible. Headline metric is `pnl`.

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
  "symbols": {"SPY": {"symbol", "price", "open", "high", "low", "close",
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
 "symbol": "SPY", "quantity": 10,
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
