# Realtime Stock Trading

## Problem

Trade `SPY`, `QQQ`, and `IWM` on live US equity market data. A broker service
holds the market feed and the account; you drive an episode by submitting one
action per step and are scored on the account it produces.

The episode runs on real market data as it arrives, so there is no data file and
no look-ahead: you only ever see prices up to the current step.

## Protocol

The broker is reachable at `$BROKER_URL` (`http://broker:8000`).

```python
import os, requests

broker = os.environ["BROKER_URL"]
obs = requests.post(f"{broker}/reset").json()

while True:
    action = decide(obs)
    result = requests.post(f"{broker}/step", json=action).json()
    if result["done"]:
        break
    obs = result["observation"]
```

`reset` flattens the account and returns the first observation. Each `step`
submits your order, waits for it to fill, then lets the market move before
returning the next observation and the mark-to-market change over that interval.

## Observation

```python
{
  "step": int,
  "steps_remaining": int,
  "symbols": {
    "SPY": {
      "quote": {"bid": float, "ask": float, "mid": float, "t": str},
      "recent_bars_by_interval": {"1m": [...], "5m": [...], "1h": [...]},
    },
    ...
  },
  "target_symbols": ["SPY", "QQQ"],
  "portfolio": {"cash": float, "equity": float, "buying_power": float,
                "positions": {"SPY": float}},
}
```

`positions` only lists symbols you currently hold, so use
`positions.get(symbol, 0.0)`.

## Action

One action per step:

```python
{"action": "buy" | "sell" | "hold", "symbol": "SPY", "quantity": 10}
```

Orders are market orders against the live book, and a step does not return
until the order has filled. Quantity may be fractional and
must be positive; size it against `buying_power`. Selling more than you hold
opens a short. `{"action": "hold"}` needs no other fields. An invalid action is
rejected and the step still advances, so a malformed action costs you a bar.

## Scoring

Headline metric is `pnl`, the total mark-to-market change across the episode.
Also reported: `sharpe_ratio` over per-step returns, `max_drawdown`,
`win_rate` over steps that traded, and `total_return`.

The episode only runs while the US equity market is open. Outside those hours
`reset` and `step` return `503`.
