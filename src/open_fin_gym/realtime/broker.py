import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .alpaca import AlpacaClient

INTERVALS = {
    "1m": ("1Min", 60),
    "5m": ("5Min", 300),
    "15m": ("15Min", 900),
    "1h": ("1Hour", 3600),
}

# Live quotes update sub-second, so anything this old means the feed has
# stalled and the episode cannot be priced fairly.
STALE_QUOTE_SEC = 60


class MarketClosedError(RuntimeError):
    pass


@dataclass
class EpisodeConfig:
    symbols: list[str]
    target_symbols: list[str] = field(default_factory=list)
    context_resolutions: list[dict] = field(
        default_factory=lambda: [{"interval": "1m", "bars": 60}]
    )
    data_resolution: str = "1m"
    max_steps: int = 10
    step_interval_sec: float = 60.0

    def __post_init__(self):
        self.target_symbols = self.target_symbols or list(self.symbols)
        if self.data_resolution not in INTERVALS:
            raise ValueError(f"Unsupported data_resolution {self.data_resolution}")


class Broker:
    def __init__(
        self, cfg: EpisodeConfig, ledger_path: Path, client: AlpacaClient | None = None
    ) -> None:
        """
        Live trading episode over the Alpaca paper account

        Args:
            cfg: Episode configuration
            ledger_path: File the per-step ledger is appended to
            client: Alpaca client, constructed from the environment if None
        """
        self.cfg = cfg
        self.client = client or AlpacaClient()
        self.ledger_path = ledger_path
        self.step_index = 0
        self.started = False
        self.last_equity = 0.0

    def reset(self) -> dict:
        """
        Flatten the account and return the first observation

        Returns:
            Observation dict
        """
        self.client.close_all_positions()
        self.ledger_path.write_text("")
        self.step_index = 0
        self.started = True
        self.last_equity = float(self.client.account()["equity"])
        return self._observation()

    def step(self, action: dict) -> dict:
        """
        Execute one action and advance the episode

        Args:
            action: {"action": "buy"|"sell"|"hold", "symbol": str, "quantity": float}

        Returns:
            {"observation": ..., "reward": float, "done": bool, "info": ...}
        """
        if not self.started:
            raise RuntimeError("reset() must be called before step()")

        # Act, then let the market move before pricing the result, so the reward
        # reflects the action rather than the spread at submission time.
        fill = self._execute(action)
        self.step_index += 1
        time.sleep(self.cfg.step_interval_sec)

        account = self.client.account()
        equity = float(account["equity"])
        reward = equity - self.last_equity
        self.last_equity = equity
        done = self.step_index >= self.cfg.max_steps

        record = {
            "step": self.step_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "fill": fill,
            "equity": equity,
            "reward": reward,
            "positions": self._positions(),
        }
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        return {
            "observation": {} if done else self._observation(),
            "reward": reward,
            "done": done,
            "info": {"fill": fill, "equity": equity},
        }

    def _execute(self, action: dict) -> dict | None:
        side = str(action.get("action", "hold")).lower()
        if side == "hold":
            return None
        symbol = action.get("symbol")
        quantity = float(action.get("quantity", 0))
        if (
            side not in ("buy", "sell")
            or symbol not in self.cfg.symbols
            or quantity <= 0
        ):
            return {"rejected": f"invalid action {action}"}
        try:
            order = self.client.await_fill(
                self.client.submit_order(symbol, quantity, side)["id"]
            )
        except Exception as e:
            return {"rejected": str(e)}
        return {
            "id": order["id"],
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": order["status"],
            "filled_qty": order.get("filled_qty"),
            "filled_avg_price": order.get("filled_avg_price"),
        }

    def _positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self.client.positions()}

    def _observation(self) -> dict:
        quotes = self.client.latest_quotes(self.cfg.symbols)
        self._assert_open(quotes)

        history: dict[str, dict[str, list[dict]]] = {s: {} for s in self.cfg.symbols}
        for entry in self.cfg.context_resolutions:
            interval = entry["interval"]
            bars = self.client.recent_bars(
                self.cfg.symbols, INTERVALS[interval][0], int(entry["bars"])
            )
            for symbol, series in bars.items():
                history[symbol][interval] = series

        account = self.client.account()
        return {
            "step": self.step_index,
            "steps_remaining": self.cfg.max_steps - self.step_index,
            "symbols": {
                s: {"quote": quotes[s], "recent_bars_by_interval": history[s]}
                for s in self.cfg.symbols
            },
            "target_symbols": self.cfg.target_symbols,
            "portfolio": {
                "cash": float(account["cash"]),
                "equity": float(account["equity"]),
                "buying_power": float(account["buying_power"]),
                "positions": self._positions(),
            },
        }

    def _assert_open(self, quotes: dict[str, dict]) -> None:
        # Pre- and post-market sessions still quote, so the clock decides whether
        # the session is tradable and quote age catches a stalled feed.
        if not self.client.is_open():
            raise MarketClosedError("market is closed")
        now = datetime.now(timezone.utc)
        for symbol in self.cfg.symbols:
            quote = quotes.get(symbol)
            if quote is None:
                raise MarketClosedError(f"no quote for {symbol}")
            age = (
                now - datetime.fromisoformat(quote["t"].replace("Z", "+00:00"))
            ).total_seconds()
            if age > STALE_QUOTE_SEC:
                raise MarketClosedError(
                    f"{symbol} quote is {int(age)}s old, feed has stalled"
                )
