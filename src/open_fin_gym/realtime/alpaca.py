import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

DATA_URL = "https://data.alpaca.markets"
PAPER_URL = "https://paper-api.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


class AlpacaClient:
    def __init__(self, feed: str = "iex", timeout_sec: float = 15.0) -> None:
        """
        Alpaca market-data and paper-trading client

        Args:
            feed: Market data feed, iex on the free tier
            timeout_sec: Per-request timeout
        """
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise AlpacaError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set")
        self.feed = feed
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update(
            {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        )

    def _call(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self.session.request(method, url, timeout=self.timeout_sec, **kwargs)
        if response.status_code >= 400:
            raise AlpacaError(
                f"{method} {url} -> {response.status_code} {response.text}"
            )
        return response.json() if response.content else None

    def is_open(self) -> bool:
        return bool(self._call("GET", f"{PAPER_URL}/v2/clock")["is_open"])

    def latest_quotes(self, symbols: list[str]) -> dict[str, dict]:
        # Quotes rather than bars: a minute bar only publishes once its minute
        # closes, so at the open the newest bar is still a pre-market one.
        params = {"symbols": ",".join(symbols), "feed": self.feed}
        quotes = self._call(
            "GET", f"{DATA_URL}/v2/stocks/quotes/latest", params=params
        )["quotes"]
        return {
            s: {
                "bid": q["bp"],
                "ask": q["ap"],
                "mid": (q["bp"] + q["ap"]) / 2,
                "t": q["t"],
            }
            for s, q in quotes.items()
        }

    def recent_bars(
        self, symbols: list[str], interval: str, bars: int
    ) -> dict[str, list[dict]]:
        # Ask for a window well past `bars` worth of time: the feed skips
        # closed sessions, so a wall-clock window under-fills near the open.
        minutes = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}[interval]
        start = datetime.now(timezone.utc) - timedelta(minutes=minutes * bars * 8)
        params = {
            "symbols": ",".join(symbols),
            "timeframe": interval,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": bars * len(symbols),
            "feed": self.feed,
        }
        found = self._call("GET", f"{DATA_URL}/v2/stocks/bars", params=params)["bars"]
        return {s: found.get(s, [])[-bars:] for s in symbols}

    def account(self) -> dict:
        return self._call("GET", f"{PAPER_URL}/v2/account")

    def positions(self) -> list[dict]:
        return self._call("GET", f"{PAPER_URL}/v2/positions")

    def close_all_positions(self) -> None:
        self._call(
            "DELETE", f"{PAPER_URL}/v2/positions", params={"cancel_orders": "true"}
        )

    def order(self, order_id: str) -> dict:
        return self._call("GET", f"{PAPER_URL}/v2/orders/{order_id}")

    def await_fill(self, order_id: str, timeout_sec: float = 30.0) -> dict:
        """
        Poll an order until it leaves the open state

        Args:
            order_id: Order to poll
            timeout_sec: How long to wait before giving up

        Returns:
            The order in its final observed state
        """
        deadline = time.time() + timeout_sec
        while True:
            order = self.order(order_id)
            if order["status"] not in (
                "new",
                "accepted",
                "pending_new",
                "partially_filled",
            ):
                return order
            if time.time() >= deadline:
                return order
            time.sleep(0.5)

    def submit_order(self, symbol: str, quantity: float, side: str) -> dict:
        return self._call(
            "POST",
            f"{PAPER_URL}/v2/orders",
            json={
                "symbol": symbol,
                "qty": str(quantity),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            },
        )
