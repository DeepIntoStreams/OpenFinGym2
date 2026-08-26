import os
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

    def latest_bars(self, symbols: list[str]) -> dict[str, dict]:
        params = {"symbols": ",".join(symbols), "feed": self.feed}
        return self._call("GET", f"{DATA_URL}/v2/stocks/bars/latest", params=params)[
            "bars"
        ]

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
