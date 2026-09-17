"""Binance REST + WebSocket data provider."""

import json as _json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from open_fin_gym.realtime.data_providers.base import (
    MarketSnapshot,
    OrderBookSnapshot,
    interval_to_seconds,
    to_binance_interval,
)

logger = logging.getLogger(__name__)

_MAX_KLINES_PER_REQUEST = 1000


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class _AggTradeBarBuilder:
    """Synthesise OHLCV bars at ``interval_s`` from a per-trade aggTrade stream.

    Buckets with no trades are skipped, leaving a gap in the timestamps.
    """

    def __init__(
        self,
        *,
        symbol: str,
        interval_s: int,
        callback: Callable[[MarketSnapshot], None],
        rest: "BinanceProvider",
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._symbol = symbol
        self._interval_ms = interval_s * 1000
        self._callback = callback
        self._rest = rest
        # Sequence-ID continuity check; None until first message.
        self._last_a: int | None = None
        # In-progress bar state; None until first trade in this session.
        self._bar_open_ms: int | None = None
        self._o = 0.0
        self._h = 0.0
        self._l = 0.0
        self._c = 0.0
        self._v = 0.0

    def _bucket(self, trade_time_ms: int) -> int:
        return (trade_time_ms // self._interval_ms) * self._interval_ms

    def _emit_current(self) -> None:
        """Emit a snapshot of the current in-progress bar."""
        if self._bar_open_ms is None:
            return
        snap = MarketSnapshot(
            symbol=self._symbol,
            timestamp=_from_ms(self._bar_open_ms),
            price=self._c,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            volume=self._v,
        )
        self._callback(snap)

    def _apply_trade(
        self, trade_time_ms: int, price: float, qty: float, emit: bool
    ) -> None:
        bucket = self._bucket(trade_time_ms)
        if self._bar_open_ms is None:
            # First trade ever — open a fresh bar.
            self._bar_open_ms = bucket
            self._o = self._h = self._l = self._c = price
            self._v = qty
        elif bucket == self._bar_open_ms:
            # Same bar — fold in.
            if price > self._h:
                self._h = price
            if price < self._l:
                self._l = price
            self._c = price
            self._v += qty
        else:
            # Emit the just-closed bar, then open the next one, so later live
            # trades land in the right bucket even for backfill callers.
            self._emit_current()
            self._bar_open_ms = bucket
            self._o = self._h = self._l = self._c = price
            self._v = qty
        if emit:
            self._emit_current()

    def on_aggtrade(self, msg: dict[str, Any]) -> None:
        """Process one ``@aggTrade`` WS message."""
        a = int(msg["a"])
        if self._last_a is not None:
            if a == self._last_a:
                logger.debug(
                    "aggTrade duplicate a=%d (ignored) sym=%s", a, self._symbol
                )
                return
            if a < self._last_a:
                logger.debug(
                    "aggTrade out-of-order a=%d < last_a=%d (ignored) sym=%s",
                    a,
                    self._last_a,
                    self._symbol,
                )
                return
            if a > self._last_a + 1:
                self._bridge_gap(self._last_a + 1, a - 1)
        T = int(msg["T"])
        price = float(msg["p"])
        qty = float(msg["q"])
        self._apply_trade(T, price, qty, emit=True)
        self._last_a = a

    def _bridge_gap(self, from_id: int, to_id: int) -> None:
        """Fetch ``[from_id, to_id]`` from REST and apply each trade.

        On failure it logs and stops, leaving the in-progress bar slightly stale.
        """
        logger.warning(
            "aggTrade gap detected sym=%s from_id=%d to_id=%d size=%d — REST backfill",
            self._symbol,
            from_id,
            to_id,
            to_id - from_id + 1,
        )
        cursor = from_id
        applied = 0
        while cursor <= to_id:
            try:
                batch = self._rest.get_agg_trades(
                    self._symbol, from_id=cursor, limit=1000
                )
            except Exception:
                logger.warning(
                    "aggTrade backfill REST call failed sym=%s cursor=%d",
                    self._symbol,
                    cursor,
                    exc_info=True,
                )
                break
            if not batch:
                logger.warning(
                    "aggTrade backfill returned empty batch sym=%s cursor=%d",
                    self._symbol,
                    cursor,
                )
                break
            stop = False
            for trade in batch:
                a = int(trade["a"])
                if a > to_id:
                    stop = True
                    break
                self._apply_trade(
                    int(trade["T"]),
                    float(trade["p"]),
                    float(trade["q"]),
                    emit=False,
                )
                applied += 1
            last_a_in_batch = int(batch[-1]["a"])
            cursor = last_a_in_batch + 1
            if stop or len(batch) < 1000:
                break
        logger.info(
            "aggTrade gap bridged sym=%s applied=%d/%d",
            self._symbol,
            applied,
            to_id - from_id + 1,
        )
        # Surface the post-backfill state once.
        self._emit_current()


class BinanceProvider:
    """Market data from the Binance public REST API + WebSocket streams."""

    name = "binance"

    _WS_BASE = "wss://stream.binance.com:9443/ws"

    # No client-side throttle against Binance's request-weight ceiling: warn
    # when it runs low and retry once on 429.
    _WEIGHT_LIMIT_PER_MIN: int = 1200
    _WEIGHT_WARN_THRESHOLD: float = 0.8  # warn when used-weight > 80%

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        rate_limit_per_min: int | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._session = requests.Session()
        # rate_limit_per_min no longer sleeps; a non-default value becomes the
        # soft warning threshold instead.
        if rate_limit_per_min is not None:
            self._weight_limit = int(rate_limit_per_min)
        else:
            self._weight_limit = self._WEIGHT_LIMIT_PER_MIN

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(url, params=params, timeout=15)
        self._inspect_rate_limit_headers(resp)
        if resp.status_code == 429:
            # Honor the server-advised back-off and retry exactly once.
            retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
            logger.warning(
                "binance 429; sleeping %.2fs and retrying once (path=%s)",
                retry_after,
                path,
            )
            time.sleep(retry_after)
            resp = self._session.get(url, params=params, timeout=15)
            self._inspect_rate_limit_headers(resp)
        resp.raise_for_status()
        return resp.json()

    def _inspect_rate_limit_headers(self, resp: requests.Response) -> None:
        # Binance reports rolling request-weight usage in this header; warn past
        # the threshold so callers can back off before the 429s start.
        used_raw = resp.headers.get("X-MBX-USED-WEIGHT-1M")
        if used_raw is None:
            return
        try:
            used = int(used_raw)
        except ValueError:
            return
        if used > self._weight_limit * self._WEIGHT_WARN_THRESHOLD:
            logger.warning(
                "binance request weight at %d/%d (>%.0f%%); consider backing off",
                used,
                self._weight_limit,
                self._WEIGHT_WARN_THRESHOLD * 100,
            )

    @staticmethod
    def _parse_retry_after(raw: str | None) -> float:
        # Retry-After may be seconds or an HTTP-date; Binance sends seconds, so
        # fall back to 1.0s if it will not parse.
        if raw is None:
            return 1.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 1.0

    # ── DataProvider interface ────────────────────────────────────────

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        # The in-progress kline, not the ticker, so the snapshot carries OHLCV
        # and its minute-boundary timestamp dedups repeated calls in the buffer.
        data = self._get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": "1m", "limit": 1},
        )
        if not data:
            raise ValueError(f"No kline data returned for {symbol}")
        k = data[0]
        return MarketSnapshot(
            symbol=symbol,
            timestamp=_from_ms(int(k[0])),
            price=float(k[4]),  # close == latest tick price for in-progress bar
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
        )

    def get_price_at(
        self, symbol: str, at: datetime, interval: str = "1m"
    ) -> MarketSnapshot:
        start_ms = _ms(at)
        data = self._get(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": to_binance_interval(interval),
                "startTime": start_ms,
                "limit": 1,
            },
        )
        if not data:
            raise ValueError(f"No kline data for {symbol} at {at.isoformat()}")
        k = data[0]
        return MarketSnapshot(
            symbol=symbol,
            timestamp=_from_ms(k[0]),
            price=float(k[4]),  # close
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
        )

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketSnapshot]:
        bi = to_binance_interval(interval)
        all_bars: list[MarketSnapshot] = []
        cursor_ms = _ms(start)
        end_ms = _ms(end)

        while cursor_ms < end_ms:
            data = self._get(
                "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": bi,
                    "startTime": cursor_ms,
                    "endTime": end_ms,
                    "limit": _MAX_KLINES_PER_REQUEST,
                },
            )
            if not data:
                break
            for k in data:
                all_bars.append(
                    MarketSnapshot(
                        symbol=symbol,
                        timestamp=_from_ms(k[0]),
                        price=float(k[4]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                    )
                )
            cursor_ms = int(data[-1][0]) + 1  # next ms after last bar
            if len(data) < _MAX_KLINES_PER_REQUEST:
                break

        return all_bars

    # ── Aggregate trades (REST) ──────────────────────────────────────

    def get_agg_trades(
        self,
        symbol: str,
        *,
        from_id: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch aggregate trades via ``GET /api/v3/aggTrades``."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "limit": min(int(limit), 1000),
        }
        if from_id is not None:
            params["fromId"] = int(from_id)
        elif start_time is not None:
            params["startTime"] = _ms(start_time)
            if end_time is not None:
                params["endTime"] = _ms(end_time)
        return self._get("/api/v3/aggTrades", params) or []

    # ── Order book (REST) ────────────────────────────────────────────

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        """Fetch L2 order book via ``GET /api/v3/depth``."""
        data = self._get("/api/v3/depth", {"symbol": symbol, "limit": depth})
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            best_bid=bids[0][0] if bids else None,
            best_bid_qty=bids[0][1] if bids else None,
            best_ask=asks[0][0] if asks else None,
            best_ask_qty=asks[0][1] if asks else None,
            bids=bids,
            asks=asks,
        )

    # ── WebSocket streaming ──────────────────────────────────────────

    def supports_websocket(self) -> bool:  # noqa: PLR6301
        return True

    async def subscribe_bars(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[MarketSnapshot], None],
    ) -> None:
        """Stream bars synthesised from ``@aggTrade`` ticks.

        The first bar can miss trades that land between the REST backfill and the
        socket opening.
        """
        import websockets  # optional dependency

        stream = f"{symbol.lower()}@aggTrade"
        url = f"{self._WS_BASE}/{stream}"
        builder = _AggTradeBarBuilder(
            symbol=symbol,
            interval_s=interval_to_seconds(interval),
            callback=callback,
            rest=self,
        )
        async with websockets.connect(url) as ws:
            async for raw in ws:
                try:
                    msg = _json.loads(raw)
                    builder.on_aggtrade(msg)
                except Exception:
                    logger.warning(
                        "aggTrade processing error symbol=%s",
                        symbol,
                        exc_info=True,
                    )

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable[[OrderBookSnapshot], None],
    ) -> None:
        """Stream top-20 order book snapshots via Binance WebSocket."""
        import websockets  # optional dependency

        stream = f"{symbol.lower()}@depth20@100ms"
        url = f"{self._WS_BASE}/{stream}"
        async with websockets.connect(url) as ws:
            async for raw in ws:
                data = _json.loads(raw)
                bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
                snap = OrderBookSnapshot(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    best_bid=bids[0][0] if bids else None,
                    best_bid_qty=bids[0][1] if bids else None,
                    best_ask=asks[0][0] if asks else None,
                    best_ask_qty=asks[0][1] if asks else None,
                    bids=bids,
                    asks=asks,
                )
                callback(snap)
