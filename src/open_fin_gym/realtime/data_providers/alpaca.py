"""Alpaca Markets REST + WebSocket data provider.

Requires API keys (free tier available for IEX data + paper trading).

REST endpoints:
    GET /v2/stocks/{symbol}/bars/latest        (current price + in-progress OHLCV)
    GET /v2/stocks/{symbol}/bars?timeframe=1Min&start=...&end=...
    GET /v2/stocks/{symbol}/trades?start=...&end=...   (raw trade tape; gap recovery)
    GET /v2/stocks/{symbol}/quotes/latest

WebSocket:
    wss://stream.data.alpaca.markets/v2/iex  (trades + quotes + bars)

Hot-path bars are synthesised in-process from the ``trades`` WS channel
rather than the ``bars`` channel. The free IEX ``bars`` channel emits
one closed-minute bar per symbol per minute, which gives a 60-second
hot-path freshness floor — useless for sub-second strategies. The
``trades`` channel pushes per-trade with ~67 ms one-way latency, and
locally-aggregated OHLCV matches Alpaca's REST ``/bars`` output
bit-for-bit when the standard Intermarket-Sweep-Order (`'I'`)
exclusion is applied to OHLC (volume sums all trades unconditionally).
See ``data/run_output/realtime_diagnostics/alpaca_trades_rest_reconciliation.md``
for the empirical validation (256/256 bars × 5 OHLCV fields, 4 symbols
× 2 runs, 0 divergent).

Two gap-recovery mechanisms layered on top:

* **Watchdog** — Alpaca trade messages have no contiguous sequence ID
  (``i`` is a per-exchange-trade ID, not stream-position), so gap
  detection uses a silence watchdog: when no trade arrives for
  ``WATCHDOG_TIMEOUT_S`` on the WS, fire REST ``/v2/stocks/{sym}/trades``
  from the last-seen trade time to now and replay through the builder.
* **Cross-check** — on every local bar close, asynchronously fetch the
  canonical REST 1-minute bar for that minute and compare. On
  divergence, the REST bar overwrites the local one in the
  :class:`MarketDataBuffer` (dedup-by-timestamp makes this a simple
  callback re-fire). Catches untested edge cases like auction prints
  with rare condition codes (`'O'`) or trading-halt prints, without
  slowing the WS hot path.
"""

import asyncio
import json as _json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests

from open_fin_gym.realtime.data_providers.base import (
    DataProvider,
    MarketSnapshot,
    OrderBookSnapshot,
    interval_to_seconds,
    to_alpaca_timeframe,
)

logger = logging.getLogger(__name__)


# Trade conditions Alpaca's REST /bars endpoint excludes from OHLC
# (open/high/low/close) on the free IEX feed. Volume sums ALL trades
# regardless of condition; this set applies only to OHLC.
#
# Empirically derived (see ``data/run_output/realtime_diagnostics/
# alpaca_trades_rest_reconciliation.md``): the precise filter is just
# 'I' (Intermarket Sweep Order). Broader defensive codes (O/M/B/L/X/...)
# never appeared in the 1-hour validation window across SPY/AAPL/QQQ/IWM,
# so they're neither confirmed nor refuted. We add them defensively
# below — they're a no-op on the validated symbols and tier, but if
# an auction-print / halt-print condition does appear, the cross-check
# fallback below will surface a divergence and we can refine.
_DEFAULT_NON_LAST_SALE_CODES: frozenset[str] = frozenset({
    "I",  # Intermarket Sweep Order — empirically validated
    # Defensive (not seen in 1-hour validation; cross-check would catch):
    "O", "M", "B", "L", "X", "P", "T", "U", "Z", "9", "V", "4", "5", "6", "7",
})


class _AlpacaTradesBarBuilder:
    """Synthesise OHLCV bars at ``interval_s`` resolution from Alpaca's
    per-trade WS stream, with two safety nets layered on top:

    * **Condition-aware OHLC fold** — every trade contributes to bar
      volume; only last-sale-eligible trades (those whose ``c`` array
      contains no code in ``last_sale_filter``) contribute to
      open/high/low/close. Matches Alpaca's REST ``/bars`` aggregator
      exactly under filter ``{'I'}`` (validated 256/256 bars across 4
      symbols × 2 runs).

    * **Watchdog gap detection** — Alpaca trades don't carry a
      contiguous sequence ID, so we can't do Binance-style ID
      arithmetic. Instead we record the last received-wall-clock per
      symbol; an external watcher (see :meth:`subscribe_bars`) calls
      :meth:`backfill_silence_gap` if no trade has arrived for
      ``WATCHDOG_TIMEOUT_S`` to replay missed trades via REST.

    * **REST cross-check on bar close** — whenever a trade crosses
      into a new bar, the just-closed bar's locally-computed OHLCV is
      asynchronously compared against Alpaca's REST ``/bars`` value
      for that minute. On divergence the REST bar is re-fired through
      ``callback``; the buffer dedups by timestamp, so the REST value
      replaces the local one. Catches untested edge cases (auction
      prints, halt prints) without slowing the WS hot path. Runs on a
      private :class:`ThreadPoolExecutor` so REST blocking doesn't
      stall WS message processing.
    """

    #: Seconds of WS silence (no trade message for any subscribed
    #: symbol) before the watchdog fires REST backfill. Conservative;
    #: liquid US-equity names typically trade many times per second
    #: during market hours.
    WATCHDOG_TIMEOUT_S: float = 30.0

    #: Seconds to wait after a local bar closes before firing the REST
    #: cross-check. Alpaca's bars aggregator needs a few seconds to
    #: finalise the just-closed bar server-side.
    CROSS_CHECK_DELAY_S: float = 6.0

    def __init__(
        self,
        *,
        symbol: str,
        interval_s: int,
        callback: Callable[[MarketSnapshot], None],
        rest: "AlpacaProvider",
        last_sale_filter: frozenset[str] | None = None,
        cross_check_enabled: bool = True,
        cross_check_pool: ThreadPoolExecutor | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._symbol = symbol
        self._interval_s = interval_s
        self._interval_ms = interval_s * 1000
        self._callback = callback
        self._rest = rest
        self._last_sale_filter = (
            last_sale_filter
            if last_sale_filter is not None
            else _DEFAULT_NON_LAST_SALE_CODES
        )
        self._cross_check_enabled = cross_check_enabled
        # Allow the caller to share one pool across symbols; otherwise
        # create a tiny dedicated one. Cleaned up by ``close``.
        if cross_check_pool is not None:
            self._cross_check_pool = cross_check_pool
            self._owns_pool = False
        else:
            self._cross_check_pool = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix=f"alp-cc-{symbol}",
            )
            self._owns_pool = True
        # In-progress bar state. ``_ohlc_started`` separates "we've
        # opened a bar bucket" from "we've seen a last-sale-eligible
        # trade for OHLC" — until the latter, the bar has only volume.
        self._bar_open_ms: int | None = None
        self._o = 0.0
        self._h = 0.0
        self._l = 0.0
        self._c = 0.0
        self._v = 0.0
        self._ohlc_started = False
        self._ohlc_trade_count = 0
        # Watchdog state — last trade-time we observed (NOT
        # local-recv-time): used to span the REST backfill window
        # after a silence.
        self._last_trade_ms: int | None = None
        # Used by the subscribe_bars loop to know when the WS has
        # really gone idle vs when it's mid-flight.
        self._last_msg_perf: float = time.perf_counter()

    # --- core ----------------------------------------------------------

    def _bucket(self, trade_time_ms: int) -> int:
        return (trade_time_ms // self._interval_ms) * self._interval_ms

    def _is_eligible(self, conditions: tuple[str, ...]) -> bool:
        return not any(c in self._last_sale_filter for c in conditions)

    def _emit_current(self) -> None:
        """Emit a snapshot for the in-progress bar.

        Skips emission until at least one last-sale-eligible trade has
        landed in the bar — until then we have volume but no canonical
        OHLC value to surface.
        """
        if self._bar_open_ms is None or not self._ohlc_started:
            return
        snap = MarketSnapshot(
            symbol=self._symbol,
            timestamp=datetime.fromtimestamp(
                self._bar_open_ms / 1000, tz=timezone.utc
            ),
            price=self._c,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            volume=self._v,
        )
        self._callback(snap)

    def _close_current_bar(self) -> None:
        """Emit the final snapshot for the closed bar and asynchronously
        schedule the REST cross-check.

        **Hot-path contract — no blocking I/O.** This method is called
        synchronously from :meth:`on_trade` on the WS receive path; any
        latency introduced here directly delays the next trade's
        processing. So:

        * ``_emit_current()`` is a constant-time dict construction +
          buffer-lock acquisition (~10 µs on liquid pairs);
        * ``_cross_check_pool.submit()`` returns in ~5 µs once the task
          is enqueued; the actual REST work (a 6-second sleep followed
          by the ``/bars`` call) runs on a worker thread, releasing the
          GIL for the duration of both the ``time.sleep`` and the
          ``requests``-driven HTTP I/O. The WS thread continues
          consuming trades unimpeded.

        Total added cost on the hot path at every bar boundary:
        ~15 µs. Test ``test_cross_check_does_not_block_hot_path`` pins
        this with a deliberately slow REST mock.
        """
        if self._bar_open_ms is None or not self._ohlc_started:
            return
        closed_open_ms = self._bar_open_ms
        closed_snapshot = {
            "o": self._o, "h": self._h, "l": self._l,
            "c": self._c, "v": self._v,
            "ohlc_trade_count": self._ohlc_trade_count,
        }
        self._emit_current()
        if self._cross_check_enabled:
            # Fire-and-forget: cross-check runs entirely on the
            # threadpool worker; this submit() returns immediately.
            self._cross_check_pool.submit(
                self._cross_check, closed_open_ms, closed_snapshot
            )

    def _apply_trade(
        self,
        trade_time_ms: int,
        price: float,
        size: float,
        conditions: tuple[str, ...],
        *,
        emit: bool,
    ) -> None:
        bucket = self._bucket(trade_time_ms)
        eligible = self._is_eligible(conditions)
        if self._bar_open_ms is None:
            self._bar_open_ms = bucket
            if eligible:
                self._o = self._h = self._l = self._c = price
                self._ohlc_started = True
                self._ohlc_trade_count = 1
            self._v = size
        elif bucket == self._bar_open_ms:
            self._v += size
            if eligible:
                if not self._ohlc_started:
                    self._o = self._h = self._l = self._c = price
                    self._ohlc_started = True
                else:
                    if price > self._h:
                        self._h = price
                    if price < self._l:
                        self._l = price
                    self._c = price
                self._ohlc_trade_count += 1
        else:
            # Bar boundary crossed — close the previous bar and start fresh.
            self._close_current_bar()
            self._bar_open_ms = bucket
            self._v = size
            if eligible:
                self._o = self._h = self._l = self._c = price
                self._ohlc_started = True
                self._ohlc_trade_count = 1
            else:
                # Reset OHLC state but don't surface a snapshot yet;
                # wait for the first eligible trade.
                self._o = self._h = self._l = self._c = 0.0
                self._ohlc_started = False
                self._ohlc_trade_count = 0
        self._last_trade_ms = trade_time_ms
        self._last_msg_perf = time.perf_counter()
        if emit:
            self._emit_current()

    # --- public entry points -------------------------------------------

    def on_trade(self, msg: dict[str, Any]) -> None:
        """Process one Alpaca WS trade message (``{"T": "t", ...}``)."""
        ts_str = msg.get("t")
        price_raw = msg.get("p")
        size_raw = msg.get("s")
        if ts_str is None or price_raw is None or size_raw is None:
            return
        try:
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        except Exception:
            return
        trade_time_ms = int(ts.timestamp() * 1000)
        conditions = tuple(msg.get("c") or [])
        self._apply_trade(
            trade_time_ms,
            float(price_raw),
            float(size_raw),
            conditions,
            emit=True,
        )

    def silence_ms(self) -> float:
        """Wall-clock ms since the last received trade message.

        Used by the supervising subscribe_bars loop to decide whether
        to fire :meth:`backfill_silence_gap`.
        """
        return (time.perf_counter() - self._last_msg_perf) * 1000.0

    def backfill_silence_gap(self) -> None:
        """REST-fetch trades since ``_last_trade_ms`` and replay them.

        Called by the supervisor when the WS has been silent for
        :attr:`WATCHDOG_TIMEOUT_S`. The REST fetch covers
        ``(_last_trade_ms, now)`` exclusive of the original (since the
        WS already delivered that one). Trades land in the builder via
        ``_apply_trade(emit=False)`` to avoid flooding the buffer
        during recovery; one summary emit happens at the end.
        """
        if self._last_trade_ms is None:
            return
        start = datetime.fromtimestamp(
            (self._last_trade_ms + 1) / 1000, tz=timezone.utc
        )
        now = datetime.now(timezone.utc)
        logger.warning(
            "alpaca trade-stream silence detected sym=%s — REST backfill "
            "from %s to %s",
            self._symbol, start.isoformat(), now.isoformat(),
        )
        try:
            trades = self._rest.get_trades(
                self._symbol, start=start, end=now,
            )
        except Exception:
            logger.warning(
                "alpaca silence-backfill REST call failed sym=%s",
                self._symbol, exc_info=True,
            )
            return
        applied = 0
        for trade in trades:
            ts_str = trade.get("t")
            price_raw = trade.get("p")
            size_raw = trade.get("s")
            if ts_str is None or price_raw is None or size_raw is None:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            except Exception:
                continue
            self._apply_trade(
                int(ts.timestamp() * 1000),
                float(price_raw),
                float(size_raw),
                tuple(trade.get("c") or []),
                emit=False,
            )
            applied += 1
        logger.info(
            "alpaca silence-backfill applied sym=%s trades=%d",
            self._symbol, applied,
        )
        self._emit_current()

    # --- cross-check ---------------------------------------------------

    def _cross_check(
        self, closed_open_ms: int, local: dict[str, float],
    ) -> None:
        """Compare just-closed local bar to Alpaca REST ``/bars`` and
        replace if divergent.

        **Threading.** Runs on the cross-check :class:`ThreadPoolExecutor`,
        **not** the WS event loop. The ``time.sleep`` releases the GIL;
        the ``requests``-driven HTTP call inside ``get_bars`` also
        releases the GIL during socket I/O. So the WS thread continues
        receiving and dispatching trades the entire ~6.07 s this method
        is alive (≈6 s sleep + 1 RTT for REST).

        **Race-free overwrite.** By the time the REST result lands,
        the WS has moved on to bar *N+1*; any subsequent ``_emit``
        from the WS thread writes to bar *N+1*'s timestamp. The
        replacement here writes to bar *N*'s timestamp. Different
        timestamps → different buffer entries → no collision. The
        only shared mutation is the buffer lock, acquired briefly by
        the ``_callback`` re-fire.
        """
        try:
            time.sleep(self.CROSS_CHECK_DELAY_S)
            start = datetime.fromtimestamp(
                closed_open_ms / 1000, tz=timezone.utc
            )
            end = datetime.fromtimestamp(
                (closed_open_ms + self._interval_ms) / 1000, tz=timezone.utc
            )
            rest_bars = self._rest.get_bars(
                self._symbol, self._interval_label(), start, end,
            )
        except Exception:
            logger.debug(
                "alpaca cross-check REST failed sym=%s bar=%d",
                self._symbol, closed_open_ms, exc_info=True,
            )
            return
        # Find the bar whose timestamp matches.
        target = None
        for b in rest_bars:
            if int(b.timestamp.timestamp() * 1000) == closed_open_ms:
                target = b
                break
        if target is None:
            # REST hasn't produced the bar yet (race) or excluded it
            # entirely (zero last-sale-eligible trades). Either way,
            # nothing to compare against; leave the local bar in place.
            return
        rest = {
            "o": float(target.open if target.open is not None else target.price),
            "h": float(target.high if target.high is not None else target.price),
            "l": float(target.low if target.low is not None else target.price),
            "c": float(target.close if target.close is not None else target.price),
            "v": float(target.volume if target.volume is not None else 0.0),
        }
        if all(
            self._approx_equal(local[f], rest[f]) for f in ("o", "h", "l", "c", "v")
        ):
            return
        logger.warning(
            "alpaca cross-check divergence sym=%s bar_open_ms=%d "
            "local=%s rest=%s — replacing local with REST canonical",
            self._symbol, closed_open_ms, local, rest,
        )
        # Replace by re-firing callback with REST's snapshot. The
        # buffer dedups by timestamp, so this overwrites in place.
        snap = MarketSnapshot(
            symbol=self._symbol,
            timestamp=datetime.fromtimestamp(
                closed_open_ms / 1000, tz=timezone.utc
            ),
            price=rest["c"],
            open=rest["o"],
            high=rest["h"],
            low=rest["l"],
            close=rest["c"],
            volume=rest["v"],
        )
        try:
            self._callback(snap)
        except Exception:
            logger.warning(
                "alpaca cross-check replacement callback raised sym=%s",
                self._symbol, exc_info=True,
            )

    def _interval_label(self) -> str:
        # ``get_bars`` expects an interval string; we hold seconds.
        # Limit to common cases: 1m, 5m, 15m, 1h, 1d.
        s = self._interval_s
        if s % 86_400 == 0:
            return f"{s // 86_400}d"
        if s % 3600 == 0:
            return f"{s // 3600}h"
        if s % 60 == 0:
            return f"{s // 60}m"
        return f"{s}s"

    @staticmethod
    def _approx_equal(a: float, b: float, rel_tol: float = 1e-6) -> bool:
        if a == b:
            return True
        denom = max(abs(a), abs(b))
        if denom == 0.0:
            return True
        return abs(a - b) / denom < rel_tol

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        """Release the per-builder cross-check pool (if owned)."""
        if self._owns_pool:
            try:
                self._cross_check_pool.shutdown(wait=False)
            except Exception:
                pass


class AlpacaProvider:
    """Market data from the Alpaca Data API (v2) + WebSocket streams."""

    name = "alpaca"

    _WS_URL = "wss://stream.data.alpaca.markets/v2/iex"

    # Alpaca's free/basic data tier allows ~200 requests/min. We don't
    # pre-throttle on the client; instead we surface a warning when the
    # remaining-quota header drops low and retry once on 429.
    _REMAINING_WARN_THRESHOLD: int = 20

    def __init__(
        self,
        base_url: str = "https://data.alpaca.markets",
        api_key_env: str = "ALPACA_API_KEY",
        api_secret_env: str = "ALPACA_SECRET_KEY",
        rate_limit_per_min: int | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._session = requests.Session()
        self._api_key = os.environ.get(api_key_env, "")
        self._api_secret = os.environ.get(api_secret_env, "")
        if self._api_key:
            self._session.headers.update(
                {"APCA-API-KEY-ID": self._api_key, "APCA-API-SECRET-KEY": self._api_secret}
            )
        else:
            logger.warning(
                "Alpaca API key not set (%s).  Requests will likely fail.",
                api_key_env,
            )
        # rate_limit_per_min retained for back-compat; previously drove a
        # client-side sleep loop. Now informational only: stored so a caller
        # who explicitly passed a tighter envelope can still inspect it.
        self._declared_rate_limit_per_min = rate_limit_per_min

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(url, params=params, timeout=15)
        self._inspect_rate_limit_headers(resp)
        if resp.status_code == 429:
            retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
            logger.warning(
                "alpaca 429; sleeping %.2fs and retrying once (path=%s)",
                retry_after,
                path,
            )
            time.sleep(retry_after)
            resp = self._session.get(url, params=params, timeout=15)
            self._inspect_rate_limit_headers(resp)
        resp.raise_for_status()
        return resp.json()

    def _inspect_rate_limit_headers(self, resp: requests.Response) -> None:
        # Alpaca exposes X-RateLimit-Remaining (integer count) and
        # X-RateLimit-Limit. When remaining drops below the warn
        # threshold we log so the caller knows the next few calls might
        # hit 429.
        remaining_raw = resp.headers.get("X-RateLimit-Remaining")
        if remaining_raw is None:
            return
        try:
            remaining = int(remaining_raw)
        except ValueError:
            return
        if remaining < self._REMAINING_WARN_THRESHOLD:
            limit_raw = resp.headers.get("X-RateLimit-Limit", "?")
            logger.warning(
                "alpaca rate-limit remaining=%d (of %s); consider backing off",
                remaining,
                limit_raw,
            )

    @staticmethod
    def _parse_retry_after(raw: str | None) -> float:
        if raw is None:
            return 1.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 1.0

    # ── DataProvider interface ────────────────────────────────────────

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        # Use the latest 1-minute bar rather than `trades/latest` so the
        # snapshot carries OHLCV + volume. The bar's open-time timestamp
        # (minute boundary) lets MarketDataBuffer's dedup-by-timestamp
        # overwrite the same entry on repeated calls within the minute —
        # keeps `recent_bars[-1]` coherent and stops microsecond-precision
        # wall-clock injections from accumulating as separate volume=None
        # entries. `feed=iex` matches `get_bars` for free-tier consistency.
        data = self._get(
            f"/v2/stocks/{symbol}/bars/latest", {"feed": "iex"}
        )
        bar = data.get("bar")
        if not bar:
            raise ValueError(f"No latest bar returned for {symbol}")
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
            price=float(bar["c"]),  # close == latest tick price for in-progress bar
            open=float(bar["o"]),
            high=float(bar["h"]),
            low=float(bar["l"]),
            close=float(bar["c"]),
            volume=float(bar["v"]),
        )

    def get_price_at(
        self, symbol: str, at: datetime, interval: str = "1m"
    ) -> MarketSnapshot:
        bars = self.get_bars(symbol, interval, at, at)
        if not bars:
            raise ValueError(
                f"No {interval} bar data for {symbol} at {at.isoformat()}"
            )
        return bars[0]

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketSnapshot]:
        tf = to_alpaca_timeframe(interval)
        all_bars: list[MarketSnapshot] = []
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "timeframe": tf,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 1000,
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token

            data = self._get(f"/v2/stocks/{symbol}/bars", params)
            bars_raw = data.get("bars") or []
            for b in bars_raw:
                all_bars.append(
                    MarketSnapshot(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                        price=float(b["c"]),
                        open=float(b["o"]),
                        high=float(b["h"]),
                        low=float(b["l"]),
                        close=float(b["c"]),
                        volume=float(b["v"]),
                    )
                )
            page_token = data.get("next_page_token")
            if not page_token:
                break

        return all_bars

    # ── Raw trades (REST) ────────────────────────────────────────────

    def get_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime | None = None,
        limit: int = 10_000,
        feed: str = "iex",
    ) -> list[dict[str, Any]]:
        """Fetch raw trades via ``GET /v2/stocks/{symbol}/trades``.

        Used by :meth:`subscribe_bars` to backfill the tape across a
        WS silence gap (watchdog timeout) — same trade dicts the WS
        delivers, so the bar builder can replay them through the
        normal :meth:`on_trade` path without a separate code shape.

        Pagination: Alpaca caps each response at 10 000 trades. If
        the result is paginated (``next_page_token`` non-null), the
        loop fetches additional pages until exhausted.

        ``feed='iex'`` matches the WS subscription and the
        :meth:`get_bars` aggregator's coverage — REST and WS see the
        same trade tape under this feed.
        """
        all_trades: list[dict[str, Any]] = []
        page_token: str | None = None
        params_base: dict[str, Any] = {
            "start": start.isoformat(),
            "limit": min(int(limit), 10_000),
            "feed": feed,
        }
        if end is not None:
            params_base["end"] = end.isoformat()
        while True:
            params = dict(params_base)
            if page_token:
                params["page_token"] = page_token
            data = self._get(f"/v2/stocks/{symbol}/trades", params)
            trades = data.get("trades") or []
            all_trades.extend(trades)
            page_token = data.get("next_page_token")
            if not page_token:
                break
        return all_trades

    # ── Order book / NBBO (REST) ─────────────────────────────────────

    def get_order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        """Fetch NBBO (best bid/ask) via ``GET /v2/stocks/{symbol}/quotes/latest``.

        Alpaca does not expose full L2 depth -- only the National Best
        Bid and Offer.  ``bids`` and ``asks`` lists remain empty.
        """
        data = self._get(f"/v2/stocks/{symbol}/quotes/latest")
        quote = data.get("quote", data)
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            best_bid=float(quote.get("bp", 0)),
            best_bid_qty=float(quote.get("bs", 0)),
            best_ask=float(quote.get("ap", 0)),
            best_ask_qty=float(quote.get("as", 0)),
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
        """Stream bars synthesised from Alpaca WS ``trades`` ticks.

        Why ``trades`` rather than ``bars``: the free IEX ``bars``
        channel emits **one closed-minute bar per symbol per minute, at
        minute close** — a 60-second hot-path freshness ceiling on the
        same tier that exposes a per-trade ``trades`` channel with ~67
        ms one-way latency. Reading those trades into a local
        condition-aware OHLCV synthesiser reproduces Alpaca's REST
        ``/bars`` output bit-for-bit at sub-second latency (see module
        docstring + validation harness).

        Three layered safeties:

        1. **Condition-aware OHLC fold**: trades with `'I'` (Intermarket
           Sweep Order) or other rare non-last-sale codes are excluded
           from OHL/C. All trades count toward volume.
        2. **Silence watchdog**: if no trade arrives for
           ``_AlpacaTradesBarBuilder.WATCHDOG_TIMEOUT_S`` we fire REST
           ``/v2/stocks/{sym}/trades`` to replay the missed window. The
           framework's :class:`MarketDataBuffer` also has its own
           reconnect supervisor; the two are complementary (reconnect
           handles connection death, this watchdog handles a stalled
           but still-open WS).
        3. **REST cross-check on bar close**: every locally-closed bar
           is async-compared against ``/v2/stocks/{sym}/bars``; on
           divergence the REST value is re-fired through ``callback``
           and the buffer's dedup-by-timestamp swaps the local bar for
           the canonical one. Catches untested edge cases (auction
           prints, halt prints) without slowing the WS hot path.
        """
        import websockets  # optional dependency

        builder = _AlpacaTradesBarBuilder(
            symbol=symbol,
            interval_s=interval_to_seconds(interval),
            callback=callback,
            rest=self,
        )
        try:
            async with websockets.connect(self._WS_URL) as ws:
                await ws.send(_json.dumps({
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._api_secret,
                }))
                await ws.recv()  # auth ack
                await ws.send(_json.dumps({
                    "action": "subscribe",
                    "trades": [symbol],
                }))
                await ws.recv()  # subscription ack

                # Watchdog loop: poll for messages with a timeout
                # equal to the silence threshold. On TimeoutError, run
                # the REST silence-backfill and resume.
                watchdog_s = _AlpacaTradesBarBuilder.WATCHDOG_TIMEOUT_S
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=watchdog_s
                        )
                    except asyncio.TimeoutError:
                        builder.backfill_silence_gap()
                        continue
                    msgs = _json.loads(raw)
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        if msg.get("T") != "t":
                            # Non-trade frames (auth/sub acks, error,
                            # status, etc.); ignored on the bar path.
                            continue
                        try:
                            builder.on_trade(msg)
                        except Exception:
                            logger.warning(
                                "alpaca trade processing error sym=%s",
                                symbol, exc_info=True,
                            )
        finally:
            builder.close()

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable[[OrderBookSnapshot], None],
    ) -> None:
        """Stream NBBO quotes via Alpaca WebSocket (IEX feed)."""
        import websockets  # optional dependency

        async with websockets.connect(self._WS_URL) as ws:
            auth_msg = _json.dumps({"action": "auth", "key": self._api_key, "secret": self._api_secret})
            await ws.send(auth_msg)
            await ws.recv()

            sub_msg = _json.dumps({"action": "subscribe", "quotes": [symbol]})
            await ws.send(sub_msg)
            await ws.recv()

            async for raw in ws:
                msgs = _json.loads(raw)
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    if msg.get("T") != "q":  # "q" = quote message
                        continue
                    snap = OrderBookSnapshot(
                        symbol=msg.get("S", symbol),
                        timestamp=datetime.fromisoformat(
                            msg["t"].replace("Z", "+00:00")
                        ),
                        best_bid=float(msg.get("bp", 0)),
                        best_bid_qty=float(msg.get("bs", 0)),
                        best_ask=float(msg.get("ap", 0)),
                        best_ask_qty=float(msg.get("as", 0)),
                    )
                    callback(snap)
