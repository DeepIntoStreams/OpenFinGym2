"""RealtimeTradingTask -- real-time paper trading with immediate rewards.

The live-feed implementation of the :class:`TradingTask` data-source seam:
the shared base owns the per-step engine (tick → dispatch → reward), the
observation skeleton, and evaluation; this subclass only supplies the live
data source — a :class:`DataProvider` + :class:`MarketDataBuffer`
(REST + WebSocket) — and an execution engine (:class:`SimulatedExecutor`
or :class:`AlpacaPaperExecutor`).

Action surface — single order, or ``{"orders": [...]}`` for a batch::

    {
      "action": "buy" | "sell" | "hold" | "cancel",
      "symbol": str,
      "quantity": float,
      "order_type": "market" | "limit" | "stop" | "stop_limit",  # default "market"
      "limit_price": float | None,
      "stop_price": float | None,
      "tif": "ioc" | "gtc",     # default "gtc"; ignored for market/hold
      "order_id": str,           # required only for action="cancel"
    }

The per-step flow (``executor.tick`` → dispatch orders → reward =
per-symbol ``compute_pnl`` delta, with episode-end ``expire_all``) lives in
:meth:`TradingTask._execute_trade`. Realtime feeds the executor *scalar*
ticks (not OHLC bars — the in-progress bar's high/low aren't known yet) and
refreshes the buffer once at the top of each step via
:meth:`_on_step_start`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from open_fin_gym.realtime.config import TradingConfig
from open_fin_gym.realtime.contracts import TaskMetadata, TradingTask
from open_fin_gym.realtime.data_providers.base import (
    DataProvider,
    downsample_bars,
)
from open_fin_gym.realtime.execution import (
    BaseExecutor,
    OrderType,
    create_executor,
)
from open_fin_gym.realtime.market_data_buffer import MarketDataBuffer
from open_fin_gym.realtime.rewards import (
    MaxDrawdown,
    PnL,
    SharpeRatio,
    TradingReward,
    WinRate,
)
from open_fin_gym.realtime.tasks.base_realtime_task import (
    _resolve_context_resolutions,
    _validate_target_symbols,
)

logger = logging.getLogger(__name__)


class RealtimeTradingTask(TradingTask):
    """Realtime paper-trading task with immediate PnL-based rewards.

    Parameters:

    - ``provider``: data backend (Binance, Alpaca, ...).
    - ``symbols``: symbols to trade.
    - ``trading_config``: slippage, transaction costs, execution mode.
    - ``initial_capital``: starting cash for SimulatedExecutor
      (default 100000.0). Ignored for ``execution_mode="alpaca_paper"``
      — the real paper account's cash is the source of truth.
    - ``context_resolutions``: non-empty list of
      ``{"interval": str, "bars": positive int}`` entries. The
      ``data_resolution`` entry drives the *only*
      :class:`MarketDataBuffer` (latest-price lookups, PnL, WebSocket
      sub, dataset shipping). Sidecars are **downsampled from this
      single buffer** at observation time — no per-sidecar buffer or
      extra subs. Buffer depth auto-scales to the deepest sidecar.
    - ``data_resolution``: which entry's interval drives the primary
      buffer + WebSocket sub + execution price + dataset shipping.
      Required when 2+ entries are given. **Must be the finest
      interval** (sidecars are downsampled from it; no upsampling).
      This controls data buffering only — agents call ``step()`` at
      whatever frequency they want; ``data_resolution`` does not pace
      the gym loop.
    - ``buffer_size``: per-symbol buffer capacity.
    - ``max_steps``: max steps per episode (0 = unlimited).
    - ``target_symbols``: subset whose trades feed eval (default =
      all). Non-target trades are allowed at :meth:`step` (for
      hedging on input-only context symbols) but
      :meth:`evaluate` filters ``_trade_history`` to target trades.
    """

    #: See :attr:`TradingTask.DEFAULT_TRADING_REWARDS`. Same default set;
    #: reward-bank dispatch happens once per ``evaluate()`` call, not in
    #: :meth:`step` — so per-tick latency is untouched.
    DEFAULT_TRADING_REWARDS: tuple[type[TradingReward], ...] = (
        PnL,
        SharpeRatio,
        MaxDrawdown,
        WinRate,
    )

    #: Staleness budgets (ms) for the WS-fed buffer before the per-step
    #: REST fallback fires. With Binance @aggTrade pushing per-trade
    #: (sub-100ms cadence on liquid pairs) and @depth20@100ms pushing
    #: order book every 100ms, these TTLs are 5-10× the expected push
    #: interval — generous enough that healthy WS never trips them, yet
    #: tight enough that silent WS death surfaces within one TTL window.
    #: REST fallback then closes the gap until WS reconnect succeeds.
    PRICE_STALENESS_TTL_MS: float = 500.0
    OB_STALENESS_TTL_MS: float = 1000.0

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: DataProvider,
        symbols: list[str],
        trading_config: TradingConfig | None = None,
        context_resolutions: list[dict[str, Any]] | None = None,
        data_resolution: str | None = None,
        buffer_size: int = 500,
        max_steps: int = 0,
        target_symbols: list[str] | None = None,
        initial_capital: float = 100000.0,
    ) -> None:
        super().__init__(config)
        self._provider = provider
        self._symbols = list(symbols)
        if not self._symbols:
            raise ValueError("symbols must be non-empty")
        self._trading_config = trading_config or TradingConfig()
        self._max_steps = max_steps
        self._initial_capital = float(initial_capital)
        # Sidecar resolutions downsample from the single primary buffer
        # at observation time — no extra buffers, no extra WS subs.
        # _lookback_bars auto-scales to the deepest sidecar window.
        if context_resolutions is None:
            context_resolutions = [{"interval": "1m", "bars": 60}]
        (
            data_resolution_interval,
            data_resolution_bars,
            extras,
            primary_lookback,
        ) = _resolve_context_resolutions(context_resolutions, data_resolution)
        self._interval = data_resolution_interval
        self._context_bars = data_resolution_bars
        self._lookback_bars = primary_lookback
        self._extra_resolutions: tuple[tuple[str, int], ...] = extras
        # Trades outside target_symbols are allowed (agent may hedge on
        # context-only symbols) but excluded from reward-bank metrics.
        self._target_symbols = _validate_target_symbols(target_symbols, self._symbols)

        # Auto-scale buffer_size up if it can't span the sidecars'
        # lookback — silent truncation would corrupt downsampling.
        required_buffer = primary_lookback + max(data_resolution_bars, 100)
        effective_buffer_size = max(buffer_size, required_buffer)

        self._buffer = MarketDataBuffer(
            provider=provider,
            symbols=self._symbols,
            interval=data_resolution_interval,
            buffer_size=effective_buffer_size,
            backfill_bars=self._lookback_bars,
        )
        self._executor: BaseExecutor = create_executor(
            self._trading_config,
            provider.name,
            initial_capital=self._initial_capital,
        )

        self._step_count: int = 0
        self._done: bool = False
        self._trade_history: List[Dict[str, Any]] = []
        # Persistent pool for per-symbol REST fan-out. Sized 2× symbols
        # so price + order-book fetches dispatch concurrently. Persistent
        # avoids the per-step pool-creation cost (~50-200 µs); daemon
        # workers don't block process exit.
        self._fetch_pool = ThreadPoolExecutor(
            max_workers=max(4, len(self._symbols) * 2),
            thread_name_prefix="rtt-fetch",
        )

    # ------------------------------------------------------------------
    # BaseTask abstract implementations
    # ------------------------------------------------------------------

    def metadata(self) -> TaskMetadata:
        sym_label = "_".join(self._symbols[:3])
        return TaskMetadata(
            task_id=f"realtime_trading_{sym_label}",
            title=f"Realtime Trading ({', '.join(self._symbols)})",
            description=(
                f"Realtime paper trading for {self._symbols} via {self._provider.name} "
                f"with {self._trading_config.execution_mode} execution."
            ),
            task_type="realtime",
            interaction_model="trading",
        )

    def load_data(self) -> Any:
        """Backfill the (single) MarketDataBuffer via REST and start
        streaming. Sidecar resolutions are derived by downsampling
        this buffer at observation time, so no per-sidecar fetch or
        WebSocket sub is needed.

        After kicking off the WS streamer, this also waits up to 5 s
        for the WS to push its first message per symbol. Hitting the
        timeout is not fatal — backfill is already in place and the
        observation hot path tolerates a cold WS via REST fallback —
        but the wait pays the warmup cost once at startup so the first
        few `step()` calls don't each take the latency hit individually.
        """
        if self._data is not None:
            return self._data
        self._buffer.backfill()
        stream_thread = self._buffer.start_streaming_background()
        # Only wait for warmup if a streaming thread was launched —
        # mock / WS-less providers return None and would just burn the
        # full timeout per test.
        if stream_thread is not None:
            warm = self._buffer.wait_for_warmup(timeout=5.0)
            if not warm:
                logger.warning(
                    "WS warmup timeout (5s) for some symbols; hot path "
                    "will fall back to REST until WS catches up. "
                    "provider=%s symbols=%s",
                    self._provider.name,
                    self._symbols,
                )
        self._data = True
        return self._data

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "step": "int",
                "steps_remaining": "int (-1 when max_steps is unbounded)",
                "symbols": {
                    "per_symbol": {
                        "symbol": "str",
                        "price": "float",
                        "volume": (
                            "float | None — in-progress bar's running cumulative "
                            "volume (monotonically increasing within the minute "
                            "for Binance WS klines, reset at minute close). None "
                            "when the provider hasn't supplied volume yet."
                        ),
                        "timestamp": "str (ISO-8601; bar-boundary)",
                        "recent_bars": f"list[MarketSnapshot] (last {self._context_bars})",
                        "order_book": "OrderBookSnapshot | None",
                    },
                },
                "portfolio": (
                    "dict[cash (float | None if executor doesn't track cash "
                    "locally), reserved_cash, positions(dict[str,float]), "
                    "pnl(dict[str,float]), value(float | None), "
                    "pending_orders(list[PendingOrder])]"
                ),
            },
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": {
                "action": "'buy' | 'sell' | 'hold' | 'cancel' (required)",
                "symbol": "str (required for buy/sell)",
                "quantity": "float (> 0 for buy/sell, ignored for hold/cancel)",
                "order_type": (
                    "'market' | 'limit' | 'stop' | 'stop_limit' (default 'market')"
                ),
                "limit_price": "float (required for limit / stop_limit)",
                "stop_price": "float (required for stop / stop_limit)",
                "tif": "'ioc' | 'gtc' (default 'gtc'; ignored for market/hold)",
                "order_id": "str (required only for action='cancel')",
            },
            "batch": (
                "{'orders': [<single-order dict>, ...]} for multiple orders / "
                "cancels in one step"
            ),
            "order_types": list(OrderType.ALL),
            "time_in_force": ["ioc", "gtc"],
        }

    def reset(self) -> Any:
        """Reset executor and buffer; return initial observation."""
        self._executor.reset()
        self._step_count = 0
        self._done = False
        self._trade_history = []
        self._prev_pnl_per_sym = {}
        if self._data is None:
            self.load_data()
        # WS stream is just starting up; one-shot REST poll seeds the
        # initial observation. Subsequent obs read whatever is freshest
        # in the buffer (WS push or per-step REST refresh).
        self._refresh_prices_from_provider()
        return self._get_market_observation()

    def _on_step_start(self) -> None:
        # ONE provider fetch per step (was duplicated in _execute_trade
        # and _get_market_observation). Ensures the reward in step N
        # matches the prices in obs(N+1)'s recent_bars[-1] exactly,
        # instead of drifting by one RTT. The shared base ``step()``
        # (TradingTask) owns the trade-log + episode-end-expire loop;
        # this hook supplies the only realtime-specific pre-step work.
        self._refresh_prices_from_provider()

    # ------------------------------------------------------------------
    # Data-source hooks (the live-feed seam; engine/reward/eval are in the
    # shared TradingTask base).
    # ------------------------------------------------------------------

    def _market_observation_block(self) -> Dict[str, Dict[str, Any]]:
        # NOTE: provider price fetch happens once at the top of `step()`
        # (and once in `reset()`), not here. The buffer is the single
        # source of truth for current prices on the read path.
        sym_obs: Dict[str, Dict[str, Any]] = {}
        for symbol in self._symbols:
            latest = self._buffer.get_latest_price(symbol)
            recent = self._buffer.get_recent_bars(symbol, self._context_bars)
            order_book = self._buffer.get_order_book(symbol)

            entry: Dict[str, Any] = {
                "symbol": symbol,
                "price": latest.price if latest else None,
                "volume": latest.volume if latest else None,
                "timestamp": latest.timestamp.isoformat() if latest else None,
                "recent_bars": recent,
                "order_book": order_book,
            }
            if self._extra_resolutions:
                # Multi-resolution view: sidecars downsample from the
                # primary buffer's _lookback_bars history, so they stay
                # in sync with every WS update — no extra fetch.
                by_interval: dict[str, list[Any]] = {self._interval: recent}
                full_history = self._buffer.get_recent_bars(symbol, self._lookback_bars)
                for ex_interval, ex_bars in self._extra_resolutions:
                    by_interval[ex_interval] = downsample_bars(
                        full_history, ex_interval, ex_bars
                    )
                entry["recent_bars_by_interval"] = by_interval
            sym_obs[symbol] = entry
        return sym_obs

    def _execution_quotes(self) -> Dict[str, float]:
        # Realtime feeds the executor SCALAR ticks — the in-progress bar's
        # high/low aren't known yet, so no OHLC bounds (and AlpacaPaper
        # never receives a bar dict). _quote_bounds collapses scalar→(p,p,p).
        return self._current_prices()

    def _market_timestamp(self) -> Any:
        return datetime.now(timezone.utc)

    def _episode_done(self) -> bool:
        return self._max_steps > 0 and self._step_count >= self._max_steps - 1

    def _steps_remaining(self) -> int:
        # -1 sentinel when the episode is unbounded (max_steps == 0).
        return self._max_steps - self._step_count if self._max_steps > 0 else -1

    def _current_prices(self) -> dict[str, float]:
        """Collect latest prices for all symbols."""
        prices: dict[str, float] = {}
        for s in self._symbols:
            lp = self._buffer.get_latest_price(s)
            if lp is not None:
                prices[s] = lp.price
        return prices

    def _refresh_prices_from_provider(self) -> None:
        """TTL-gated REST fallback for price + order book.

        In steady state, the WS-fed buffer carries fresh data
        (BinanceProvider's ``@aggTrade`` pushes per trade,
        ``@depth20@100ms`` pushes the OB every 100ms). REST is only
        useful when the buffer hasn't received a recent push — i.e.,
        cold start, WS reconnecting, or the rare server-side stall.
        So we read the buffer's staleness clock first and only dispatch
        REST when ``staleness_ms > TTL``.

        Per-symbol fan-out:
        - ``get_current_price`` — dispatched when
          ``buffer.price_staleness_ms(sym) > PRICE_STALENESS_TTL_MS``.
        - ``get_order_book`` — dispatched when
          ``buffer.ob_staleness_ms(sym) > OB_STALENESS_TTL_MS``
          *and* the provider exposes ``get_order_book``.

        REST results are applied through the same callbacks the WS
        uses (``insert_bar`` / ``_on_order_book``) so the staleness
        clocks tick forward identically regardless of source.

        Race guard: a WS push could land while a REST call is in
        flight. After receiving the REST response we re-check
        staleness; if it's already under the TTL again (meaning WS
        won the race), we skip the apply rather than overwriting the
        fresher WS data with the stale-by-now REST snapshot.

        Steady-state cost when WS healthy: zero REST calls, no thread-
        pool dispatch, ~microseconds per symbol for the staleness
        check. Worst case (WS dead): identical to the prior
        per-step-REST behavior, modulo one extra dict read per
        symbol.
        """
        if not self._symbols:
            return
        futures: list[tuple[str, str, Any]] = []
        has_ob_api = getattr(self._provider, "get_order_book", None) is not None
        for sym in self._symbols:
            if self._buffer.price_staleness_ms(sym) >= self.PRICE_STALENESS_TTL_MS:
                futures.append(
                    (
                        sym,
                        "price",
                        self._fetch_pool.submit(self._provider.get_current_price, sym),
                    )
                )
            if (
                has_ob_api
                and self._buffer.ob_staleness_ms(sym) >= self.OB_STALENESS_TTL_MS
            ):
                futures.append(
                    (
                        sym,
                        "ob",
                        self._fetch_pool.submit(self._provider.get_order_book, sym),
                    )
                )
        if not futures:
            return  # happy path: WS-fed buffer is fresh for every symbol
        for sym, kind, future in futures:
            try:
                result = future.result(timeout=15.0)
            except Exception:
                logger.debug(
                    "parallel %s fetch failed for %s", kind, sym, exc_info=True
                )
                continue
            if kind == "price":
                # Race guard: if WS landed a fresh bar between our
                # REST dispatch and result, skip the REST apply (WS
                # value is at least as fresh and authoritative).
                # Comparison is ``>=`` so TTL=0 (used by tests as the
                # "always-REST" sentinel) consistently applies the
                # REST result even when the staleness delta is below
                # the timer's resolution floor.
                if self._buffer.price_staleness_ms(sym) >= self.PRICE_STALENESS_TTL_MS:
                    self._buffer.insert_bar(sym, result)
            else:  # "ob"
                if self._buffer.ob_staleness_ms(sym) >= self.OB_STALENESS_TTL_MS:
                    self._buffer._on_order_book(result)

    def close(self) -> None:
        """Release the per-task thread pool. Safe to call multiple times.

        Optional: ThreadPoolExecutor workers are daemon threads so they
        won't block process exit, but a clean shutdown frees the
        sockets in the requests.Session connection pool sooner. The
        agent_runtime's gym loop doesn't currently call this; runners
        embedded in long-lived processes should.
        """
        pool = getattr(self, "_fetch_pool", None)
        if pool is None:
            return
        pool.shutdown(wait=False)
        self._fetch_pool = None  # type: ignore[assignment]
