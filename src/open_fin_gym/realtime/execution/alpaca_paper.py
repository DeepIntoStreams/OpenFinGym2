"""Alpaca paper-trading executor — submits real orders to paper-api.alpaca.markets.

Maintains a **local model** of broker state (positions, cost basis,
cash) on the hot path, kept consistent with Alpaca by two parallel
mechanisms:

1. **WebSocket ``trade_updates`` stream** — real-time push of fill /
   partial_fill / cancel / expire events. Primary source of state
   mutations.
2. **Background REST refresh** of ``/v2/positions`` + ``/v2/account``
   every 5s — authoritative snapshot, corrects drift if the WS missed
   an event.

Read-path methods (``compute_pnl``, ``get_positions``, ``get_cash``,
``get_pending_orders``) serve from the local model — zero network
calls per ``step()``. Market-order ``submit()`` returns synchronously
via a *provisional* booking at ``market_price``; the WS push
(typically <100 ms later) replaces it with ``filled_avg_price`` and
adjusts ``_cost_basis`` by the delta. ``tick()`` is a no-op (fills
arrive via WS, not polling) — kept for interface parity with
:class:`SimulatedExecutor`.

Lifecycle: ``__init__`` runs a 3-way parallel REST sanity check
(``/v2/account`` validates creds + seeds ``_cash``; positions + open
orders MUST be empty). A "dirty" account is either flattened
(``flatten_on_start=True``) or hard-fails (default). Dirty-start
refusal is deliberate — residual positions carry a stale cost basis
that would silently contaminate every PnL number. ``reset()``
suspends the refresher, cancels open orders, closes positions, clears
local state, and resumes. ``close()`` joins the two daemons.
"""

from __future__ import annotations

import asyncio
import atexit
import json as _json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import requests

from open_fin_gym.realtime.execution.base import (
    ActionVerb,
    BaseExecutor,
    CancelResult,
    ExecutionReport,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    Rejection,
    RejectionCode,
    SubmitResult,
    TickResult,
    TimeInForce,
)

logger = logging.getLogger(__name__)


# Alpaca's REST vocabulary mostly aligns with ours but isn't 1:1.
_ALPACA_TYPE = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP: "stop",
    OrderType.STOP_LIMIT: "stop_limit",
}
_ALPACA_TIF = {
    TimeInForce.IOC: "ioc",
    TimeInForce.GTC: "gtc",
}
# Forward maps have unique values, so reversal is unambiguous.
_REV_ALPACA_TYPE = {v: k for k, v in _ALPACA_TYPE.items()}
_REV_ALPACA_TIF = {v: k for k, v in _ALPACA_TIF.items()}

# Tolerance for fractional-share float representation when comparing
# local vs REST /v2/positions qty.
_QTY_EPSILON: float = 1e-6


class AlpacaPaperExecutor(BaseExecutor):
    """Submits real orders to Alpaca's paper-trading environment with
    a WS-primary + REST-reconciled local state model.

    Args:
        api_key_env: Environment variable holding the API key (default
            ``ALPACA_API_KEY``).
        api_secret_env: Environment variable holding the secret key.
        base_url: Alpaca paper-trading base URL.
        refresh_interval_sec: Background REST reconciliation cadence in
            seconds. Default 5.0.
        ws_url: Alpaca account-stream WebSocket URL.
        flatten_on_start: When ``True``, residual positions / open orders
            observed at construction time are flushed before the executor
            reports ready. When ``False`` (default), a dirty account causes
            ``__init__`` to raise :class:`RuntimeError` so the operator can
            decide whether to flatten manually or pass
            ``flatten_on_start=True`` to opt in to automatic cleanup.
            Per-session PnL accounting depends on the executor's local cost
            basis matching the broker's; adopting residual positions from
            prior sessions would silently contaminate it.

    Threading:
        Two background daemon threads are launched at ``__init__``:
        ``_refresher`` (REST every ``refresh_interval_sec``) and
        ``_ws_listener`` (WS trade_updates). All state mutations and reads
        are protected by ``_snapshot_lock``. Call ``close()`` to stop them
        cleanly.

    Concurrency precondition:
        This executor assumes exclusive ownership of the underlying Alpaca
        paper-trading account for its lifetime. ``flatten_on_start=True`` +
        ``reset()`` issue account-wide DELETEs that will clobber a
        concurrent session's live orders and positions. Bundles that share
        a paper account must serialise their trials externally — there is
        no in-process lock against this.
    """

    _DEFAULT_REFRESH_INTERVAL: float = 5.0

    # Poll budget for /v2/positions after DELETE during dirty-start
    # flatten. Alpaca closes in-RTH positions within ~200ms; outside RTH
    # the orders queue server-side and won't clear here — we log and
    # proceed.
    _FLATTEN_POLL_TIMEOUT_SEC: float = 2.0

    # Debounce window for the post-submit "validate sooner" nudge to the
    # background reconciler. A burst of submits within this window
    # coalesces to a single early reconcile — the WS is the primary fill
    # source, so the 5s periodic reconcile plus an occasional nudge is
    # ample. Caps extra REST volume against Alpaca's 200/min trading quota
    # on fast-trading episodes.
    _NUDGE_MIN_INTERVAL_S: float = 0.25

    # Cash from /v2/account is synced only after this much quiet (no local
    # fill touching cash). Shields a just-filled cash delta from being
    # erased by the account aggregate's propagation lag; long enough to
    # cover that lag, short enough that fill-less drift (fees/dividends/
    # interest) still gets picked up between trades.
    _CASH_SYNC_IDLE_S: float = 3.0

    def __init__(
        self,
        api_key_env: str = "ALPACA_API_KEY",
        api_secret_env: str = "ALPACA_SECRET_KEY",
        base_url: str = "https://paper-api.alpaca.markets",
        refresh_interval_sec: float = _DEFAULT_REFRESH_INTERVAL,
        ws_url: str = "wss://paper-api.alpaca.markets/stream",
        *,
        flatten_on_start: bool = False,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._ws_url = ws_url
        self._refresh_interval_sec = float(refresh_interval_sec)
        self._session = requests.Session()
        api_key = os.environ.get(api_key_env, "")
        api_secret = os.environ.get(api_secret_env, "")
        if not api_key:
            raise ValueError(
                f"Alpaca paper trading requires API key ({api_key_env} env var)"
            )
        self._api_key = api_key
        self._api_secret = api_secret
        self._session.headers.update(
            {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
        )

        # --------- Local model state (all guarded by _snapshot_lock) ---------
        self._positions: dict[str, float] = {}
        # PnL = current_price * qty - cost_basis - tx_total.
        self._cost_basis: dict[str, float] = {}
        self._tx_cost_total: dict[str, float] = {}
        self._cash: float = 0.0
        self._trade_log: list[ExecutionReport] = []
        self._local_pending: dict[str, PendingOrder] = {}
        # Idempotency guard against double-counting fills via reconnect
        # REST gap-fill or duplicate WS pushes.
        self._reported_order_ids: set[str] = set()
        self._last_rest_snapshot: dict[str, Any] = {
            "positions": None,
            "account": None,
            "ts": None,
        }
        # Cursor for /v2/orders?after=<ts> gap-fill on WS reconnect.
        self._last_trade_event_ts: str | None = None

        # --------- Synchronization primitives ----------
        self._snapshot_lock = threading.Lock()
        self._refresh_event = threading.Event()  # signal "refresh now"
        self._refresh_pause = threading.Event()  # set by reset() to suspend
        self._refresh_stop = threading.Event()   # set by close() to terminate
        self._last_nudge_ts: float = 0.0         # debounce for _nudge_refresh
        # Monotonic time of the last LOCAL cash mutation (a fill). Gates the
        # REST cash sync in _reconcile_with_rest so a lagging /v2/account
        # snapshot can't erase a just-filled cash delta.
        self._cash_mut_ts: float = 0.0

        # --------- Sanity check + dirty-start guard (in parallel) ----------
        # All three calls are required for the dirty-start decision:
        # "can't read state" must not silently read as "account is clean".
        account, positions, open_orders = self._initial_snapshot_parallel()
        self._cash = float(account.get("cash", 0.0))
        self._last_rest_snapshot["account"] = account
        self._last_rest_snapshot["positions"] = positions
        self._last_rest_snapshot["ts"] = time.monotonic()

        # Dirty-start guard: residual positions carry a cost basis from
        # a prior session, and residual open orders can fill mid-this-
        # session without the agent knowing. Either contaminates PnL.
        residual_positions = [
            p for p in positions
            if isinstance(p, dict)
            and abs(float(p.get("qty", 0.0))) > _QTY_EPSILON
        ]
        residual_orders = [
            o for o in open_orders if isinstance(o, dict) and o.get("id")
        ]
        if residual_positions or residual_orders:
            if not flatten_on_start:
                raise RuntimeError(
                    self._format_dirty_start_error(
                        residual_positions, residual_orders
                    )
                )
            logger.warning(
                "Alpaca paper account is dirty at startup "
                "(positions=%d, open_orders=%d); flatten_on_start=True, "
                "flushing before trial begins.",
                len(residual_positions),
                len(residual_orders),
            )
            post_flatten_cash = self._flatten_account_inline(
                expected_positions=residual_positions,
                expected_orders=residual_orders,
            )
            self._cash = post_flatten_cash
            self._last_rest_snapshot["positions"] = []
        # Invariant: _positions / _cost_basis never seed from REST —
        # they only reflect trades observed in THIS session, so PnL
        # attribution stays honest across executor restarts.

        # --------- Launch daemon threads ----------
        self._refresher = threading.Thread(
            target=self._account_refresh_loop,
            daemon=True,
            name="alpaca-acct-refresh",
        )
        self._refresher.start()
        self._ws_listener = threading.Thread(
            target=self._trade_updates_loop,
            daemon=True,
            name="alpaca-trade-updates",
        )
        self._ws_listener.start()

        # atexit hook to stop daemons before asyncio's internal threadpool
        # tears down (else the WS reconnect emits "cannot schedule new
        # futures after interpreter shutdown"). Captures Events only so
        # self can still be GC'd normally.
        _refresh_stop = self._refresh_stop
        _refresh_event = self._refresh_event

        def _signal_stop_at_exit() -> None:
            _refresh_stop.set()
            _refresh_event.set()

        atexit.register(_signal_stop_at_exit)

    def _nudge_refresh(self) -> None:
        """Ask the background reconciler to validate sooner, debounced.

        A burst of submits within :attr:`_NUDGE_MIN_INTERVAL_S` coalesces
        to one early reconcile. Drift correction is at most one window
        later than an un-debounced nudge — negligible against the 5s
        periodic cadence — while extra REST volume stays bounded.
        """
        now = time.monotonic()
        if now - self._last_nudge_ts >= self._NUDGE_MIN_INTERVAL_S:
            self._last_nudge_ts = now
            self._refresh_event.set()

    # ------------------------------------------------------------------
    # Construction-time helpers
    # ------------------------------------------------------------------

    def _initial_snapshot_parallel(self) -> tuple[dict, list, list]:
        """Fetch ``/v2/account``, ``/v2/positions`` and
        ``/v2/orders?status=open`` concurrently.

        All three are required for the dirty-start decision, so
        failure on any of them is fatal — we raise ``ValueError`` so
        the executor never proceeds against an unknown broker state.
        Returns ``(account, positions, open_orders)`` where positions
        and open_orders are guaranteed to be lists (empty if the
        endpoint returned a non-list payload).
        """
        with ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="alpaca-init"
        ) as pool:
            fut_account = pool.submit(self._get, "/v2/account")
            fut_positions = pool.submit(self._get, "/v2/positions")
            fut_orders = pool.submit(
                self._get, "/v2/orders", {"status": "open", "limit": 500}
            )
            try:
                account = fut_account.result()
            except Exception as exc:
                raise ValueError(
                    f"Alpaca paper-trading sanity check failed (could not "
                    f"GET /v2/account): {exc}"
                ) from exc
            try:
                positions = fut_positions.result()
            except Exception as exc:
                raise ValueError(
                    f"Alpaca paper-trading dirty-start check failed (could "
                    f"not GET /v2/positions): {exc}. With the new clean-"
                    f"start contract we cannot proceed without knowing "
                    f"whether the account is clean."
                ) from exc
            try:
                open_orders = fut_orders.result()
            except Exception as exc:
                raise ValueError(
                    f"Alpaca paper-trading dirty-start check failed (could "
                    f"not GET /v2/orders?status=open): {exc}. With the new "
                    f"clean-start contract we cannot proceed without "
                    f"knowing whether the account is clean."
                ) from exc
        if not isinstance(positions, list):
            positions = []
        if not isinstance(open_orders, list):
            open_orders = []
        return account, positions, open_orders

    @staticmethod
    def _format_dirty_start_error(
        residual_positions: list[dict], residual_orders: list[dict]
    ) -> str:
        """Build the RuntimeError message for a dirty start.

        Lists every offending position (symbol, signed qty, avg entry
        price) and every open order (id, symbol, side, qty, type) so
        the operator can decide whether to flatten manually via the
        Alpaca dashboard or re-launch with ``flatten_on_start=True``.
        """
        lines: list[str] = [
            "Alpaca paper-trading account is not clean at session start.",
            (
                "Per-session PnL accounting requires a flat broker state — "
                "residual positions carry a cost basis from a prior session "
                "and residual orders can fill mid-this-session without the "
                "agent knowing. Either contaminates PnL."
            ),
            "",
        ]
        if residual_positions:
            lines.append(
                f"Open positions ({len(residual_positions)}):"
            )
            for p in residual_positions:
                sym = p.get("symbol", "?")
                qty = p.get("qty", "?")
                avg = p.get("avg_entry_price", "?")
                side = p.get("side", "?")
                lines.append(
                    f"  - {sym}: qty={qty} avg_entry_price={avg} side={side}"
                )
            lines.append("")
        if residual_orders:
            lines.append(
                f"Open orders ({len(residual_orders)}):"
            )
            for o in residual_orders:
                lines.append(
                    "  - id={id} symbol={symbol} side={side} qty={qty} "
                    "type={type} status={status}".format(
                        id=o.get("id", "?"),
                        symbol=o.get("symbol", "?"),
                        side=o.get("side", "?"),
                        qty=o.get("qty", "?"),
                        type=o.get("type", "?"),
                        status=o.get("status", "?"),
                    )
                )
            lines.append("")
        lines.append(
            "To proceed, either close the residual state manually via the "
            "Alpaca paper-trading dashboard OR construct the executor with "
            "flatten_on_start=True (TradingConfig.flatten_on_start) to "
            "have it issue DELETE /v2/orders + DELETE /v2/positions "
            "before the trial begins."
        )
        return "\n".join(lines)

    def _flatten_and_reread_cash(
        self,
        *,
        delete_orders: bool,
        delete_positions: bool,
    ) -> float:
        """Cancel orders, close positions, poll until flat, re-read cash.

        Shared REST core of :meth:`reset` and the dirty-start
        :meth:`_flatten_account_inline`. Order matters: cancel orders
        BEFORE closing positions so a GTC limit can't fire mid-close and
        reopen a just-closed position. Each DELETE is independently
        fault-tolerant. The position poll retries until
        :attr:`_FLATTEN_POLL_TIMEOUT_SEC` — a transient GET failure counts
        as "still dirty", never as a reason to stop waiting.

        Out-of-hours, ``DELETE /v2/positions`` is accepted but the
        liquidations queue until market open, so the poll times out
        without seeing positions clear; we log and proceed (the
        agent_runtime market-staleness pre-flight would already have
        aborted a closed-market trial).

        Returns the post-flatten cash. Falls back to the current local
        ``_cash`` if ``/v2/account`` is unreadable (never zeroes it).
        """
        if delete_orders:
            try:
                self._delete("/v2/orders")
            except Exception:
                logger.warning(
                    "DELETE /v2/orders failed during flatten; continuing "
                    "into position close-out anyway",
                    exc_info=True,
                )
        if delete_positions:
            try:
                self._delete("/v2/positions")
            except Exception:
                logger.warning(
                    "DELETE /v2/positions failed during flatten",
                    exc_info=True,
                )
        # Poll until positions report empty. Retry-until-deadline: a
        # transient GET failure is treated as "still dirty", not a reason
        # to abort the wait.
        deadline = time.monotonic() + self._FLATTEN_POLL_TIMEOUT_SEC
        cleared = False
        remaining_count = 0
        while time.monotonic() < deadline:
            try:
                positions = self._get("/v2/positions")
            except Exception:
                time.sleep(0.1)
                continue
            if isinstance(positions, list):
                if not positions:
                    cleared = True
                    break
                remaining_count = len(positions)
            time.sleep(0.1)
        if not cleared:
            # Out-of-hours queue or transient REST failures: Alpaca
            # accepted the close but couldn't confirm empty in time. Log
            # so operators can correlate any unexpected next-session PnL.
            logger.warning(
                "flatten: positions did not confirm empty within %.1fs "
                "(last seen %d). Proceeding; the next reset()/agent_runtime "
                "pre-flight will catch any residual state.",
                self._FLATTEN_POLL_TIMEOUT_SEC,
                remaining_count,
            )
        # Re-read cash so the local model reflects realised PnL Alpaca
        # applied during close-out.
        try:
            account = self._get("/v2/account")
            cash = float(account.get("cash", 0.0))
            self._last_rest_snapshot["account"] = account
        except Exception:
            logger.warning(
                "/v2/account refresh failed after flatten; keeping the "
                "current local cash value as a best-effort seed",
                exc_info=True,
            )
            cash = self._cash
        return cash

    def _flatten_account_inline(
        self,
        *,
        expected_positions: list[dict],
        expected_orders: list[dict],
    ) -> float:
        """Dirty-start flush. Thin wrapper over
        :meth:`_flatten_and_reread_cash` that issues only the DELETEs
        actually needed — at construction we already know which of orders
        / positions are residual.
        """
        return self._flatten_and_reread_cash(
            delete_orders=bool(expected_orders),
            delete_positions=bool(expected_positions),
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.post(url, json=body, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post_or_4xx(
        self, path: str, body: dict[str, Any]
    ) -> tuple[Any | None, str | None]:
        """POST that tolerates Alpaca 4xx -- returns (json, error_str)."""
        url = f"{self._base}{path}"
        try:
            resp = self._session.post(url, json=body, timeout=15)
        except requests.RequestException as exc:
            return None, f"network error: {exc}"
        if 400 <= resp.status_code < 500:
            try:
                detail = resp.json()
            except ValueError:
                detail = {"message": resp.text}
            msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            return None, f"alpaca {resp.status_code}: {msg}"
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            return None, f"alpaca http error: {exc}"
        return resp.json(), None

    def _delete(self, path: str) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.delete(url, timeout=15)
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    def _delete_or_4xx(self, path: str) -> tuple[bool, str | None]:
        url = f"{self._base}{path}"
        try:
            resp = self._session.delete(url, timeout=15)
        except requests.RequestException as exc:
            return False, f"network error: {exc}"
        if resp.status_code in (404, 422):
            try:
                detail = resp.json()
            except ValueError:
                detail = {"message": resp.text}
            msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            return False, f"alpaca {resp.status_code}: {msg}"
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            return False, f"alpaca http error: {exc}"
        return True, None

    # ------------------------------------------------------------------
    # Background daemon: periodic REST reconciliation
    # ------------------------------------------------------------------

    def _fetch_positions_and_account_parallel(
        self,
    ) -> tuple[Any, Any]:
        """Issue ``GET /v2/positions`` and ``GET /v2/account`` concurrently.

        Two separate Alpaca REST endpoints; can't be merged into a
        single HTTP call. Dispatching them on two threads cuts the
        wall-clock cost roughly in half (~200 ms → ~100 ms over the
        wire) at the cost of running two simultaneous requests against
        the trading API (well within the 200/min quota even at a 1 Hz
        refresh).

        Returns ``(positions, account)`` — same shape as if the two
        calls had been issued sequentially. Either result may be
        ``None`` if its request failed; the caller decides what to do.
        """
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="alpaca-acct"
        ) as pool:
            fut_positions = pool.submit(self._get, "/v2/positions")
            fut_account = pool.submit(self._get, "/v2/account")
            try:
                positions = fut_positions.result()
            except Exception:
                logger.debug("/v2/positions fetch failed", exc_info=True)
                positions = None
            try:
                account = fut_account.result()
            except Exception:
                logger.debug("/v2/account fetch failed", exc_info=True)
                account = None
        return positions, account

    def _account_refresh_loop(self) -> None:
        """Polls /v2/positions and /v2/account every refresh_interval_sec
        (in parallel), compares with local model, corrects drift.
        See ``_reconcile_with_rest``.

        Catches ``BaseException`` (not just ``Exception``) inside the
        loop body because during interpreter shutdown asyncio /
        ThreadPoolExecutor internals can raise ``RuntimeError`` or
        ``SystemExit`` that ``Exception`` doesn't cover. When that
        happens we silently exit if ``_refresh_stop`` is set; otherwise
        we best-effort log and continue.
        """
        while not self._refresh_stop.is_set():
            if not self._refresh_pause.is_set():
                try:
                    positions, account = self._fetch_positions_and_account_parallel()
                    # Reconcile if at least one fetch succeeded. If both
                    # failed, skip this cycle — keep the cached snapshot.
                    if positions is not None or account is not None:
                        self._reconcile_with_rest(
                            positions if positions is not None else [],
                            account if account is not None else {},
                        )
                except BaseException:  # noqa: BLE001 — daemon thread isolation
                    if self._refresh_stop.is_set():
                        return
                    self._safe_log(
                        "debug", "account refresh failed", exc_info=True
                    )
            try:
                # Wait either for the periodic timer OR an immediate-refresh
                # signal (set by submit/cancel/expire_all to nudge the
                # reconciler to validate sooner).
                self._refresh_event.wait(timeout=self._refresh_interval_sec)
                self._refresh_event.clear()
            except BaseException:
                return

    def _reconcile_with_rest(
        self, positions: Any, account: Any
    ) -> None:
        """Detect divergence between the local model and the REST snapshot,
        then resolve it via the authoritative order stream — never by
        trusting the eventually-consistent ``/v2/positions`` aggregate.

        ``/v2/positions`` and ``/v2/account`` lag the order lifecycle:
        right after a fill, ``GET /v2/orders/{id}`` already reports
        ``filled`` while these aggregates can still return the pre-fill
        snapshot for up to a few seconds. The previous policy (drop a
        local position when REST showed none; overwrite cash from REST
        unconditionally) clobbered a just-filled position/cash during
        that window. Instead:

        - **Positions** mutate ONLY from observed fills (WS
          ``trade_updates`` + the order-history replay below). Any
          local↔REST mismatch triggers
          :meth:`_sync_orders_since_last_event`, which replays the order
          events not yet processed. That replay is idempotent: a fill
          already booked is a no-op (so propagation lag self-resolves),
          while a genuinely missed fill or external close converges local
          to the truth via the order stream. ``/v2/positions`` is never
          allowed to drop or overwrite a position on its own — that is
          the race the live test surfaced.
        - **Cash** syncs from ``/v2/account`` only when the snapshot is
          consistent with local (not divergent) AND no local fill has
          touched cash within :attr:`_CASH_SYNC_IDLE_S`. The idle gate
          stops a lagging account snapshot from erasing a just-filled
          delta; the not-divergent gate stops a double-count when the
          order-history replay is about to (re)book the very fill the
          REST cash already reflects. Outside those windows the REST
          value still catches slow fill-less drift (fees, dividends,
          interest) that has no event-stream source.

        Hot path is untouched: this runs on the background refresher; the
        only added work is a divergence comparison and, when divergent,
        one extra ``/v2/orders`` replay.
        """
        divergent = False
        with self._snapshot_lock:
            now = time.monotonic()
            self._last_rest_snapshot["positions"] = positions
            self._last_rest_snapshot["account"] = account
            self._last_rest_snapshot["ts"] = now

            if isinstance(positions, list):
                rest_by_symbol: dict[str, float] = {}
                for p in positions:
                    if not isinstance(p, dict):
                        continue
                    sym = str(p.get("symbol", ""))
                    if not sym:
                        continue
                    try:
                        rest_by_symbol[sym] = float(p.get("qty", 0.0))
                    except (TypeError, ValueError):
                        rest_by_symbol[sym] = 0.0

                # DETECT only — never mutate positions from the aggregate.
                for sym, rest_qty in rest_by_symbol.items():
                    if abs(
                        rest_qty - self._positions.get(sym, 0.0)
                    ) > _QTY_EPSILON:
                        divergent = True
                        break
                if not divergent:
                    for sym, local_qty in self._positions.items():
                        if (
                            abs(local_qty) > _QTY_EPSILON
                            and sym not in rest_by_symbol
                        ):
                            divergent = True
                            break

            # Cash: sync only when consistent (not divergent) and idle.
            if (
                isinstance(account, dict)
                and not divergent
                and now - self._cash_mut_ts >= self._CASH_SYNC_IDLE_S
            ):
                self._cash = float(account.get("cash", self._cash))

        # Resolve OUTSIDE the lock: _sync_orders_since_last_event does
        # network I/O and _handle_trade_update re-acquires _snapshot_lock.
        if divergent and not self._refresh_stop.is_set():
            logger.info(
                "reconcile: local positions diverge from /v2/positions; "
                "replaying order history to resolve via the order stream"
            )
            self._sync_orders_since_last_event()

    # ------------------------------------------------------------------
    # Background daemon: WebSocket trade_updates listener
    # ------------------------------------------------------------------

    def _trade_updates_loop(self) -> None:
        """Run the WS session in a fresh asyncio loop; reconnect on drop.

        On reconnect (after any exception), we issue a one-shot REST
        sync via /v2/orders?after=<_last_trade_event_ts> to replay any
        fills we may have missed during the gap. `_handle_trade_update`
        is idempotent via `_reported_order_ids`.

        Catches ``BaseException`` because during interpreter shutdown
        asyncio's internal ThreadPoolExecutor (used for DNS resolution
        via getaddrinfo) can be torn down before our daemon thread
        sees ``_refresh_stop``. The resulting ``RuntimeError: cannot
        schedule new futures after interpreter shutdown`` doesn't
        inherit from ``Exception`` cleanly across all Python versions,
        and even if it did, ``logger.warning`` may itself fail during
        shutdown. We short-circuit silently when shutting down.
        """
        while not self._refresh_stop.is_set():
            try:
                asyncio.run(self._trade_updates_session())
            except BaseException:  # noqa: BLE001 — daemon thread isolation
                # Most WS errors are now caught inside the coroutine
                # (see _trade_updates_session). This block only fires
                # when asyncio.run itself fails — typically only
                # during interpreter shutdown. Bail silently in that
                # case.
                if self._refresh_stop.is_set():
                    return
                self._safe_log(
                    "debug",
                    "asyncio.run for trade_updates failed; "
                    "reconnecting in 1s",
                    exc_info=True,
                )
            if self._refresh_stop.is_set():
                return
            try:
                time.sleep(1.0)
            except BaseException:
                return
            if self._refresh_stop.is_set():
                return
            # On reconnect, fetch any orders updated since our last
            # known event to fill the gap.
            try:
                self._sync_orders_since_last_event()
            except BaseException:
                if self._refresh_stop.is_set():
                    return
                self._safe_log(
                    "debug",
                    "sync_orders_since_last_event failed",
                    exc_info=True,
                )

    @staticmethod
    def _safe_log(level: str, msg: str, *, exc_info: bool = False) -> None:
        """Best-effort logger call that swallows any error.

        Used inside daemon-thread exception handlers. During interpreter
        shutdown the logging module can itself fail (file handles
        closed, lock acquisition raises). Suppressing here keeps the
        traceback from escaping the daemon.
        """
        try:
            getattr(logger, level)(msg, exc_info=exc_info)
        except BaseException:
            pass

    async def _trade_updates_session(self) -> None:
        """Single-pass WS session: auth, subscribe, dispatch updates.

        All exceptions are caught inside this coroutine and converted
        to a normal return. This prevents asyncio's default exception
        handler from logging the traceback to stderr when the WS drops
        (or when interpreter shutdown races our connection attempt).
        The outer ``_trade_updates_loop`` decides whether to reconnect
        based on ``_refresh_stop``, not on whether this coroutine
        raised.
        """
        import websockets  # optional dependency
        try:
            async with websockets.connect(self._ws_url) as ws:
                await ws.send(
                    _json.dumps(
                        {
                            "action": "authenticate",
                            "data": {
                                "key_id": self._api_key,
                                "secret_key": self._api_secret,
                            },
                        }
                    )
                )
                # Drain auth response (success / authorization)
                await ws.recv()
                await ws.send(
                    _json.dumps(
                        {
                            "action": "listen",
                            "data": {"streams": ["trade_updates"]},
                        }
                    )
                )
                # Drain listen confirmation
                await ws.recv()
                async for raw in ws:
                    if self._refresh_stop.is_set():
                        return
                    try:
                        payload = _json.loads(raw)
                    except ValueError:
                        continue
                    # Alpaca wraps trade_updates in {"stream": "trade_updates",
                    # "data": {...event payload...}}. Tolerate the bare-event
                    # shape too (some test fixtures use that).
                    data = (
                        payload.get("data") if isinstance(payload, dict) else None
                    )
                    if data is None:
                        data = payload
                    self._handle_trade_update(data)
        except BaseException:  # noqa: BLE001 — daemon thread isolation
            # Swallow ALL exceptions, including the
            # "cannot schedule new futures after interpreter shutdown"
            # RuntimeError that asyncio internals can raise during
            # Python teardown. If we're shutting down we just return;
            # otherwise we log at debug (warning would be too noisy for
            # routine reconnect cycles) and let the outer loop's sleep
            # + retry handle it.
            if not self._refresh_stop.is_set():
                self._safe_log(
                    "debug",
                    "trade_updates WS session error (will reconnect)",
                    exc_info=True,
                )

    def _handle_trade_update(self, msg: dict[str, Any]) -> None:
        """Apply a single trade_updates event to the local model.

        Recognised events:
        - ``fill`` / ``partial_fill``: apply the fill (update position,
          cost basis, cash) and surface in the trade log. If a
          PROVISIONAL entry already exists for this order_id from
          ``submit()``, update it in-place; otherwise append a new
          FILLED entry.
        - ``canceled`` / ``expired`` / ``rejected``: append a record
          with the matching status; pop from _local_pending.

        Idempotent via _reported_order_ids — duplicate WS pushes
        (network retry, reconnect gap-fill) are no-ops.
        """
        if not isinstance(msg, dict):
            return
        event = str(msg.get("event", ""))
        order = msg.get("order")
        if not isinstance(order, dict):
            return
        order_id = str(order.get("id", ""))
        if not order_id:
            return
        # Record the event timestamp for reconnect gap-fill, even if we
        # end up skipping this event below.
        ts_raw = msg.get("timestamp") or order.get("updated_at")
        if isinstance(ts_raw, str):
            self._last_trade_event_ts = ts_raw

        if event in ("fill", "partial_fill"):
            self._apply_fill_event(order, order_id)
        elif event in ("canceled", "cancelled", "expired", "rejected"):
            self._apply_terminal_non_fill(order, order_id, event)
        # Other events (new, accepted, pending_new, etc.) are
        # informational only — local state already reflects the
        # provisional booking from submit().

    def _apply_fill_event(
        self, order: dict[str, Any], order_id: str
    ) -> None:
        symbol = str(order.get("symbol", ""))
        if not symbol:
            return
        side = str(order.get("side", ""))
        if side not in (ActionVerb.BUY, ActionVerb.SELL):
            return
        try:
            filled_qty = float(order.get("filled_qty") or order.get("qty") or 0.0)
            filled_avg_price = float(order.get("filled_avg_price") or 0.0)
        except (TypeError, ValueError):
            return
        if filled_qty <= 0.0 or filled_avg_price <= 0.0:
            return
        ts_raw = order.get("filled_at") or order.get("updated_at")
        try:
            ts = (
                datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                if ts_raw
                else datetime.now(timezone.utc)
            )
        except ValueError:
            ts = datetime.now(timezone.utc)

        order_type = self._reverse_alpaca_type(
            str(order.get("type", "market"))
        )
        tif = self._reverse_alpaca_tif(str(order.get("time_in_force", "gtc")))
        limit_price = (
            float(order["limit_price"])
            if order.get("limit_price") is not None
            else None
        )
        stop_price = (
            float(order["stop_price"])
            if order.get("stop_price") is not None
            else None
        )

        with self._snapshot_lock:
            # Already processed this fill? (e.g., duplicate WS push or
            # reconnect REST gap-fill replaying).
            if order_id in self._reported_order_ids:
                # If a matching trade-log entry still has PROVISIONAL
                # status, upgrade it to FILLED with the real price.
                # (Common case: submit booked provisional, WS arrived,
                # we processed it once; then a duplicate push arrives.)
                self._upgrade_provisional_to_filled_locked(
                    order_id,
                    real_price=filled_avg_price,
                    filled_qty=filled_qty,
                    side=side,
                    symbol=symbol,
                )
                return

            # Find an existing PROVISIONAL entry for this order_id.
            provisional_idx = self._find_trade_log_idx_by_order_id(
                order_id, status_filter=OrderStatus.PROVISIONAL
            )
            if provisional_idx is not None:
                # Upgrade in place + apply price correction delta.
                prov = self._trade_log[provisional_idx]
                price_delta = filled_avg_price - prov.executed_price
                signed = 1.0 if side == ActionVerb.BUY else -1.0
                self._cost_basis[symbol] = (
                    self._cost_basis.get(symbol, 0.0)
                    + price_delta * filled_qty * signed
                )
                # Cash correction: buys overestimated cost (we debited
                # market_price * qty), sells underestimated proceeds.
                self._cash -= price_delta * filled_qty * signed
                self._cash_mut_ts = time.monotonic()
                prov.executed_price = filled_avg_price
                prov.market_price = filled_avg_price
                prov.slippage_cost = 0.0
                prov.status = OrderStatus.FILLED
                prov.timestamp = ts
            else:
                # No matching provisional → fill arrived before submit
                # finished (race) OR this is a fill for an order from a
                # prior session OR a partial_fill for a not-yet-booked
                # remainder. Apply as a fresh fill.
                self._book_fill_locked(
                    symbol=symbol,
                    side=side,
                    qty=filled_qty,
                    price=filled_avg_price,
                    order_id=order_id,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    tif=tif,
                    ts=ts,
                )

            # Mark order_id as processed and drop any local pending
            # entry. (partial_fill conceptually shouldn't drop pending
            # — but Alpaca's paper engine almost always fills market
            # orders in one shot, so the simplification is OK for the
            # curated bundles.)
            self._reported_order_ids.add(order_id)
            self._local_pending.pop(order_id, None)

    def _apply_terminal_non_fill(
        self, order: dict[str, Any], order_id: str, event: str
    ) -> None:
        symbol = str(order.get("symbol", ""))
        side = str(order.get("side", ActionVerb.HOLD))
        try:
            qty = float(order.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        order_type = self._reverse_alpaca_type(
            str(order.get("type", "market"))
        )
        tif = self._reverse_alpaca_tif(str(order.get("time_in_force", "gtc")))
        limit_price = (
            float(order["limit_price"])
            if order.get("limit_price") is not None
            else None
        )
        stop_price = (
            float(order["stop_price"])
            if order.get("stop_price") is not None
            else None
        )
        mapped = (
            OrderStatus.CANCELLED
            if event.startswith("cancel")
            else (
                OrderStatus.REJECTED
                if event == "rejected"
                else OrderStatus.EXPIRED
            )
        )
        with self._snapshot_lock:
            if order_id in self._reported_order_ids:
                return
            local = self._local_pending.pop(order_id, None)
            submitted_step = local.submitted_step if local else 0
            client_tag = local.client_tag if local else None
            report = ExecutionReport(
                action=side,
                symbol=symbol,
                quantity=qty,
                market_price=0.0,
                executed_price=0.0,
                timestamp=datetime.now(timezone.utc),
                order_id=order_id,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                tif=tif,
                status=mapped,
                submitted_step=submitted_step,
                client_tag=client_tag,
            )
            self._trade_log.append(report)
            self._reported_order_ids.add(order_id)

    def _book_fill_locked(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_id: str,
        order_type: str,
        limit_price: float | None,
        stop_price: float | None,
        tif: str,
        ts: datetime,
        submitted_step: int = 0,
        status: str = OrderStatus.FILLED,
    ) -> ExecutionReport:
        """Apply a fill to local state. Caller MUST hold _snapshot_lock."""
        signed = 1.0 if side == ActionVerb.BUY else -1.0
        notional = price * qty
        self._positions[symbol] = (
            self._positions.get(symbol, 0.0) + qty * signed
        )
        # Collapse to 0 to keep the dict tidy. (Math still works either
        # way because the entry is multiplied by current_price.)
        if abs(self._positions[symbol]) < _QTY_EPSILON:
            self._positions.pop(symbol, None)
        self._cost_basis[symbol] = (
            self._cost_basis.get(symbol, 0.0) + price * qty * signed
        )
        self._cash -= notional * signed  # buy debits, sell credits
        self._cash_mut_ts = time.monotonic()
        report = ExecutionReport(
            action=side,
            symbol=symbol,
            quantity=qty,
            market_price=price,
            executed_price=price,
            timestamp=ts,
            order_id=order_id,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            status=status,
            submitted_step=submitted_step,
        )
        self._trade_log.append(report)
        return report

    def _find_trade_log_idx_by_order_id(
        self, order_id: str, status_filter: str | None = None
    ) -> int | None:
        for i in range(len(self._trade_log) - 1, -1, -1):
            report = self._trade_log[i]
            if report.order_id == order_id:
                if status_filter is None or report.status == status_filter:
                    return i
        return None

    def _upgrade_provisional_to_filled_locked(
        self,
        order_id: str,
        *,
        real_price: float,
        filled_qty: float,
        side: str,
        symbol: str,
    ) -> None:
        idx = self._find_trade_log_idx_by_order_id(
            order_id, status_filter=OrderStatus.PROVISIONAL
        )
        if idx is None:
            return
        prov = self._trade_log[idx]
        price_delta = real_price - prov.executed_price
        if abs(price_delta) > 1e-9:
            signed = 1.0 if side == ActionVerb.BUY else -1.0
            self._cost_basis[symbol] = (
                self._cost_basis.get(symbol, 0.0)
                + price_delta * filled_qty * signed
            )
            self._cash -= price_delta * filled_qty * signed
            self._cash_mut_ts = time.monotonic()
        prov.executed_price = real_price
        prov.market_price = real_price
        prov.slippage_cost = 0.0
        prov.status = OrderStatus.FILLED

    def _sync_orders_since_last_event(self) -> None:
        """REST gap-fill on WS reconnect.

        Fetches /v2/orders updated since `_last_trade_event_ts` and
        replays each through _handle_trade_update. Idempotent.
        """
        params: dict[str, Any] = {"status": "all", "limit": 200}
        if self._last_trade_event_ts:
            params["after"] = self._last_trade_event_ts
        try:
            recent = self._get("/v2/orders", params=params)
        except Exception:
            return
        if not isinstance(recent, list):
            return
        for entry in recent:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", ""))
            if status == "filled":
                event = "fill"
            elif status in ("partially_filled",):
                event = "partial_fill"
            elif status in ("canceled", "cancelled"):
                event = "canceled"
            elif status == "expired":
                event = "expired"
            elif status == "rejected":
                event = "rejected"
            else:
                continue
            synthetic = {"event": event, "order": entry,
                         "timestamp": entry.get("updated_at")}
            self._handle_trade_update(synthetic)

    # ------------------------------------------------------------------
    # BaseExecutor interface — submission lifecycle
    # ------------------------------------------------------------------

    def submit(
        self,
        intent: OrderIntent,
        market_price: float,
        *,
        timestamp: datetime | None = None,
        step: int = 0,
    ) -> SubmitResult:
        ts = timestamp or datetime.now(timezone.utc)

        # HOLD: synthetic, no broker interaction.
        if intent.action == ActionVerb.HOLD:
            report = ExecutionReport(
                action=ActionVerb.HOLD,
                symbol=intent.symbol,
                quantity=0.0,
                market_price=market_price,
                executed_price=market_price,
                timestamp=ts,
                order_type=OrderType.MARKET,
                tif=TimeInForce.IOC,
                status=OrderStatus.FILLED,
                submitted_step=step,
                filled_step=step,
                client_tag=intent.client_tag,
            )
            with self._snapshot_lock:
                self._trade_log.append(report)
            return SubmitResult(kind="filled", fill=report)

        # Schema validation.
        if intent.action not in (ActionVerb.BUY, ActionVerb.SELL):
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_ACTION,
                    reason=(
                        f"Unknown action: {intent.action!r} "
                        "(expected buy/sell/hold)"
                    ),
                ),
            )
        if not (intent.quantity > 0.0):
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_QUANTITY,
                    reason=(
                        f"Quantity must be positive for {intent.action}, "
                        f"got {intent.quantity}"
                    ),
                ),
            )
        if intent.order_type not in _ALPACA_TYPE:
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_ORDER_TYPE,
                    reason=f"Unsupported order_type: {intent.order_type!r}",
                ),
            )
        if intent.tif not in _ALPACA_TIF:
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=RejectionCode.INVALID_TIF,
                    reason=f"Unsupported tif: {intent.tif!r}",
                ),
            )

        order_body: dict[str, Any] = {
            "symbol": intent.symbol,
            "qty": str(intent.quantity),
            "side": intent.action,
            "type": _ALPACA_TYPE[intent.order_type],
            "time_in_force": _ALPACA_TIF[intent.tif],
        }
        if intent.limit_price is not None:
            order_body["limit_price"] = str(intent.limit_price)
        if intent.stop_price is not None:
            order_body["stop_price"] = str(intent.stop_price)

        # POST is a network call; no lock held.
        order, err = self._post_or_4xx("/v2/orders", order_body)
        if err is not None:
            lower = err.lower()
            code = RejectionCode.SCHEMA_ERROR
            if "buying power" in lower or "insufficient" in lower:
                code = RejectionCode.INSUFFICIENT_CASH
            elif "qty" in lower or "quantity" in lower:
                code = RejectionCode.INVALID_QUANTITY
            elif "price" in lower:
                code = RejectionCode.INVALID_PRICE
            elif "symbol" in lower:
                code = RejectionCode.UNKNOWN_SYMBOL
            return SubmitResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent=intent,
                    reason_code=code,
                    reason=err,
                ),
            )

        order_id = str(order["id"])

        if intent.order_type == OrderType.MARKET:
            # Provisional booking at market_price. The WS trade_updates
            # push will arrive shortly with the real filled_avg_price
            # and adjust the local model in-place. Synchronous-fill
            # contract preserved for callers: we return kind="filled"
            # with an ExecutionReport, just with status=PROVISIONAL.
            with self._snapshot_lock:
                # Race guard: if WS already reported this order (rare
                # but possible with very fast paper fills), skip
                # provisional booking — the WS handler already booked
                # the real fill.
                if order_id in self._reported_order_ids:
                    real_idx = self._find_trade_log_idx_by_order_id(order_id)
                    if real_idx is not None:
                        return SubmitResult(
                            kind="filled", fill=self._trade_log[real_idx]
                        )
                report = self._book_fill_locked(
                    symbol=intent.symbol,
                    side=intent.action,
                    qty=intent.quantity,
                    price=market_price,
                    order_id=order_id,
                    order_type=intent.order_type,
                    limit_price=intent.limit_price,
                    stop_price=intent.stop_price,
                    tif=intent.tif,
                    ts=ts,
                    submitted_step=step,
                    status=OrderStatus.PROVISIONAL,
                )
            # Nudge the REST refresher to validate soon (debounced).
            self._nudge_refresh()
            return SubmitResult(kind="filled", fill=report)

        # Non-market (limit/stop/stop_limit): queue locally; the WS will
        # report when (if) the fill happens.
        pending = PendingOrder(
            order_id=order_id,
            symbol=intent.symbol,
            action=intent.action,
            quantity=intent.quantity,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            tif=intent.tif,
            submitted_at=ts,
            submitted_step=step,
            client_tag=intent.client_tag,
        )
        with self._snapshot_lock:
            self._local_pending[order_id] = pending
        return SubmitResult(kind="accepted", accepted=pending)

    def cancel(self, order_id: str) -> CancelResult:
        ok, err = self._delete_or_4xx(f"/v2/orders/{order_id}")
        if not ok:
            return CancelResult(
                kind="rejected",
                rejection=Rejection(
                    order_intent={"action": "cancel", "order_id": order_id},
                    reason_code=RejectionCode.ORDER_NOT_FOUND,
                    reason=err or "alpaca rejected cancel",
                ),
            )
        with self._snapshot_lock:
            order = self._local_pending.pop(order_id, None)
            if order is None:
                order = PendingOrder(
                    order_id=order_id,
                    symbol="",
                    action="",
                    quantity=0.0,
                    order_type=OrderType.LIMIT,
                    limit_price=None,
                    stop_price=None,
                    tif=TimeInForce.GTC,
                )
            # WS push will arrive shortly to confirm the cancel; the
            # handler is idempotent via _reported_order_ids. We don't
            # eagerly record the cancellation here to avoid duplicating
            # the audit log entry the WS handler will write.
        return CancelResult(kind="cancelled", cancelled=order)

    def tick(
        self,
        prices: dict[str, float],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> TickResult:
        """No-op for AlpacaPaperExecutor.

        Fills, cancels, and expirations are pushed via the WebSocket
        ``trade_updates`` listener. ``prices`` is unused. Method is
        retained for interface parity with :class:`BaseExecutor` /
        :class:`SimulatedExecutor`; callers (e.g.,
        :class:`RealtimeTradingTask`) invoke it once per step and
        consume whatever it returns. An empty TickResult is correct
        here because all fill notifications have already been applied
        to the local model by the WS daemon by the time the gym loop
        reads from it.
        """
        return TickResult()

    def expire_all(
        self,
        prices: dict[str, float] | None = None,
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> list[ExecutionReport]:
        """Cancel every locally-tracked open order on Alpaca.

        Per-order ``DELETE /v2/orders/{id}`` calls dispatch in parallel
        via a transient thread pool. Each DELETE is independent on
        Alpaca's side (different order ids, different responses); the
        local-state mutations (pop from ``_local_pending``, append to
        ``_trade_log``, add to ``_reported_order_ids``) happen under
        ``_snapshot_lock`` per order, so cross-order concurrency is
        safe. Wall-clock collapses from ``O(K_pending × RTT)`` to
        ``O(RTT)`` at episode end — large win for trials that
        accumulate many open limit/stop orders.
        """
        ts = timestamp or datetime.now(timezone.utc)
        out: list[ExecutionReport] = []
        with self._snapshot_lock:
            order_ids = list(self._local_pending.keys())
        if not order_ids:
            self._refresh_event.set()
            return out

        def _delete_one(oid: str) -> tuple[str, bool, str | None]:
            ok, err = self._delete_or_4xx(f"/v2/orders/{oid}")
            return oid, ok, err

        # Transient pool — expire_all runs once per episode end, so
        # per-call setup cost is negligible. Worker count capped at 8
        # to avoid bursting Alpaca's order-rate limit too hard.
        with ThreadPoolExecutor(
            max_workers=min(len(order_ids), 8),
            thread_name_prefix="alpaca-expire",
        ) as pool:
            futures = [pool.submit(_delete_one, oid) for oid in order_ids]
            for future in futures:
                order_id, _ok, _err = future.result()
                with self._snapshot_lock:
                    order = self._local_pending.pop(order_id, None)
                    if order is None:
                        continue
                    # Synthesize an EXPIRED entry; the WS handler is
                    # idempotent so if it later pushes a "canceled" event
                    # for the same order it will be a no-op (the order_id
                    # is in _reported_order_ids by then).
                    report = ExecutionReport(
                        action=order.action,
                        symbol=order.symbol,
                        quantity=order.quantity,
                        market_price=0.0,
                        executed_price=0.0,
                        timestamp=ts,
                        order_id=order_id,
                        order_type=order.order_type,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        tif=order.tif,
                        status=OrderStatus.EXPIRED,
                        submitted_step=order.submitted_step,
                        filled_step=step,
                        client_tag=order.client_tag,
                    )
                    self._trade_log.append(report)
                    self._reported_order_ids.add(order_id)
                    out.append(report)
        # Force a refresh so the cleared pending state propagates.
        self._refresh_event.set()
        return out

    # ------------------------------------------------------------------
    # BaseExecutor interface — accessors (all read from local model)
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        with self._snapshot_lock:
            return dict(self._positions)

    def get_pending_orders(self) -> list[PendingOrder]:
        with self._snapshot_lock:
            return list(self._local_pending.values())

    def get_trade_log(self) -> list[ExecutionReport]:
        with self._snapshot_lock:
            return list(self._trade_log)

    def get_cash(self) -> float | None:
        with self._snapshot_lock:
            return self._cash

    def get_reserved_cash(self) -> float:
        # Best-effort estimate from local pending buys.
        total = 0.0
        with self._snapshot_lock:
            for order in self._local_pending.values():
                if order.action == ActionVerb.BUY:
                    ref = order.limit_price or order.stop_price or 0.0
                    total += ref * order.quantity
        return total

    def compute_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Pure local PnL computation: ``current_price * qty - cost_basis - tx_total``.

        Uses fresh WS-fed prices (passed in from the buffer) against
        locally-tracked positions and cost basis. Zero network calls.
        See class docstring for the reconciliation policy that keeps
        the local model honest.
        """
        pnl: dict[str, float] = {}
        total = 0.0
        with self._snapshot_lock:
            for sym, qty in self._positions.items():
                if abs(qty) < _QTY_EPSILON:
                    continue
                cur = current_prices.get(sym)
                cost = self._cost_basis.get(sym, 0.0)
                tx = self._tx_cost_total.get(sym, 0.0)
                if cur is None:
                    v = -tx
                else:
                    v = cur * qty - cost - tx
                pnl[sym] = v
                total += v
        pnl["__total"] = total
        return pnl

    def reset(self) -> None:
        """Cancel open orders, close all positions on Alpaca, and clear
        local state.

        Sequence:
        1. Suspend the REST refresher (so a concurrent refresh doesn't
           overwrite our just-cleared state with Alpaca's pre-DELETE
           snapshot).
        2. Flatten the broker via :meth:`_flatten_and_reread_cash`
           (DELETE orders → DELETE positions → poll empty → re-read cash).
           Orders are cancelled before positions so a GTC limit can't fire
           mid-close and reopen a freshly-closed position.
        3. Under lock: clear all local model state and adopt the fresh
           cash.
        4. Resume refresher; force an immediate refresh.

        The REST flatten is fault-tolerant (failures logged, not raised);
        the local-state clear always runs.
        """
        self._refresh_pause.set()
        try:
            fresh_cash = self._flatten_and_reread_cash(
                delete_orders=True, delete_positions=True
            )
            with self._snapshot_lock:
                self._positions.clear()
                self._cost_basis.clear()
                self._tx_cost_total.clear()
                self._trade_log.clear()
                self._local_pending.clear()
                self._reported_order_ids.clear()
                self._cash = fresh_cash
                self._last_trade_event_ts = None
        finally:
            self._refresh_pause.clear()
            self._refresh_event.set()

    def close(self) -> None:
        """Stop background daemons. Safe to call multiple times."""
        self._refresh_stop.set()
        self._refresh_event.set()  # wake refresher so it sees stop flag
        # Daemon threads are daemon=True so they won't block process
        # exit even if join times out. The .join() here is best-effort.
        try:
            self._refresher.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._ws_listener.join(timeout=2.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reverse_alpaca_type(raw: str) -> str:
        return _REV_ALPACA_TYPE.get(raw, OrderType.MARKET)

    @staticmethod
    def _reverse_alpaca_tif(raw: str) -> str:
        return _REV_ALPACA_TIF.get(raw, TimeInForce.GTC)
