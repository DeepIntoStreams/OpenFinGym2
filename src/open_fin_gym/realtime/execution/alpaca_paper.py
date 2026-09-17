"""Alpaca paper-trading executor: orders go to paper-api.alpaca.markets.

Fills arrive over the trade-updates WebSocket rather than polling, so ``tick``
is a no-op kept for parity with :class:`SimulatedExecutor`.
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

    # Poll budget after the dirty-start flatten; outside regular hours the
    # closes queue server-side and never clear here, so we log and proceed.
    _FLATTEN_POLL_TIMEOUT_SEC: float = 2.0

    # Debounce for the post-submit reconcile nudge, so a burst of orders costs
    # one early check instead of one per order.
    _NUDGE_MIN_INTERVAL_S: float = 0.25

    # Quiet period before cash syncs from REST, so the account aggregate's lag
    # cannot erase a just-filled delta.
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
        self._refresh_stop = threading.Event()  # set by close() to terminate
        self._last_nudge_ts: float = 0.0  # debounce for _nudge_refresh
        # Time of the last local cash change, gating the REST cash sync.
        self._cash_mut_ts: float = 0.0

        # All three calls are required: an unreadable account must not pass as
        # a clean one.
        account, positions, open_orders = self._initial_snapshot_parallel()
        self._cash = float(account.get("cash", 0.0))
        self._last_rest_snapshot["account"] = account
        self._last_rest_snapshot["positions"] = positions
        self._last_rest_snapshot["ts"] = time.monotonic()

        # Residual positions carry a prior cost basis and residual orders can
        # fill mid-session; either one contaminates PnL.
        residual_positions = [
            p
            for p in positions
            if isinstance(p, dict) and abs(float(p.get("qty", 0.0))) > _QTY_EPSILON
        ]
        residual_orders = [
            o for o in open_orders if isinstance(o, dict) and o.get("id")
        ]
        if residual_positions or residual_orders:
            if not flatten_on_start:
                raise RuntimeError(
                    self._format_dirty_start_error(residual_positions, residual_orders)
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
        # Positions and cost basis never seed from REST, so PnL only reflects
        # trades this session observed.

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

        # Stop the daemons before asyncio's threadpool tears down; it captures
        # Events only, so self stays collectable.
        _refresh_stop = self._refresh_stop
        _refresh_event = self._refresh_event

        def _signal_stop_at_exit() -> None:
            _refresh_stop.set()
            _refresh_event.set()

        atexit.register(_signal_stop_at_exit)

    def _nudge_refresh(self) -> None:
        """Ask the background reconciler to validate sooner, debounced."""
        now = time.monotonic()
        if now - self._last_nudge_ts >= self._NUDGE_MIN_INTERVAL_S:
            self._last_nudge_ts = now
            self._refresh_event.set()

    # ── Construction-time helpers ────────────────────────────────────────

    def _initial_snapshot_parallel(self) -> tuple[dict, list, list]:
        """Fetch ``/v2/account``, ``/v2/positions`` and ``/v2/orders?status=open`` concurrently.

        Raises:
            ValueError: Any of the three fails, since the dirty-start decision cannot
                be made against an unknown account.
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
            lines.append(f"Open positions ({len(residual_positions)}):")
            for p in residual_positions:
                sym = p.get("symbol", "?")
                qty = p.get("qty", "?")
                avg = p.get("avg_entry_price", "?")
                side = p.get("side", "?")
                lines.append(f"  - {sym}: qty={qty} avg_entry_price={avg} side={side}")
            lines.append("")
        if residual_orders:
            lines.append(f"Open orders ({len(residual_orders)}):")
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

        A failed poll counts as still dirty, and unreadable cash falls back to the
        local value rather than zeroing it.
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
        # A failed poll counts as still dirty rather than ending the wait.
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
            # Alpaca took the close but could not confirm it in time; log it so
            # unexpected next-session PnL can be traced.
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

    # ── HTTP helpers ─────────────────────────────────────────────────────

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

    # ── Background daemon: periodic REST reconciliation ──────────────────

    def _fetch_positions_and_account_parallel(
        self,
    ) -> tuple[Any, Any]:
        """Issue ``GET /v2/positions`` and ``GET /v2/account`` concurrently."""
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
        """Poll positions and cash, reconciling the local model against REST.

        Catches ``BaseException`` because interpreter shutdown can raise outside the
        ``Exception`` hierarchy.
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
                    self._safe_log("debug", "account refresh failed", exc_info=True)
            try:
                # Wake on the periodic timer or on a nudge from submit, cancel
                # or expire_all.
                self._refresh_event.wait(timeout=self._refresh_interval_sec)
                self._refresh_event.clear()
            except BaseException:
                return

    def _reconcile_with_rest(self, positions: Any, account: Any) -> None:
        """Detect divergence from the REST snapshot and resolve it from order history.

        Positions are never dropped on the strength of ``/v2/positions`` alone, and
        cash only syncs while idle, so a lagging snapshot cannot erase a fresh fill.
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
                    if abs(rest_qty - self._positions.get(sym, 0.0)) > _QTY_EPSILON:
                        divergent = True
                        break
                if not divergent:
                    for sym, local_qty in self._positions.items():
                        if abs(local_qty) > _QTY_EPSILON and sym not in rest_by_symbol:
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

    # ── Background daemon: WebSocket trade_updates listener ──────────────

    def _trade_updates_loop(self) -> None:
        """Run the WS session in a fresh asyncio loop; reconnect on drop.

        Reconnection is decided by ``_refresh_stop``, not by whether the session
        raised.
        """
        while not self._refresh_stop.is_set():
            try:
                asyncio.run(self._trade_updates_session())
            except BaseException:  # noqa: BLE001 — daemon thread isolation
                # The session coroutine handles its own errors, so this only
                # fires if asyncio.run itself fails during shutdown.
                if self._refresh_stop.is_set():
                    return
                self._safe_log(
                    "debug",
                    "asyncio.run for trade_updates failed; reconnecting in 1s",
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

        Logging itself can fail during interpreter shutdown, and a traceback escaping
        the daemon thread is worse than a lost line.
        """
        try:
            getattr(logger, level)(msg, exc_info=exc_info)
        except BaseException:
            pass

    async def _trade_updates_session(self) -> None:
        """Single-pass WS session: auth, subscribe, dispatch updates.

        Exceptions are handled here so asyncio does not print a traceback when the
        socket drops.
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
                    # Alpaca wraps events in a stream envelope; the bare shape is
                    # tolerated for fixtures.
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if data is None:
                        data = payload
                    self._handle_trade_update(data)
        except BaseException:  # noqa: BLE001 — daemon thread isolation
            # Swallow everything, including the shutdown-time RuntimeError from
            # asyncio internals, and let the outer loop retry.
            if not self._refresh_stop.is_set():
                self._safe_log(
                    "debug",
                    "trade_updates WS session error (will reconnect)",
                    exc_info=True,
                )

    def _handle_trade_update(self, msg: dict[str, Any]) -> None:
        """Apply a single trade_updates event to the local model."""
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
        # Other events are informational: submit already booked provisionally.

    def _apply_fill_event(self, order: dict[str, Any], order_id: str) -> None:
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

        order_type = self._reverse_alpaca_type(str(order.get("type", "market")))
        tif = self._reverse_alpaca_tif(str(order.get("time_in_force", "gtc")))
        limit_price = (
            float(order["limit_price"])
            if order.get("limit_price") is not None
            else None
        )
        stop_price = (
            float(order["stop_price"]) if order.get("stop_price") is not None else None
        )

        with self._snapshot_lock:
            # Already processed this fill? (e.g., duplicate WS push or
            # reconnect REST gap-fill replaying).
            if order_id in self._reported_order_ids:
                # Upgrade a still-provisional entry to the real fill price,
                # which also makes duplicate pushes harmless.
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
                # Nothing provisional to match: the fill beat submit, or belongs
                # to another session, so apply it fresh.
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

            # Dropping pending on a partial fill is a simplification: the paper
            # engine fills market orders in one shot.
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
        order_type = self._reverse_alpaca_type(str(order.get("type", "market")))
        tif = self._reverse_alpaca_tif(str(order.get("time_in_force", "gtc")))
        limit_price = (
            float(order["limit_price"])
            if order.get("limit_price") is not None
            else None
        )
        stop_price = (
            float(order["stop_price"]) if order.get("stop_price") is not None else None
        )
        mapped = (
            OrderStatus.CANCELLED
            if event.startswith("cancel")
            else (OrderStatus.REJECTED if event == "rejected" else OrderStatus.EXPIRED)
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
        self._positions[symbol] = self._positions.get(symbol, 0.0) + qty * signed
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
                self._cost_basis.get(symbol, 0.0) + price_delta * filled_qty * signed
            )
            self._cash -= price_delta * filled_qty * signed
            self._cash_mut_ts = time.monotonic()
        prov.executed_price = real_price
        prov.market_price = real_price
        prov.slippage_cost = 0.0
        prov.status = OrderStatus.FILLED

    def _sync_orders_since_last_event(self) -> None:
        """REST gap-fill on WS reconnect."""
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
            synthetic = {
                "event": event,
                "order": entry,
                "timestamp": entry.get("updated_at"),
            }
            self._handle_trade_update(synthetic)

    # ── BaseExecutor interface — submission lifecycle ────────────────────

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
                        f"Unknown action: {intent.action!r} (expected buy/sell/hold)"
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
            # Book provisionally at the market price and still report a fill, so
            # callers keep their synchronous contract until the WS corrects it.
            with self._snapshot_lock:
                # Skip the provisional booking when the WS already reported this
                # order, which fast paper fills can do.
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
            # Leave the audit entry to the WS handler, which is idempotent, so
            # the cancel is not logged twice.
        return CancelResult(kind="cancelled", cancelled=order)

    def tick(
        self,
        prices: dict[str, float],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> TickResult:
        """No-op for AlpacaPaperExecutor."""
        return TickResult()

    def expire_all(
        self,
        prices: dict[str, float] | None = None,
        *,
        step: int = 0,
        timestamp: datetime | None = None,
    ) -> list[ExecutionReport]:
        """Cancel every locally-tracked open order on Alpaca."""
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

        # Transient pool, capped at 8 workers so the episode-end cancels do not
        # burst Alpaca's order-rate limit.
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
                    # Synthesise the expiry here; a later "canceled" push is a
                    # no-op because the handler is idempotent.
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

    # ── BaseExecutor interface — accessors (all read from local model) ───

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
        """Pure local PnL computation: ``current_price * qty - cost_basis - tx_total``."""
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
        """Cancel open orders, close all positions on Alpaca, and clear local state.

        The REST flatten logs failures rather than raising, but the local clear always
        runs.
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

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _reverse_alpaca_type(raw: str) -> str:
        return _REV_ALPACA_TYPE.get(raw, OrderType.MARKET)

    @staticmethod
    def _reverse_alpaca_tif(raw: str) -> str:
        return _REV_ALPACA_TIF.get(raw, TimeInForce.GTC)
