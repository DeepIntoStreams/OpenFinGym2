"""RealtimePolymarketTask — event-resolution probability forecasting.

This task lives in the curated realtime family alongside
``RealtimeForecastingTask`` / ``RealtimeTradingTask``, but its
interaction pattern is **single-shot batch**, not streaming gym loop:

1. At ``__init__`` time, the task calls
   :meth:`PolymarketProvider.discover_active_markets` with the
   ``[curated.default_config.discovery]`` filter block from
   ``task.toml``. Active markets are cached as the trial's
   prediction universe along with each market's absolute
   ``resolve_at`` timestamp (from Polymarket's ``endDateIso``).

2. The in-container agent calls ``GET /markets/active`` on the
   verifier RPC to fetch the universe payload (question text,
   description, resolution criteria, current YES price, orderbook,
   tags, historical price bars).

3. The agent submits one batch of probability predictions via
   ``POST /submit/event_predictions_async`` — one
   ``predicted_yes_probability ∈ [0, 1]`` per market. The verifier
   captures ``entry_price`` server-side (= current YES price) and
   writes per-prediction rows to the ``PredictionLedger`` with each
   row's ``resolve_at`` set to that market's absolute resolution
   timestamp (NOT a uniform horizon).

4. After all markets in the session have resolved, the user runs
   ``python -m openfinai_harbor.resolve_deferred <trial_dir>`` to
   fetch event outcomes via
   :meth:`PolymarketProvider.get_event_outcome` and compute the
   ``EventReward`` metrics (Brier / log-loss / ECE). Predictions
   on UMA-disputed (``outcome == 0.5``) markets are dropped from
   scoring upstream in the resolver.

The gym-loop methods (``reset`` / ``step``) are minimal because the
agent does not interact with the task object directly — all
interaction goes through the verifier RPC. ``step()`` is a no-op
that immediately signals ``done=True``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from open_fin_gym.realtime.contracts import BaseTask, TaskMetadata
from open_fin_gym.realtime.data_providers.base import EventDataProvider
from open_fin_gym.realtime.data_providers.polymarket import PolymarketProvider
from open_fin_gym.realtime.ledger import PredictionLedger

logger = logging.getLogger(__name__)


_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
_DEFAULT_DB = _RESULTS_DIR / "polymarket_predictions.db"


# Default discovery filters — kept in sync with the curated bundle's
# task.toml [curated.default_config.discovery] block. Operators
# override per-bundle; these are the fallbacks if the task is
# instantiated programmatically without a config.
_DEFAULT_DISCOVERY: dict[str, Any] = {
    "resolution_window_hours_min": 1,
    "resolution_window_hours_max": 48,
    "min_yes_price": 0.01,
    "max_yes_price": 0.99,
    "min_24h_volume_usd": 1000.0,
    "min_orderbook_depth_usd": 100.0,
    "max_markets_per_trial": 20,
    "categories": [],
    "exclude_disputed": True,
    "binary_only": True,
}


class RealtimePolymarketTask(BaseTask):
    """Discover-once / batch-submit / deferred-score polymarket forecasting.

    Args:
        config: See ``task.toml [curated.default_config]`` for the canonical
            keys:

            - ``discovery``: dict of filter values passed to
              :meth:`EventDataProvider.discover_active_markets`. See
              ``_DEFAULT_DISCOVERY`` for defaults.
            - ``db_path``: SQLite ledger path (default
              ``results/polymarket_predictions.db``).
            - ``headline_metric``: which reward becomes the trial's
              ``reward.json`` headline (default ``"brier_score"``).
        provider: Inject a custom provider — primarily for testing.
            Defaults to :class:`PolymarketProvider`.
        ledger: Inject a custom ledger — primarily for testing. Defaults
            to a SQLite ledger at ``config["db_path"]``.
        skip_discovery: When True, do NOT call
            ``provider.discover_active_markets`` at construction. Used for
            tests that pre-seed the universe via constructor injection.
            Default False.
        markets: Pre-seeded universe payloads (only consulted when
            ``skip_discovery=True``). Each dict must have the shape
            :meth:`PolymarketProvider._market_to_snapshot_payload` returns.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[EventDataProvider] = None,
        ledger: Optional[PredictionLedger] = None,
        skip_discovery: bool = False,
        markets: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(config=config)
        cfg = self.config

        discovery = dict(_DEFAULT_DISCOVERY)
        discovery.update(cfg.get("discovery") or {})

        db_path = str(cfg.get("db_path", str(_DEFAULT_DB)))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._headline_metric = str(cfg.get("headline_metric", "brier_score"))
        self._discovery_filters: dict[str, Any] = discovery

        if provider is None:
            provider = PolymarketProvider()
        if ledger is None:
            ledger = PredictionLedger(db_path)

        if not isinstance(provider, EventDataProvider):
            raise TypeError(
                f"RealtimePolymarketTask requires an EventDataProvider; "
                f"got {type(provider).__name__}"
            )

        self._provider: EventDataProvider = provider
        self._ledger = ledger

        if skip_discovery:
            universe = list(markets or [])
        else:
            try:
                universe = provider.discover_active_markets(discovery)
            except Exception as exc:
                logger.exception("polymarket discovery failed: %s", exc)
                universe = []

        # Per-symbol indexes for the handler. _target_symbols mirrors
        # realtime_forecasting so "symbol in target set" validation
        # works uniformly.
        self._market_metadata: dict[str, dict[str, Any]] = {}
        self._market_resolve_at: dict[str, datetime] = {}
        self._target_symbols: list[str] = []
        for m in universe:
            sym = m.get("symbol")
            if not sym:
                continue
            resolve_at = m.get("resolution_at")
            if not isinstance(resolve_at, datetime):
                # Skip markets without a parseable resolution timestamp —
                # we have no way to set the ledger's ``resolve_at`` field.
                continue
            self._target_symbols.append(sym)
            self._market_metadata[sym] = m
            self._market_resolve_at[sym] = resolve_at

        # Mirrors target_symbols for handler-compat (realtime_forecasting
        # on_startup reads both as a fallback).
        self._symbols: list[str] = list(self._target_symbols)

        logger.info(
            "RealtimePolymarketTask initialised with %d markets "
            "(filters=%s)",
            len(self._target_symbols),
            sorted(discovery.keys()),
        )

    # ── BaseTask contract ──────────────────────────────────────────────

    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            task_id="realtime_polymarket",
            title="Realtime Polymarket Event Forecasting",
            description=(
                "Probability forecasting over active Polymarket "
                "binary-event markets. Agent fetches a universe of "
                "markets (question, description, resolution criteria, "
                "current YES price, orderbook, historical prices), "
                "submits one probability per market, and is scored "
                f"after each market resolves. Headline metric: "
                f"{self._headline_metric}. Uses the free Polymarket "
                "Gamma + CLOB APIs (no key required)."
            ),
            task_type="realtime",
            interaction_model="forecasting",
            tags=["polymarket", "event_prediction", "forecasting", "realtime"],
            difficulty="medium",
            version="1.0.0",
        )

    def load_data(self) -> Any:
        """No static dataset — the universe was materialised at __init__.

        Returns the cached metadata dict to satisfy callers that expect
        ``load_data()`` to return *something*. The verifier RPC exposes
        the same payload via ``GET /markets/active``.
        """
        if self._data is None:
            self._data = {"markets": list(self._market_metadata.values())}
        return self._data

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "type": "list",
            "items": {
                "symbol": "str (condition_id, 0x-prefixed hash)",
                "question": "str",
                "description": "str",
                "resolution_at": "str (ISO-8601 UTC)",
                "current_yes_price": "float in [0, 1]",
                "best_bid": "float | None",
                "best_ask": "float | None",
                "spread": "float | None",
                "volume_24h": "float | None",
                "liquidity": "float | None",
                "tags": "list[str]",
                "categories": "list[str]",
                "historical_prices": "list[{t: int, p: float}] (CLOB bars)",
            },
        }

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "list",
            "items": {
                "symbol": "str (must be in target_symbols)",
                "predicted_yes_probability": "float in [0, 1]",
            },
            "submission": (
                "One batch per trial via "
                "POST /submit/event_predictions_async"
            ),
        }

    def reset(self) -> Any:
        """Return the current market universe as the initial observation."""
        return self.load_data()

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """No-op: polymarket is single-shot batch via HTTP, not gym loop."""
        return (self.load_data(), 0.0, True, {"single_shot": True})

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Return a deferred placeholder.

        Actual scoring happens out-of-band in the resolver after
        markets resolve. Returns a marker so harbor's per-trial
        scoring path sees a non-empty rewards dict (the resolver
        overwrites it later).
        """
        return {"status_deferred": 1.0, "reward": 0.0}
