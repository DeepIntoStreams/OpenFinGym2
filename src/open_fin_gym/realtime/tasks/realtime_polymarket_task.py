"""RealtimePolymarketTask -- event-resolution probability forecasting.

Single-shot rather than a gym loop: the market universe is frozen at
construction and each market carries its own resolution time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from open_fin_gym.realtime.contracts import BaseTask, TaskMetadata
from open_fin_gym.realtime.data_providers.base import EventDataProvider
from open_fin_gym.realtime.data_providers.polymarket import (
    PolymarketProvider,
)
from open_fin_gym.realtime.ledger import PredictionLedger

logger = logging.getLogger(__name__)


_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
_DEFAULT_DB = _RESULTS_DIR / "polymarket_predictions.db"


# Fallback discovery filters for programmatic use; bundles override them from
# their own config.
_DEFAULT_DISCOVERY: dict[str, Any] = {
    "resolution_window_hours_min": 0.05,
    "resolution_window_hours_max": 1.0,
    "min_yes_price": 0.01,
    "max_yes_price": 0.99,
    # Order-book liquidity, not 24h volume: a market opening minutes before it
    # settles has no history, and those are the ones a trial can score.
    "min_liquidity": 2000.0,
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
        # Filled in by step(); evaluate() reports what was recorded.
        self._submitted: dict[str, Any] = {}

        if skip_discovery:
            universe = list(markets or [])
        else:
            try:
                universe = provider.discover_active_markets(discovery)
            except Exception as exc:
                logger.exception("polymarket discovery failed: %s", exc)
                universe = []

        # _target_symbols mirrors realtime_forecasting, so membership checks
        # work the same way.
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
        """No static dataset — the universe was materialised at __init__."""
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

    def _ledger_row(
        self,
        symbol: str,
        probability: float,
        session_id: str,
        submitted_at: datetime,
    ) -> dict[str, Any]:
        """Shape one YES-probability prediction into a pending ledger row."""
        resolve_at = self._market_resolve_at[symbol]
        meta = self._market_metadata.get(symbol, {})
        end_date = meta.get("resolution_at")
        return {
            "session_id": session_id,
            "provider": self._provider.name,
            "symbol": symbol,
            # Events carry an absolute resolve_at, but the schema requires
            # horizon_minutes; record the delta for context.
            "horizon_minutes": max(
                0, int((resolve_at - submitted_at).total_seconds() // 60)
            ),
            "submitted_at": submitted_at,
            "resolve_at": resolve_at,
            # Always "long YES": the number the agent gives is the YES-side
            # probability.
            "direction": "long",
            "predicted_price": probability,
            # Read server-side, so a prediction cannot be scored against a
            # price the agent invented.
            "entry_price": float(self._provider.get_current_price(symbol).price),
            "snapshot": {
                "event_prediction": True,
                "predicted_yes_probability": probability,
                "question": meta.get("question"),
                "categories": list(meta.get("categories") or []),
                "end_date_iso": (
                    end_date.isoformat() if isinstance(end_date, datetime) else None
                ),
            },
        }

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """Record one batch of YES probabilities for deferred scoring.

        Args:
            action: ``{"predictions": [...]}`` or a bare list; ``done`` is always True.
        """
        predictions = action.get("predictions") if isinstance(action, dict) else action
        if not isinstance(predictions, list):
            raise ValueError(
                "action must be a list of predictions or carry one under "
                f"'predictions'; got {type(predictions).__name__}"
            )

        session_id = str(uuid4())
        submitted_at = datetime.now(timezone.utc)
        rows = []
        for pred in predictions:
            symbol = pred.get("symbol")
            if symbol not in self._market_resolve_at:
                raise ValueError(
                    f"symbol {symbol!r} is not in this trial's discovered "
                    f"universe ({len(self._target_symbols)} markets)"
                )
            rows.append(
                self._ledger_row(
                    symbol,
                    float(pred["predicted_yes_probability"]),
                    session_id,
                    submitted_at,
                )
            )

        pred_ids = [self._ledger.submit(row) for row in rows]
        resolve_ats = [row["resolve_at"] for row in rows]
        self._submitted = {
            "session_id": session_id,
            "n_predictions": len(pred_ids),
            "prediction_ids": pred_ids,
            "resolve_at_first": min(resolve_ats).isoformat() if resolve_ats else None,
            "resolve_at_last": max(resolve_ats).isoformat() if resolve_ats else None,
        }
        return (self.load_data(), 0.0, True, {"single_shot": True, **self._submitted})

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Report how many predictions were recorded; /score resolves them."""
        return {
            "status_deferred": 1.0,
            "reward": 0.0,
            "n_predictions": float(self._submitted.get("n_predictions", 0)),
        }
