"""Resolve pending predictions against realised outcomes and score them.

Deferred-scoring tasks record predictions and stop; this fills in the ground
truth once each horizon closes and turns the ledger rows into rewards.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from open_fin_gym.realtime.data_providers.base import (
    EventDataProvider,
    interval_to_seconds,
)
from open_fin_gym.realtime.ledger import PredictionLedger
from open_fin_gym.realtime.rewards import ALL_EVENT_REWARDS, EventReward
from open_fin_gym.realtime.rewards.reward_bank import (
    ALL_TRADING_REWARDS,
    TradingReward,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 15.0
# Bars close on the venue's clock, so allow a little slack before reading back.
_GRACE_SEC = 5.0


def _as_dict(raw: Any) -> dict[str, Any]:
    """Return a ground-truth column as a dict, whether stored as JSON or not."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_ts(raw: Any) -> datetime:
    """Parse a stored timestamp into an aware UTC datetime."""
    moment = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def resolve_pending(ledger: PredictionLedger, provider: Any) -> dict[str, int]:
    """Fill in ground truth for every prediction already past its horizon.

    Event markets that have not settled yet stay pending rather than failing.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_GRACE_SEC)
    rows = ledger.get_pending(before=cutoff)
    is_event = isinstance(provider, EventDataProvider)
    resolved = failed = waiting = 0
    for row in rows:
        symbol = row["symbol"]
        try:
            if is_event:
                outcome = provider.get_event_outcome(symbol)
                if outcome is None:
                    waiting += 1
                    continue
                ledger.mark_resolved(
                    row["id"],
                    exit_price=None,
                    actual_return=None,
                    ground_truth={"outcome": float(outcome)},
                )
            else:
                # resolve_at is the target bar's close, so read the bar that
                # opens one interval earlier and take its close.
                interval = row.get("resolution_interval") or "1m"
                at = _parse_ts(row["resolve_at"]) - timedelta(
                    seconds=interval_to_seconds(interval)
                )
                exit_price = float(provider.get_price_at(symbol, at, interval).price)
                entry = float(row["entry_price"] or 0.0)
                actual_return = (exit_price - entry) / entry if entry else 0.0
                ledger.mark_resolved(
                    row["id"],
                    exit_price=exit_price,
                    actual_return=actual_return,
                    ground_truth={
                        "exit_price": exit_price,
                        "actual_return": actual_return,
                    },
                )
            resolved += 1
        except Exception as exc:
            logger.warning("Could not resolve %s: %s", row["id"][:8], exc)
            ledger.mark_failed(row["id"], str(exc))
            failed += 1
    return {"resolved": resolved, "failed": failed, "waiting": waiting}


def score_resolved(
    ledger: PredictionLedger,
    *,
    event_rewards: list[type[EventReward]] | None = None,
    trading_rewards: list[type[TradingReward]] | None = None,
) -> dict[str, float]:
    """Score every resolved-but-unscored row and write the rewards back.

    Event rows are scored on probability against outcome, price rows on the
    predicted against the realised price.
    """
    rows = ledger.get_resolved_unscored(limit=1000)
    if not rows:
        return {"n_scored": 0.0}

    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []
    dropped = 0
    is_event = "outcome" in _as_dict(rows[0].get("ground_truth"))
    for row in rows:
        gt = _as_dict(row.get("ground_truth"))
        if is_event:
            outcome = gt.get("outcome")
            if outcome is None or abs(float(outcome) - 0.5) < 1e-9:
                dropped += 1
                continue
            predictions.append(
                {"predicted_yes_probability": row.get("predicted_price")}
            )
            ground_truths.append({"outcome": float(outcome)})
        else:
            predictions.append(
                {
                    "predicted_price": row.get("predicted_price"),
                    "direction": row.get("direction"),
                }
            )
            ground_truths.append(
                {
                    "entry_price": row.get("entry_price"),
                    "exit_price": gt.get("exit_price"),
                    "actual_return": gt.get("actual_return"),
                }
            )

    classes: list[Any] = [
        cls()
        for cls in (
            (event_rewards or ALL_EVENT_REWARDS)
            if is_event
            else (trading_rewards or ALL_TRADING_REWARDS)
        )
    ]
    rewards: dict[str, float] = {}
    for reward in classes:
        try:
            rewards[reward.name] = float(
                reward.compute_aggregate(predictions, ground_truths)
            )
        except Exception as exc:
            logger.warning("Reward %s failed: %s", reward.name, exc)
            rewards[reward.name] = math.nan
    for row in rows:
        ledger.mark_scored(row["id"], rewards)

    out = dict(rewards)
    out["n_scored"] = float(len(predictions))
    if dropped:
        out["n_dropped_ambiguous"] = float(dropped)
    return out


def resolve_and_score(
    ledger: PredictionLedger,
    provider: Any,
    *,
    deadline_sec: float,
    poll_interval_sec: float = _POLL_INTERVAL_SEC,
) -> dict[str, float]:
    """Wait for the recorded horizons to close, then score what resolved.

    Gives up at ``deadline_sec`` and reports whatever is still pending, so a
    trial always returns a result rather than hanging.
    """
    started = time.monotonic()
    while True:
        counts = resolve_pending(ledger, provider)
        summary = ledger.get_summary()
        pending = float(summary.get("pending", 0) or 0)
        if pending <= 0 or time.monotonic() - started >= deadline_sec:
            break
        time.sleep(min(poll_interval_sec, max(1.0, deadline_sec - (time.monotonic() - started))))

    out = score_resolved(ledger)
    out["n_pending"] = float(ledger.get_summary().get("pending", 0) or 0)
    out["n_failed"] = float(counts.get("failed", 0))
    return out
