"""Curated task: Realtime Polymarket Event Forecasting."""

from typing import Any, Dict, Optional

from open_fin_gym.realtime.tasks.realtime_polymarket_task import (
    RealtimePolymarketTask,
)


class RealtimePolymarket(RealtimePolymarketTask):
    """One-liner realtime event forecasting on Polymarket.

    Args:
        config: See ``task.toml [curated.default_config]`` for the canonical
            keys. Recognised:

            - ``discovery``: nested dict of filter values passed to
              ``PolymarketProvider.discover_active_markets`` (resolution
              window, volume, price bounds, categories, etc.).
            - ``db_path``: SQLite ledger path (default
              ``results/polymarket_predictions.db``).
            - ``headline_metric``: reward name surfaced as the trial
              ``reward.json`` headline (default ``"brier_score"``).

    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
