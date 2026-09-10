"""Curated task: Realtime Polymarket Event Forecasting.

Probability forecasting over active Polymarket binary-event markets.
At task setup the verifier calls Polymarket's Gamma API to discover
markets matching the bundle's ``[curated.default_config.discovery]``
filters (resolution window, volume, price-extremes, categories) and
freezes that universe for the trial. The in-container agent fetches
the universe via ``GET /markets/active``, submits one
``predicted_yes_probability`` per market via
``POST /submit/event_predictions_async``, and is scored after every
market in the session has resolved.

No API key required — uses the free Polymarket Gamma + CLOB read
endpoints.

Interaction pattern (single-shot batch, NOT a streaming gym loop):
    obs = task.reset()                          # full market universe
    action = agent.act(obs)                     # list[{symbol, predicted_yes_probability}]
    obs, reward, done, info = task.step(action) # done=True immediately
    rewards = task.evaluate(actions)            # {"status_deferred": 1.0, ...}
"""

from typing import Any, Dict, Optional

from open_fin_gym.realtime.tasks.realtime_polymarket_task import (
    RealtimePolymarketTask,
)


class RealtimePolymarket(RealtimePolymarketTask):
    """One-liner realtime event forecasting on Polymarket.

    Pre-configured with sensible defaults so agent authors do not need
    to manually wire the data provider and prediction ledger. All
    behaviour is inherited from :class:`RealtimePolymarketTask`; this
    subclass exists only as the curated-bundle entry point named in
    ``task.toml [curated] class_name``.

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
