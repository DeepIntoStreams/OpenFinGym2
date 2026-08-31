"""Curated task: Realtime Crypto Forecasting via Binance API.

Streaming **price prediction** on real-time BTC/USD market data. The
agent submits an absolute future price (``predicted_price``) for
``horizon_bars`` ahead and direction is *derived* server-side from
``sign(predicted_price - entry_price)``. Direction-only submissions
also work but score on directional metrics only (price metrics NaN
out). Ground truth is deferred until the resolver service fetches the
exit price after the prediction horizon elapses.

No API key required -- uses the free Binance public API.

Interaction pattern (gym loop, batch_mode=False):
    obs = task.reset()                          # current market snapshot
    while not done:
        action = agent.act(obs)                 # {"symbol": ..., "predicted_price": ...}
        obs, reward, done, info = task.step(action)  # records to ledger
    rewards = task.evaluate(actions)            # {"status_deferred": 1.0, ...}
"""

from pathlib import Path
from typing import Any, Dict, Optional

from open_fin_gym.realtime.contracts import TaskMetadata
from open_fin_gym.realtime.data_providers.binance import BinanceProvider
from open_fin_gym.realtime.ledger import PredictionLedger
from open_fin_gym.realtime.tasks.base_realtime_task import RealtimeForecastingTask

_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
_DEFAULT_DB = _RESULTS_DIR / "predictions.db"


class RealtimeCryptoForecasting(RealtimeForecastingTask):
    """One-liner realtime forecasting on Binance crypto markets.

    Pre-configured with sensible defaults so agent authors do not need to
    manually wire the data provider and prediction ledger. The curated
    bundle's ``run_evaluation_curated.py`` drives the gym loop via
    :mod:`open_fin_gym.realtime.agent_runtime`.

    Args:
        config: Recognised keys (all optional, with defaults):

            - ``"symbols"``: list of trading pairs (default ``["BTCUSDT"]``)
            - ``"horizon_bars"``: prediction horizon in bars of
              ``data_resolution`` (default ``5``); the exit price is the
              close of the bar this many bars ahead of submission
            - ``"context_resolutions"``: non-empty list of
              ``{"interval": str, "bars": positive int}`` entries (default
              ``[{"interval": "1m", "bars": 60}]``). Each entry adds bar
              history at that resolution; the ``data_resolution`` entry
              drives the primary buffer + entry-price snapshots.
            - ``"data_resolution"``: which entry's interval drives the
              primary buffer (required when ``context_resolutions`` has 2+
              entries; optional when only 1). Controls data buffering only —
              agents step at any frequency they wish.
            - ``"target_symbols"``: subset of ``symbols`` whose predictions
              are scored (default ``symbols``). Predictions for non-target
              symbols are dropped at :meth:`step` and rejected at the
              deferred-submit endpoint with a 422.
            - ``"db_path"``: SQLite ledger path (default ``data/results/predictions.db``)
            - ``"headline_metric"``: which reward becomes the trial
              ``reward.json`` headline (default ``"price_mape"``). See
              :class:`RealtimeForecastingTask` for the full panel.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
        ledger: Optional[PredictionLedger] = None,
    ) -> None:
        config = config or {}
        symbols = config.get("symbols", ["BTCUSDT"])
        horizon_bars = int(config.get("horizon_bars", 5))
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        db_path = config.get("db_path", str(_DEFAULT_DB))
        headline_metric = str(config.get("headline_metric", "price_mape"))

        # Ensure results directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        if provider is None:
            provider = BinanceProvider()
        if ledger is None:
            ledger = PredictionLedger(db_path)

        super().__init__(
            config=config,
            provider=provider,
            ledger=ledger,
            symbols=symbols,
            horizon_bars=horizon_bars,
            context_resolutions=context_resolutions,
            data_resolution=data_resolution,
            target_symbols=target_symbols,
            headline_metric=headline_metric,
        )

    def metadata(self) -> TaskMetadata:
        base = super().metadata()
        sym_label = "_".join(s.lower() for s in self._symbols)
        return TaskMetadata(
            task_id=f"realtime_crypto_forecasting_{sym_label}",
            title=f"Realtime Crypto Forecasting ({', '.join(self._symbols)}, Binance)",
            description=(
                "Streaming price forecasting on Binance crypto markets. "
                "Agent submits an absolute predicted_price for the next "
                f"{self._horizon_minutes}-minute horizon; direction is "
                "derived server-side. Ground truth is resolved after the "
                "horizon elapses. Headline metric: "
                f"{self._headline_metric}. Uses the free Binance public API."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["crypto", "forecasting", "realtime", "binance"],
            difficulty=base.difficulty,
            version="2.0.0",
        )
