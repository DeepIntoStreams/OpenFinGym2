"""Curated task: Realtime Stock Trading via Alpaca API.

Paper trading on real-time US equity market data.  The agent receives a
fresh price snapshot (plus recent bar history and NBBO quotes) each
step and submits buy/sell/hold actions with a quantity.  Rewards are
immediate mark-to-market PnL.

Requires Alpaca API keys (free tier is sufficient).  Set the
``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` environment variables.

Supports two execution modes:
  - ``"internal_paper"`` (default): in-process paper trading engine
    (SimulatedExecutor) with configurable slippage and transaction
    costs. Live prices come from Alpaca; orders never leave the agent
    container.
  - ``"alpaca_paper"``: submits real orders to Alpaca's paper-trading
    environment at ``paper-api.alpaca.markets``. Position state and
    realized PnL come from Alpaca's account.

Interaction pattern (gym loop)::

    obs = task.reset()                          # market + positions snapshot
    while not done:
        action = agent.act(obs)                 # {"action": "buy", "symbol": ..., "quantity": ...}
        obs, reward, done, info = task.step(action)  # executes via engine
    rewards = task.evaluate(actions)            # Sharpe, drawdown, PnL, etc.
"""

from typing import Any, Dict, Optional

from open_fin_gym.realtime.config import TradingConfig
from open_fin_gym.realtime.contracts import TaskMetadata
from open_fin_gym.realtime.data_providers.alpaca import AlpacaProvider
from open_fin_gym.realtime.tasks.realtime_trading_task import (
    RealtimeTradingTask,
)


class RealtimeStockTrading(RealtimeTradingTask):
    """One-liner realtime paper trading on US equities via Alpaca.

    Pre-configured with sensible defaults so agent authors do not need to
    manually wire the data provider and execution engine::

        task = RealtimeStockTrading()
        # Run a basic gym loop or use the in-container runner:
        # open_fin_gym.realtime.agent_runtime
        # .run_realtime_trading_trial

    Args:
        config: Recognised keys (all optional, with defaults):

            - ``"symbols"``: list of tickers (default ``["SPY"]``)
            - ``"slippage_pct"``: slippage as a fraction (default ``0.001``)
            - ``"transaction_cost_pct"``: per-trade cost fraction (default ``0.0``)
            - ``"execution_mode"``: ``"internal_paper"`` or ``"alpaca_paper"``
              (default ``"internal_paper"``)
            - ``"context_resolutions"``: non-empty list of
              ``{"interval": str, "bars": positive int}`` entries (default
              ``[{"interval": "1m", "bars": 60}]``). Each entry instantiates
              a :class:`MarketDataBuffer`; the ``data_resolution`` buffer
              drives latest-price lookups, mark-to-market PnL, and the
              WebSocket subscription.
            - ``"data_resolution"``: which entry's interval drives the
              primary buffer (required when ``context_resolutions`` has 2+
              entries; optional when only 1). Controls data buffering only —
              agents step at any frequency they wish.
            - ``"max_steps"``: steps per episode (default ``0`` = unlimited)
            - ``"target_symbols"``: subset of ``symbols`` whose trades count
              toward the reward-bank metrics (default ``symbols``). Trades
              on non-target symbols are still allowed (so the agent can
              hedge on context-only assets) but they do not contribute to
              PnL / Sharpe / drawdown / win-rate.
        provider: Override the default :class:`AlpacaProvider` (useful for tests).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
    ) -> None:
        config = config or {}
        symbols = config.get("symbols", ["SPY"])
        max_steps = int(config.get("max_steps", 0))
        execution_mode = config.get("execution_mode", "internal_paper")
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        initial_capital = float(config.get("initial_capital", 100000.0))

        trading_config = TradingConfig(
            slippage_pct=float(config.get("slippage_pct", 0.001)),
            transaction_cost_pct=float(config.get("transaction_cost_pct", 0.0)),
            execution_mode=execution_mode,
        )
        if provider is None:
            provider = AlpacaProvider()

        super().__init__(
            config=config,
            provider=provider,
            symbols=symbols,
            trading_config=trading_config,
            context_resolutions=context_resolutions,
            data_resolution=data_resolution,
            max_steps=max_steps,
            target_symbols=target_symbols,
            initial_capital=initial_capital,
        )

    def metadata(self) -> TaskMetadata:
        base = super().metadata()
        sym_label = "_".join(s.lower() for s in self._symbols)
        return TaskMetadata(
            task_id=f"realtime_stock_trading_{sym_label}",
            title=f"Realtime Stock Trading ({', '.join(self._symbols)}, Alpaca)",
            description=(
                "Realtime paper trading on US equities via the Alpaca data API. "
                "Agent submits buy/sell/hold actions with quantities; rewards "
                "are immediate mark-to-market PnL."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["stock", "trading", "realtime", "alpaca"],
            difficulty=base.difficulty,
            version="1.0.0",
        )
