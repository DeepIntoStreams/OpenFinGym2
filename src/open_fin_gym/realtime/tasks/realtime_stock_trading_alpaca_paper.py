"""Curated task: Realtime Stock Trading via Alpaca's paper-trading API.

This is the **real paper trading** variant: orders are submitted to
Alpaca's paper-trading environment at ``paper-api.alpaca.markets``,
filled against Alpaca's simulated order book, and PnL is read back from
the paper account's positions. Unlike :class:`RealtimeStockTrading` (which
defaults to the in-house :class:`SimulatedExecutor`), this task forces
``execution_mode="alpaca_paper"`` so the interaction goes through
:class:`AlpacaPaperExecutor`.

Use this task when you want to:
  - Validate strategies against Alpaca's actual paper-trading fill logic
    (real market hours, real slippage, real rejection behaviour).
  - Keep a persistent paper portfolio on the Alpaca dashboard between
    runs.
  - Prepare for a switch to real capital (real-money trading) by exercising
    the exact same REST order path.

Use :class:`RealtimeStockTrading` (in-house simulated mode) when you want
deterministic, offline-friendly execution with configurable slippage /
transaction costs.

Requires Alpaca API keys in the environment::

    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...

Interaction pattern (identical to :class:`RealtimeStockTrading`)::

    obs = task.reset()                          # market + account snapshot
    while not done:
        action = agent.act(obs)                 # {"action": "buy", "symbol": ..., "quantity": ...}
        obs, reward, done, info = task.step(action)
    rewards = task.evaluate(actions)
"""

from typing import Any, Dict, Optional

from open_fin_gym.realtime.config import TradingConfig
from open_fin_gym.realtime.contracts import TaskMetadata
from open_fin_gym.realtime.data_providers.alpaca import AlpacaProvider
from open_fin_gym.realtime.tasks.realtime_trading_task import (
    RealtimeTradingTask,
)


class RealtimeStockTradingAlpacaPaper(RealtimeTradingTask):
    """Realtime paper trading via Alpaca's paper-trading REST API.

    Pre-configured with ``execution_mode="alpaca_paper"``; any attempt
    to override that in ``config`` is ignored on purpose — use
    :class:`RealtimeStockTrading` if you want the in-house simulator.

    Args:
        config: Recognised keys (all optional, with defaults):

            - ``"symbols"``: list of tickers (default ``["SPY"]``)
            - ``"context_resolutions"``: non-empty list of
              ``{"interval": str, "bars": positive int}`` entries (default
              ``[{"interval": "1m", "bars": 60}]``). Each entry instantiates
              a :class:`MarketDataBuffer`; the ``data_resolution`` buffer
              drives latest-price lookups + dataset shipping.
            - ``"data_resolution"``: which entry's interval drives the
              primary buffer (required when ``context_resolutions`` has 2+
              entries; optional when only 1). Controls data buffering only —
              agents step at any frequency they wish.
            - ``"max_steps"``: steps per episode (default ``0`` = unlimited)
            - ``"target_symbols"``: subset of ``symbols`` whose trades count
              toward the reward-bank metrics (default ``symbols``). Trades
              on non-target symbols still hit Alpaca's paper account (so
              the audit log + dashboard reflect the full agent activity)
              but do not contribute to PnL / Sharpe / drawdown / win-rate.

            ``slippage_pct`` and ``transaction_cost_pct`` are fixed at ``0``
            because Alpaca's paper engine handles fill pricing and
            commission internally; attempting to override them has no
            effect on the real fills.
        provider: Override the default :class:`AlpacaProvider` (useful for
            tests). Must have ``provider.name == "alpaca"`` or the executor
            factory will raise ``ValueError``.
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
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        # initial_capital is forwarded for API symmetry but the
        # AlpacaPaperExecutor reads cash from /v2/account, not from a
        # local pool, so this knob is informational here.
        initial_capital = float(config.get("initial_capital", 100000.0))

        trading_config = TradingConfig(
            slippage_pct=0.0,
            transaction_cost_pct=0.0,
            execution_mode="alpaca_paper",
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
            task_id=f"realtime_stock_trading_alpaca_paper_{sym_label}",
            title=(
                f"Realtime Stock Trading via Alpaca Paper API "
                f"({', '.join(self._symbols)})"
            ),
            description=(
                "Realtime paper trading on US equities. Orders are submitted to "
                "Alpaca's paper-trading environment at paper-api.alpaca.markets "
                "and filled against Alpaca's simulated order book; PnL is read "
                "from account positions. Requires ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["stock", "trading", "realtime", "alpaca", "alpaca_paper"],
            difficulty=base.difficulty,
            version="1.0.0",
        )
