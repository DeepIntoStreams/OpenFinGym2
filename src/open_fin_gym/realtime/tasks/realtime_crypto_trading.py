"""Curated task: Realtime Crypto Trading via Binance API.

Paper trading on real-time BTC/USD market data.  The agent receives a fresh
price snapshot (plus recent bar history and order book) each step and
submits buy/sell/hold actions with a quantity.  Rewards are immediate
mark-to-market PnL.

No API key required -- uses the free Binance public API.

Interaction pattern (gym loop):
    obs = task.reset()                          # market + positions snapshot
    while not done:
        action = agent.act(obs)                 # {"action": "buy", "symbol": ..., "quantity": ...}
        obs, reward, done, info = task.step(action)  # executes via paper engine
    rewards = task.evaluate(actions)            # Sharpe, drawdown, PnL, etc.
"""

from typing import Any, Dict, Optional

from open_fin_gym.realtime.contracts import TaskMetadata
from open_fin_gym.realtime.config import TradingConfig
from open_fin_gym.realtime.data_providers.binance import BinanceProvider
from open_fin_gym.realtime.tasks.realtime_trading_task import RealtimeTradingTask


class RealtimeCryptoTrading(RealtimeTradingTask):
    """One-liner realtime paper trading on Binance crypto markets.

    Pre-configured with sensible defaults so agent authors do not need to
    manually wire the data provider and execution engine:

    .. code-block:: python

        task = RealtimeCryptoTrading()
        # Run a basic gym loop or use the in-container runner:
        # open_fin_gym.realtime.agent_runtime
        # .run_realtime_trading_trial

    Args:
        config: Recognised keys (all optional, with defaults):

            - ``"symbols"``: list of trading pairs (default ``["BTCUSDT"]``)
            - ``"slippage_pct"``: slippage as a fraction (default ``0.001``)
            - ``"transaction_cost_pct"``: per-trade cost fraction (default ``0.0``)
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
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        provider: Optional[Any] = None,
    ) -> None:
        config = config or {}
        symbols = config.get("symbols", ["BTCUSDT"])
        max_steps = int(config.get("max_steps", 0))
        context_resolutions = config.get(
            "context_resolutions",
            [{"interval": "1m", "bars": 60}],
        )
        data_resolution = config.get("data_resolution")
        target_symbols = config.get("target_symbols")
        initial_capital = float(config.get("initial_capital", 100000.0))

        # execution_mode is intentionally not config-exposed for crypto:
        # Binance has no paper-trading REST API analogous to Alpaca's
        # paper-api.alpaca.markets, so "internal_paper" (in-process
        # SimulatedExecutor against the live Binance feed) is the only
        # mode this task can offer. Stock tasks (RealtimeStockTrading)
        # DO expose this knob because Alpaca supports both modes.
        trading_config = TradingConfig(
            slippage_pct=float(config.get("slippage_pct", 0.001)),
            transaction_cost_pct=float(config.get("transaction_cost_pct", 0.0)),
            execution_mode="internal_paper",
        )
        if provider is None:
            provider = BinanceProvider()

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
            task_id=f"realtime_crypto_trading_{sym_label}",
            title=f"Realtime Crypto Trading ({', '.join(self._symbols)}, Binance)",
            description=(
                "Realtime paper trading on Binance crypto markets. Agent submits "
                "buy/sell/hold actions with quantities; rewards are immediate "
                "mark-to-market PnL. Uses the free Binance public API."
            ),
            interaction_model=base.interaction_model,
            task_type=base.task_type,
            data_requirements=base.data_requirements,
            tags=["crypto", "trading", "realtime", "binance"],
            difficulty=base.difficulty,
            version="1.0.0",
        )
