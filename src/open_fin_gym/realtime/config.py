from pydantic import BaseModel


class TradingConfig(BaseModel):
    """Configuration for paper trading simulation.

    Used by :class:`RealtimeTradingTask` and optionally by offline
    :class:`TradingTask` subclasses that want cost modeling. Per-task
    knobs are supplied via each curated bundle's ``task.toml`` under
    ``[curated.default_config]``; this class is not bound to any
    AppConfig field.
    """

    slippage_pct: float = 0.001  # 0.1% = 10 basis points
    transaction_cost_pct: float = 0.0  # percentage of notional per trade
    execution_mode: str = "internal_paper"  # "internal_paper" | "alpaca_paper"
    # True => AlpacaPaperExecutor closes residual positions + cancels open
    # orders at construction. False => raises on dirty account so PnL can't
    # be contaminated by prior-session carry. SimulatedExecutor ignores this.
    flatten_on_start: bool = False
