from pydantic import BaseModel


class TradingConfig(BaseModel):
    """Configuration for paper trading simulation."""

    slippage_pct: float = 0.001  # 0.1% = 10 basis points
    transaction_cost_pct: float = 0.0  # percentage of notional per trade
    execution_mode: str = "internal_paper"  # "internal_paper" | "alpaca_paper"
    # True closes residual positions at construction; False raises instead, so
    # a prior session cannot contaminate PnL.
    flatten_on_start: bool = False
