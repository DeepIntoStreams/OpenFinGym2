"""Curated task: Offline Stock Trading (Alpaca hourly replay)."""

from open_fin_gym.realtime.data_providers.alpaca import AlpacaProvider
from open_fin_gym.realtime.offline_trading import _OfflineTradingTask


class OfflineStockTrading(_OfflineTradingTask):
    """Multi-symbol historical Alpaca trading task. See base class for the
    observation/action contract and config keys (``provider`` kwarg
    overrides the default :class:`AlpacaProvider`)."""

    _CACHE_SUBDIR = "alpaca_stock_hourly_ohlcv"
    _DEFAULT_SYMBOLS = ("SPY",)
    _SOURCE_LABEL = "Alpaca"
    _ASSET_TAG = "stock"
    _TASK_ID_PREFIX = "offline_stock_trading"
    _TITLE = "Offline Stock Trading"
    _VERSION = "1.0.0"

    def _make_default_provider(self):
        return AlpacaProvider()
