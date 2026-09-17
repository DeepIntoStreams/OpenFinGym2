"""Curated task: Offline Stock Forecasting (Alpaca hourly OHLCV)."""

from open_fin_gym.realtime.data_providers.alpaca import AlpacaProvider
from open_fin_gym.realtime.offline_forecasting import _OfflineForecastingTask


class OfflineStockForecasting(_OfflineForecastingTask):
    """Multi-symbol historical Alpaca forecasting task. See the base class for
    the feature set, split rules and config keys (the ``provider`` kwarg
    overrides the default :class:`AlpacaProvider`)."""

    _CACHE_SUBDIR = "alpaca_stock_hourly_ohlcv"
    _DEFAULT_SYMBOLS = ("SPY",)
    _SOURCE_LABEL = "Alpaca"
    _ASSET_TAG = "stock"
    _TASK_ID_PREFIX = "offline_stock_forecasting"
    _TITLE = "Offline Stock Forecasting"

    def _make_default_provider(self):
        return AlpacaProvider()
