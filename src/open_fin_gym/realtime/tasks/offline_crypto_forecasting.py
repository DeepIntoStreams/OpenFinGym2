"""Curated task: Offline Crypto Forecasting (Binance hourly OHLCV)."""

from open_fin_gym.realtime.data_providers.binance import BinanceProvider
from open_fin_gym.realtime.offline_forecasting import _OfflineForecastingTask


class OfflineCryptoForecasting(_OfflineForecastingTask):
    """Multi-symbol historical Binance forecasting task. See the base class for
    the feature set, split rules and config keys (the ``provider`` kwarg
    overrides the default :class:`BinanceProvider`)."""

    _CACHE_SUBDIR = "binance_crypto_hourly_ohlcv"
    _DEFAULT_SYMBOLS = ("BTCUSDT",)
    _SOURCE_LABEL = "Binance"
    _ASSET_TAG = "crypto"
    _TASK_ID_PREFIX = "offline_crypto_forecasting"
    _TITLE = "Offline Crypto Forecasting"

    def _make_default_provider(self):
        return BinanceProvider()
