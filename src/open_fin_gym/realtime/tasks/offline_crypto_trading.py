"""Curated task: Offline Crypto Trading (Binance hourly replay)."""

from open_fin_gym.realtime.offline_trading import _OfflineTradingTask
from open_fin_gym.realtime.data_providers.binance import BinanceProvider


class OfflineCryptoTrading(_OfflineTradingTask):
    """Multi-symbol historical Binance trading task. See base class for the
    observation/action contract and config keys (``provider`` kwarg
    overrides the default :class:`BinanceProvider`)."""

    _CACHE_SUBDIR = "binance_crypto_hourly_ohlcv"
    _DEFAULT_SYMBOLS = ("BTCUSDT",)
    _SOURCE_LABEL = "Binance"
    _ASSET_TAG = "crypto"
    _TASK_ID_PREFIX = "offline_crypto_trading"
    _TITLE = "Offline Crypto Trading"
    _VERSION = "1.1.0"

    def _make_default_provider(self):
        return BinanceProvider()
