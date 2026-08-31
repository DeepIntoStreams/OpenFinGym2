"""Curated task: Offline Stock Trading (Alpaca hourly replay).

Multi-symbol sequential trading on historical Alpaca bars. Data is fetched
on first use via ``AlpacaProvider.get_bars()`` and cached as CSV under
``data/pipeline_output/datasets/alpaca_stock_hourly_ohlcv/``. Requires
Alpaca API keys (free IEX tier) for the first fetch; cached runs work
offline.

All replay machinery lives in
:class:`~open_fin_gym.realtime.offline_trading._OfflineTradingTask`
(shared with :class:`OfflineCryptoTrading`); this class only pins the
Alpaca provider, cache directory, default symbol, and metadata.

Interaction pattern (gym loop)::

    obs = task.reset()
    while not done:
        action = agent.act(obs)   # {"action": "buy", "symbol": ..., "quantity": ...}
        obs, reward, done, info = task.step(action)
    rewards = task.evaluate(actions)

Note: ``start`` / ``end`` are ISO dates; date-only strings default to
00:00 UTC, which is post-market EST — for minute intervals pass full ISO
(e.g. ``"2020-01-02T14:30:00Z"`` for US market open).
"""

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
