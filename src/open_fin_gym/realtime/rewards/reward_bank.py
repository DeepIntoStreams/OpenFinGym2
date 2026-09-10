from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class TradingReward(ABC):
    """Compute a reward from a list of (prediction, ground_truth) pairs.

    Subclasses must implement :meth:`compute_per_trade`.  The default
    :meth:`compute_aggregate` returns the mean of per-trade values;
    override it for metrics that are only meaningful in aggregate
    (e.g. Sharpe ratio).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute_per_trade(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> list[float]:
        """Return one score per (prediction, ground_truth) pair."""
        raise NotImplementedError

    def compute_aggregate(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> float:
        """Return a single aggregate score across all trades."""
        per_trade = self.compute_per_trade(predictions, ground_truths)
        if not per_trade:
            return 0.0
        return sum(per_trade) / len(per_trade)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete trading rewards
#
# Expected prediction dict keys:
#     direction        str   "long" | "short"
#     predicted_return float (optional) magnitude estimate
#     confidence       float (optional) 0-1
#
# Expected ground_truth dict keys:
#     actual_return    float realised return over the horizon
# ---------------------------------------------------------------------------


def _pnl_list(predictions: list[dict], ground_truths: list[dict]) -> list[float]:
    """Signed PnL per trade, quantity-weighted when quantity is present.

    Backward compatible: if ``quantity`` is absent the default of 1.0
    preserves the original unit-size behaviour.
    """
    out: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        sign = 1.0 if pred["direction"] == "long" else -1.0
        quantity = float(pred.get("quantity", 1.0))
        out.append(sign * gt["actual_return"] * quantity)
    return out


def _nan_safe_mean(values: list[float]) -> float:
    """Mean of non-NaN values; NaN if all are NaN or list is empty."""
    valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _derive_predicted_return(pred: dict, gt: dict) -> Optional[float]:
    """Return ``pred["predicted_return"]`` if present, else derive from
    ``(predicted_price - entry_price) / entry_price``.

    Returns ``None`` when neither path yields a value, so callers can
    decide how to handle the missing case (NaN for new metrics, 0.0
    legacy fallback for ``ReturnMAE``).
    """
    pr = pred.get("predicted_return")
    if pr is not None:
        return float(pr)
    pp = pred.get("predicted_price")
    entry = gt.get("entry_price")
    if pp is not None and entry not in (None, 0, 0.0):
        return (float(pp) - float(entry)) / float(entry)
    return None


class DirectionAccuracy(TradingReward):
    """1.0 when predicted direction matches actual, 0.0 otherwise."""

    def __init__(self) -> None:
        super().__init__("direction_accuracy")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        results: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            actual_dir = "long" if gt["actual_return"] > 0 else "short"
            results.append(1.0 if pred["direction"] == actual_dir else 0.0)
        return results


class PnL(TradingReward):
    """Signed profit-and-loss assuming a unit-size position."""

    def __init__(self) -> None:
        super().__init__("pnl")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        return _pnl_list(predictions, ground_truths)


class ReturnMAE(TradingReward):
    """Mean absolute error between predicted and actual return magnitude.

    Reads ``predicted_return`` directly when supplied; falls back to
    deriving it from ``predicted_price`` and ``entry_price`` so that
    price-only submissions also score on this metric. Legacy direction-
    only submissions (no return, no price) keep their historical 0.0
    fallback.
    """

    def __init__(self) -> None:
        super().__init__("return_mae")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for p, g in zip(predictions, ground_truths):
            pr = _derive_predicted_return(p, g)
            if pr is None:
                pr = 0.0
            out.append(abs(pr - g["actual_return"]))
        return out


class SharpeRatio(TradingReward):
    """Annualisation-free Sharpe: mean(PnL) / std(PnL)."""

    def __init__(self) -> None:
        super().__init__("sharpe_ratio")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        return _pnl_list(predictions, ground_truths)

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        pnls = self.compute_per_trade(predictions, ground_truths)
        if len(pnls) < 2:
            return 0.0
        mean_pnl = sum(pnls) / len(pnls)
        var = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(var)
        return (mean_pnl / std) if std > 0 else 0.0


class MaxDrawdown(TradingReward):
    """Worst peak-to-trough decline in cumulative PnL (negative number)."""

    def __init__(self) -> None:
        super().__init__("max_drawdown")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        return _pnl_list(predictions, ground_truths)

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        pnls = self.compute_per_trade(predictions, ground_truths)
        if not pnls:
            return 0.0
        cumulative = list(itertools.accumulate(pnls))
        peak = cumulative[0]
        max_dd = 0.0
        for val in cumulative:
            peak = max(peak, val)
            max_dd = min(max_dd, val - peak)
        return max_dd


class WinRate(TradingReward):
    """Fraction of trades with positive PnL."""

    def __init__(self) -> None:
        super().__init__("win_rate")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        return _pnl_list(predictions, ground_truths)

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        pnls = self.compute_per_trade(predictions, ground_truths)
        if not pnls:
            return 0.0
        return sum(1 for p in pnls if p > 0) / len(pnls)


class QuantityPnL(TradingReward):
    """PnL with explicit quantity * price-change, for realtime trading.

    Expected prediction dict keys (in addition to ``direction``):
        quantity       float  trade size
        action         str    "buy" | "sell" (optional, falls back to direction)

    Expected ground_truth dict keys:
        entry_price      float
        exit_price       float
        slippage_cost    float (optional, default 0)
        transaction_cost float (optional, default 0)
    """

    def __init__(self) -> None:
        super().__init__("quantity_pnl")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            qty = float(pred.get("quantity", 1.0))
            entry = float(gt.get("entry_price", 0.0))
            exit_p = float(gt.get("exit_price", 0.0))
            action = pred.get("action", pred.get("direction", "long"))
            sign = 1.0 if action in ("buy", "long") else -1.0
            pnl = sign * qty * (exit_p - entry)
            pnl -= float(gt.get("slippage_cost", 0.0))
            pnl -= float(gt.get("transaction_cost", 0.0))
            out.append(pnl)
        return out


# ---------------------------------------------------------------------------
# Price- and return-error rewards for the realtime forecasting overhaul.
#
# These classes reframe forecasting as price prediction (predicted_price ↔
# exit_price) while keeping return-shape metrics on the same data via the
# (predicted_price, entry_price) → predicted_return derivation. They all
# return NaN per-trade when the submission lacks the required fields, and
# aggregate via NaN-skip mean (or sqrt-of-mean / closed-form for series-
# level metrics) so back-compat direction-only submissions don't poison
# the panel.
#
# Expected prediction dict keys (any subset; metrics gracefully NaN out
# when their inputs are missing):
#     predicted_price  float  (primary, absolute future price)
#     direction        str    "long" | "short" (used by other classes)
#
# Expected ground_truth dict keys:
#     entry_price   float  server-captured at submit
#     exit_price    float  server-captured at resolve
#     actual_return float  (exit_price - entry_price) / entry_price
# ---------------------------------------------------------------------------


class PriceMSE(TradingReward):
    """Mean-squared error between predicted price and realised exit price.

    NaN per-trade when ``predicted_price`` or ``exit_price`` is missing;
    aggregate is NaN-skip mean.
    """

    def __init__(self) -> None:
        super().__init__("price_mse")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            out.append((float(pp) - float(ep)) ** 2)
        return out

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class PriceRMSE(TradingReward):
    """Root-mean-squared error on price.

    Per-trade returns the squared error so the aggregate can compute
    ``sqrt(mean(squared_errors))`` over the full session. The per-trade
    list is intentionally not the per-trade RMSE (which would be the
    abs error and equal to PriceMAE) — the aggregation is what makes
    this class meaningful.
    """

    def __init__(self) -> None:
        super().__init__("price_rmse")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            out.append((float(pp) - float(ep)) ** 2)
        return out

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        mean_sq = _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))
        if math.isnan(mean_sq):
            return float("nan")
        return math.sqrt(mean_sq)


class PriceMAE(TradingReward):
    """Mean-absolute error between predicted and exit price."""

    def __init__(self) -> None:
        super().__init__("price_mae")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            out.append(abs(float(pp) - float(ep)))
        return out

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class PriceMAPE(TradingReward):
    """Mean-absolute percentage error on price.

    Scale-free; the only price metric safe for cross-symbol macro
    aggregation.  NaN when ``exit_price`` is below ``eps`` (avoids divide-
    by-zero blow-up on rare zero/near-zero-priced assets).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__("price_mape")
        self._eps = eps

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                out.append(float("nan"))
                continue
            denom = abs(float(ep))
            if denom < self._eps:
                out.append(float("nan"))
                continue
            out.append(abs(float(pp) - float(ep)) / denom)
        return out

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class PriceR2(TradingReward):
    """Coefficient of determination on (predicted_price, exit_price) pairs.

    Aggregate-only metric: ``compute_per_trade`` returns NaN per element
    because R² requires the full series. ``compute_aggregate`` rebuilds
    valid pairs and applies ``1 - SS_res / SS_tot``. Returns NaN when
    fewer than 2 valid pairs exist or ground truth has zero variance
    (denominator collapse).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__("price_r2")
        self._eps = eps

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        return [float("nan")] * len(predictions)

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        pairs: list[tuple[float, float]] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                continue
            pairs.append((float(pp), float(ep)))
        if len(pairs) < 2:
            return float("nan")
        gts = [g for _, g in pairs]
        gt_mean = sum(gts) / len(gts)
        ss_tot = sum((g - gt_mean) ** 2 for g in gts)
        if ss_tot < self._eps:
            return float("nan")
        ss_res = sum((p - g) ** 2 for p, g in pairs)
        return 1.0 - ss_res / ss_tot


class PricePearson(TradingReward):
    """Pearson correlation between predicted and exit prices.

    Aggregate-only; same NaN-on-low-variance / fewer-than-2-pairs
    semantics as :class:`PriceR2`.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__("price_pearson")
        self._eps = eps

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        return [float("nan")] * len(predictions)

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        pairs: list[tuple[float, float]] = []
        for pred, gt in zip(predictions, ground_truths):
            pp = pred.get("predicted_price")
            ep = gt.get("exit_price")
            if pp is None or ep is None:
                continue
            pairs.append((float(pp), float(ep)))
        if len(pairs) < 2:
            return float("nan")
        n = len(pairs)
        mean_p = sum(p for p, _ in pairs) / n
        mean_g = sum(g for _, g in pairs) / n
        cov = sum((p - mean_p) * (g - mean_g) for p, g in pairs)
        var_p = sum((p - mean_p) ** 2 for p, _ in pairs)
        var_g = sum((g - mean_g) ** 2 for _, g in pairs)
        denom = math.sqrt(var_p * var_g)
        if denom < self._eps:
            return float("nan")
        return cov / denom


class ReturnMSE(TradingReward):
    """MSE between predicted and actual return.

    ``predicted_return`` is read directly when supplied; otherwise
    derived from ``(predicted_price - entry_price) / entry_price`` so
    price-only submissions still produce a return-error reading.
    NaN-skip aggregation keeps direction-only submissions from poisoning
    the panel.
    """

    def __init__(self) -> None:
        super().__init__("return_mse")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pr = _derive_predicted_return(pred, gt)
            ar = gt.get("actual_return")
            if pr is None or ar is None:
                out.append(float("nan"))
                continue
            out.append((pr - float(ar)) ** 2)
        return out

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        return _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))


class ReturnRMSE(TradingReward):
    """RMSE between predicted and actual return — sqrt of mean ReturnMSE."""

    def __init__(self) -> None:
        super().__init__("return_rmse")

    def compute_per_trade(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> list[float]:
        out: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            pr = _derive_predicted_return(pred, gt)
            ar = gt.get("actual_return")
            if pr is None or ar is None:
                out.append(float("nan"))
                continue
            out.append((pr - float(ar)) ** 2)
        return out

    def compute_aggregate(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> float:
        mean_sq = _nan_safe_mean(self.compute_per_trade(predictions, ground_truths))
        if math.isnan(mean_sq):
            return float("nan")
        return math.sqrt(mean_sq)


# Convenience registry of all trading reward classes
ALL_TRADING_REWARDS: list[type[TradingReward]] = [
    DirectionAccuracy,
    PnL,
    ReturnMAE,
    SharpeRatio,
    MaxDrawdown,
    WinRate,
    QuantityPnL,
    PriceMSE,
    PriceRMSE,
    PriceMAE,
    PriceMAPE,
    PriceR2,
    PricePearson,
    ReturnMSE,
    ReturnRMSE,
]


# ---------------------------------------------------------------------------
#
# Sibling base class to ``TradingReward``. Operates on probability
# predictions (∈ [0, 1]) over discrete binary outcomes (∈ {0, 1}).
# Ambiguous outcomes (0.5) are dropped upstream in
# ``ResolverService.score_resolved`` before reaching these classes.
#
# Expected prediction dict keys:
#     predicted_yes_probability  float in [0, 1]
#
# Expected ground_truth dict keys:
#     outcome  float in {0.0, 1.0}  (0.5 dropped before scoring)
#
# Same ``compute_aggregate(predictions, ground_truths)`` signature as
# TradingReward so the resolver's Phase-2 score loop is family-agnostic.
# ---------------------------------------------------------------------------


# Clamp probabilities away from {0, 1} for log-loss to avoid log(0).
_LOG_LOSS_EPS = 1e-12


class Loss:
    """Marker base for evaluation rewards.

    Subclasses implement their own ``forward`` returning a scalar — there is no
    inherited reduction or normalisation. The ``**_unused_legacy`` catch-all
    absorbs stray kwargs from legacy generated rewards so they do not crash;
    those values are intentionally not stored.
    """

    def __init__(self, **_unused_legacy: Any) -> None:
        pass


def _f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


class MSELoss(Loss):
    """Mean squared error between predicted and target."""

    def __init__(self, gt: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)

    def forward(self, pred: Any) -> float:
        return float(((_f32(pred) - self.gt) ** 2).mean())


class RMSELoss(Loss):
    """Root mean squared error."""

    def __init__(self, gt: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)

    def forward(self, pred: Any) -> float:
        return float(np.sqrt(((_f32(pred) - self.gt) ** 2).mean()))


class MAELoss(Loss):
    """Mean absolute error."""

    def __init__(self, gt: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)

    def forward(self, pred: Any) -> float:
        return float(np.abs(_f32(pred) - self.gt).mean())


class MAPELoss(Loss):
    """Mean absolute percentage error."""

    def __init__(self, gt: Any, eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)
        self.eps = eps

    def forward(self, pred: Any) -> float:
        return float(
            (np.abs(_f32(pred) - self.gt) / (np.abs(self.gt) + self.eps)).mean()
        )


class DirectionalAccuracy(Loss):
    """Fraction of times the predicted direction matches the true direction."""

    def __init__(self, gt: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)

    def forward(self, pred: Any) -> float:
        return float((np.sign(_f32(pred)) == np.sign(self.gt)).astype(np.float32).mean())


class PearsonCorrelation(Loss):
    """Pearson r between predicted and true vectors, in [-1, 1]."""

    def __init__(self, gt: Any, eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)
        self.eps = eps

    def forward(self, pred: Any) -> float:
        x = _f32(pred).flatten()
        y = self.gt.flatten()
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = np.sqrt((x_c**2).sum() * (y_c**2).sum()) + self.eps
        return float((x_c * y_c).sum() / denom)


class R2Score(Loss):
    """Coefficient of determination, 1 - SS_res / SS_tot, in (-inf, 1]."""

    def __init__(self, gt: Any, eps: float = 1e-8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gt = _f32(gt)
        self.eps = eps

    def forward(self, pred: Any) -> float:
        ss_res = ((_f32(pred) - self.gt) ** 2).sum()
        ss_tot = ((self.gt - self.gt.mean()) ** 2).sum()
        return float(1.0 - ss_res / (ss_tot + self.eps))
