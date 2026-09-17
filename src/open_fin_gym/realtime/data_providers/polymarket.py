"""Polymarket event-market provider built on the official SDK.

Symbols are market condition ids; prices are YES-side probabilities.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from polymarket import PublicClient
from polymarket.errors import PolymarketError

from open_fin_gym.realtime.data_providers.base import (
    MarketSnapshot,
    interval_to_seconds,
)

logger = logging.getLogger(__name__)

# Discovery pages through markets lazily and stops once the caller's cap
# is met, so a large page size only means fewer round trips.
_DISCOVERY_PAGE_SIZE = 500


class PolymarketProvider:
    """Market data and event outcomes from the Polymarket public API."""

    name = "polymarket"

    def __init__(self, client: PublicClient | None = None) -> None:
        self._client = client or PublicClient()
        self._market_cache: dict[str, dict[str, Any]] = {}
        self._clob_token_cache: dict[str, str] = {}

    # ── Market record helpers ──────────────────────────────────────────

    @staticmethod
    def _to_payload(market: Any) -> dict[str, Any]:
        """Flatten an SDK market model into the payload shape tasks consume."""
        yes, no = market.outcomes.yes, market.outcomes.no
        state, prices, metrics = market.state, market.prices, market.metrics
        outcome_prices = [_f(yes.price), _f(no.price)]
        current_price = outcome_prices[0]
        return {
            "symbol": str(market.condition_id or ""),
            "question": str(market.question or ""),
            "description": str(market.description or ""),
            "resolution_at": state.end_date,
            "current_price": float("nan") if current_price is None else current_price,
            "outcomes": [yes.label, no.label],
            "outcome_prices": outcome_prices,
            "clob_token_ids": [str(yes.token_id), str(no.token_id)],
            "best_bid": _f(prices.best_bid),
            "best_ask": _f(prices.best_ask),
            "spread": _f(prices.spread),
            "volume": _f(metrics.volume_num) or _f(metrics.volume),
            "volume_24h": _f(metrics.volume_24hr),
            "liquidity": _f(metrics.liquidity_num) or _f(metrics.liquidity),
            "closed": bool(state.closed),
            "active": bool(state.active),
            "uma_resolution_status": market.resolution.uma_resolution_status,
            "tags": [t.label for t in (market.tags or ()) if t.label],
            "categories": [market.category] if market.category else [],
            "slug": str(market.slug or ""),
            # Full record for diagnostics. ``mode="json"`` because the model
            # holds Decimals and datetimes and the universe is served as JSON.
            "raw": market.model_dump(mode="json"),
        }

    def _cache_market(self, payload: dict[str, Any]) -> None:
        symbol = payload["symbol"]
        if not symbol:
            return
        self._market_cache[symbol] = payload
        if payload["clob_token_ids"]:
            self._clob_token_cache[symbol] = payload["clob_token_ids"][0]

    def _fetch_market_by_condition_id(self, symbol: str) -> dict[str, Any] | None:
        """Fetch and cache a single market by its condition id."""
        market = None
        # Both states have to be asked for explicitly: a settled market is
        # absent from the default listing, which would leave outcomes unreadable.
        for closed in (False, True):
            try:
                page = self._client.list_markets(
                    condition_ids=[symbol],
                    closed=closed,
                    include_tag=True,
                    page_size=1,
                )
                market = next(page.iter_items(), None)
            except PolymarketError as exc:
                logger.warning("Polymarket fetch failed for %s: %s", symbol, exc)
                return None
            if market is not None:
                break
        if market is None:
            return None
        payload = self._to_payload(market)
        self._cache_market(payload)
        return payload

    def _get_yes_clob_token(self, symbol: str) -> str | None:
        token = self._clob_token_cache.get(symbol)
        if token is not None:
            return token
        cached = self._market_cache.get(symbol)
        if cached and cached["clob_token_ids"]:
            self._clob_token_cache[symbol] = cached["clob_token_ids"][0]
            return self._clob_token_cache[symbol]
        fetched = self._fetch_market_by_condition_id(symbol)
        if fetched and fetched["clob_token_ids"]:
            return fetched["clob_token_ids"][0]
        return None

    def _price_history(
        self, symbol: str, start: datetime, end: datetime, bucket_seconds: int
    ) -> list[Any]:
        """Return YES-side price points for *symbol* over [start, end)."""
        token = self._get_yes_clob_token(symbol)
        if token is None:
            raise ValueError(
                f"No CLOB token id known for {symbol}; "
                "discover_active_markets or fetch the market first."
            )
        page = self._client.list_price_history(
            asset_id=token,
            start=_utc(start),
            end=_utc(end),
            bucket_seconds=bucket_seconds,
        )
        return list(page.iter_items())

    # ── DataProvider interface ─────────────────────────────────────────

    def get_current_price(self, symbol: str) -> MarketSnapshot:
        """Return the current YES probability, with the NBBO in ``extra``."""
        payload = self._market_cache.get(symbol) or self._fetch_market_by_condition_id(symbol)
        if payload is None:
            raise ValueError(f"No polymarket market found for {symbol}")
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            price=float(payload["current_price"]),
            extra={
                "best_bid": payload["best_bid"],
                "best_ask": payload["best_ask"],
                "spread": payload["spread"],
                "volume_24h": payload["volume_24h"],
                "liquidity": payload["liquidity"],
                "outcome_prices": payload["outcome_prices"],
            },
        )

    def get_price_at(
        self, symbol: str, at: datetime, interval: str = "1m"
    ) -> MarketSnapshot:
        """Return the YES price nearest *at*; ``interval`` exists for protocol parity only."""
        at = _utc(at)
        points = self._price_history(
            symbol, at - timedelta(hours=1), at + timedelta(hours=1), 60
        )
        if not points:
            raise ValueError(
                f"No CLOB price history for {symbol} around {at.isoformat()}"
            )
        nearest = min(points, key=lambda p: abs(p.timestamp - at))
        return MarketSnapshot(
            symbol=symbol,
            timestamp=nearest.timestamp,
            price=float(nearest.price),
        )

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketSnapshot]:
        """Return historical YES-side price bars between *start* and *end*."""
        points = self._price_history(
            symbol, start, end, max(60, interval_to_seconds(interval))
        )
        return [
            MarketSnapshot(
                symbol=symbol,
                timestamp=p.timestamp,
                price=float(p.price),
                close=float(p.price),
            )
            for p in points
        ]

    # ── EventDataProvider interface ────────────────────────────────────

    def discover_active_markets(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Return active markets matching *filters*, caching each.

        Args:
            filters: Server-side ``resolution_window_hours_min``/``_max``,
                ``min_total_volume``, ``min_liquidity``, ``tag_id``; client-side
                ``exclude_disputed``, ``min_yes_price``/``max_yes_price``,
                ``min_24h_volume_usd``, ``min_orderbook_depth_usd``, ``categories``,
                ``binary_only``, and the ``max_markets_per_trial`` cap.
        """
        now = datetime.now(timezone.utc)
        win_min_h = filters.get("resolution_window_hours_min")
        win_max_h = filters.get("resolution_window_hours_max")
        server: dict[str, Any] = {
            "closed": False,
            # Tags only come back when explicitly requested, and the
            # observation space advertises them to the agent.
            "include_tag": True,
            "page_size": _DISCOVERY_PAGE_SIZE,
        }
        if win_min_h is not None:
            server["end_date_min"] = now + timedelta(hours=float(win_min_h))
        if win_max_h is not None:
            server["end_date_max"] = now + timedelta(hours=float(win_max_h))
        if filters.get("min_total_volume") is not None:
            server["volume_num_min"] = float(filters["min_total_volume"])
        if filters.get("min_liquidity") is not None:
            server["liquidity_num_min"] = float(filters["min_liquidity"])
        if filters.get("tag_id") is not None:
            server["tag_id"] = int(filters["tag_id"])

        # Post-filters
        min_yes = float(filters.get("min_yes_price", 0.01))
        max_yes = float(filters.get("max_yes_price", 0.99))
        min_24h = filters.get("min_24h_volume_usd")
        min_24h_v = float(min_24h) if min_24h is not None else None
        min_depth = filters.get("min_orderbook_depth_usd")
        min_depth_v = float(min_depth) if min_depth is not None else None
        categories_filter = [str(c).lower() for c in (filters.get("categories") or [])]
        binary_only = bool(filters.get("binary_only", True))
        exclude_disputed = bool(filters.get("exclude_disputed", True))
        cap = int(filters.get("max_markets_per_trial", 50))

        # Markets in the UMA dispute window are still open past their event time,
        # so re-check the window client-side, allowing 60s of clock drift.
        skew = timedelta(seconds=60)
        win_min_dt = now + timedelta(hours=float(win_min_h)) - skew if win_min_h is not None else None
        win_max_dt = now + timedelta(hours=float(win_max_h)) + skew if win_max_h is not None else None

        try:
            markets = self._client.list_markets(**server).iter_items()
        except PolymarketError as exc:
            logger.warning("Polymarket market discovery failed: %s", exc)
            return []

        kept: list[dict[str, Any]] = []
        try:
            for market in markets:
                payload = self._to_payload(market)
                if payload["closed"] or not payload["active"]:
                    continue
                # Client-side: upstream's uma_resolution_status is an inclusion
                # filter, so it cannot exclude disputed markets.
                if exclude_disputed and (
                    str(payload["uma_resolution_status"] or "").lower() == "disputed"
                ):
                    continue
                if binary_only and len(payload["outcomes"]) != 2:
                    continue
                res_at = payload["resolution_at"]
                if res_at is None:
                    continue
                if win_min_dt is not None and res_at < win_min_dt:
                    continue
                if win_max_dt is not None and res_at > win_max_dt:
                    continue
                cp = payload["current_price"]
                if cp != cp:  # NaN check
                    continue
                if cp < min_yes or cp > max_yes:
                    continue
                if min_24h_v is not None and (payload["volume_24h"] or 0.0) < min_24h_v:
                    continue
                if min_depth_v is not None:
                    # Depth proxy: liquidity when known, otherwise a spread
                    # under 5 cents as a sanity check.
                    liq = payload["liquidity"] or 0.0
                    if liq and liq < min_depth_v:
                        continue
                    if not liq:
                        spread = payload["spread"]
                        if spread is None or spread > 0.05:
                            continue
                if categories_filter:
                    cats = [c.lower() for c in payload["categories"]]
                    if not any(c in cats for c in categories_filter):
                        continue
                if not payload["symbol"]:
                    continue
                if not payload["clob_token_ids"]:
                    continue  # can't price-query; skip
                self._cache_market(payload)
                kept.append(payload)
                if len(kept) >= cap:
                    break
        except PolymarketError as exc:
            logger.warning(
                "Polymarket discovery stopped after %d markets: %s", len(kept), exc
            )
        return kept

    def get_event_outcome(self, symbol: str) -> float | None:
        """Return 1, 0 or 0.5 once *symbol* resolves, else None.

        Always re-fetched, since resolution changes after submission.
        """
        fresh = self._fetch_market_by_condition_id(symbol)
        if fresh is None or not fresh["closed"]:
            return None
        outcome_prices = fresh["outcome_prices"]
        yes_price = outcome_prices[0] if outcome_prices else None
        if yes_price is None or yes_price != yes_price:  # missing or NaN
            return None
        # Resolved binary markets pay in {0, 0.5, 1}; allow float drift.
        for target in (1.0, 0.0, 0.5):
            if abs(yes_price - target) < 1e-6:
                return target
        # If the outcome prices are non-canonical (e.g. mid-resolution),
        # treat as still-pending rather than guessing.
        logger.warning(
            "Polymarket %s closed but outcome prices=%s not in {0,0.5,1}; "
            "treating as pending",
            symbol,
            outcome_prices,
        )
        return None

    def get_market_metadata(self, symbol: str) -> dict[str, Any]:
        """Return cached market payload, fetching if not seen."""
        cached = self._market_cache.get(symbol)
        if cached is not None:
            return cached
        fetched = self._fetch_market_by_condition_id(symbol)
        if fetched is None:
            raise ValueError(f"No polymarket market found for {symbol}")
        return fetched


# ── Helpers ────────────────────────────────────────────────────────────


def _f(value: Any) -> float | None:
    """Coerce an SDK Decimal (or None) to float; None on parse failure."""
    if value is None:
        return None
    try:
        out = float(value)
    except (ValueError, TypeError):
        return None
    return None if out != out else out  # drop NaN


def _utc(moment: datetime) -> datetime:
    """Return *moment* as an aware UTC datetime."""
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment
