"""
orb_engine.py - 15-Minute Opening Range Breakout (ORB) calculation engine.

Bar-labelling convention (both yfinance and Alpaca): a 1-minute bar is stamped
with its START time, so the bar labelled 09:44 covers 09:44:00-09:44:59.999.

    Opening range  = bars 09:30 ... 09:44  (15 bars, 09:30:00 -> 09:45:00)
    ORB locked     = 09:45:01 (first poll after the 09:44 bar closes)
    First breakout candidate = the 09:45 bar, evaluated once it closes (~09:46)

Only COMPLETED bars are ever evaluated, so a still-forming bar can never
trigger a false breakout.

Usage:
    python orb_engine.py SPY                 # analyse today (or a past --date)
    python orb_engine.py SPY --live          # poll live until the close
    python orb_engine.py SPY --provider alpaca   # needs APCA_API_KEY_ID / APCA_API_SECRET_KEY
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime
from typing import Callable, Optional, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("orb_engine")

NY = ZoneInfo("America/New_York")  # handles EST/EDT automatically
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
BAR = pd.Timedelta(minutes=1)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ORBConfig:
    window_minutes: int = 15
    volume_multiplier: float = 1.5
    min_window_bars: int = 15        # lower it to tolerate no-trade minutes in thin names
    poll_lag_seconds: float = 1.0    # wake 1s after each minute -> 09:45:01, 09:46:01, ...
    first_per_side: bool = False     # True = report only the first bull and first bear break


@dataclass(frozen=True, slots=True)
class ORBLevels:
    symbol: str
    session: date
    orb_high: float
    orb_low: float
    orb_range: float
    avg_volume: float
    volume_threshold: float
    bars_used: int


@dataclass(frozen=True, slots=True)
class BreakoutSignal:
    symbol: str
    timestamp: pd.Timestamp
    direction: str        # "BULLISH" | "BEARISH"
    close: float
    volume: float
    volume_ratio: float   # bar volume / avg opening-range volume
    distance: float       # dollars beyond the broken level


class InsufficientDataError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Data providers
# --------------------------------------------------------------------------- #
class BarProvider(Protocol):
    def fetch_session(self, symbol: str, session: date) -> pd.DataFrame:
        """Return regular-session 1m bars: tz-aware NY index, OHLCV float columns."""
        ...


def _session_ts(session: date, t: dtime) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(session, t, tzinfo=NY))


def _normalize(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV, dtype="float64",
                            index=pd.DatetimeIndex([], tz=NY))
    df = df[OHLCV].astype("float64")
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(NY)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.between_time(MARKET_OPEN, dtime(15, 59))


class YFinanceProvider:
    """Free, no key. 1m history limited to ~30 days; live bars can lag a few seconds."""

    def fetch_session(self, symbol: str, session: date) -> pd.DataFrame:
        import yfinance as yf

        df = yf.Ticker(symbol).history(
            start=_session_ts(session, MARKET_OPEN).to_pydatetime(),
            end=_session_ts(session, MARKET_CLOSE).to_pydatetime(),
            interval="1m", prepost=False, auto_adjust=False, actions=False,
        )
        return _normalize(df)


class AlpacaProvider:
    """alpaca-py. feed='iex' is free but sees only IEX volume; 'sip' is consolidated."""

    def __init__(self, api_key: str, secret_key: str, feed: str = "iex"):
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._feed = DataFeed(feed)

    def fetch_session(self, symbol: str, session: date) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, feed=self._feed,
            start=_session_ts(session, MARKET_OPEN), end=_session_ts(session, MARKET_CLOSE),
        )
        df = self._client.get_stock_bars(req).df
        if df.empty:
            return _normalize(None)
        df = df.xs(symbol, level="symbol").rename(columns=str.capitalize)
        return _normalize(df)


# --------------------------------------------------------------------------- #
