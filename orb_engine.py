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
# Core calculations (pure, vectorised)
# --------------------------------------------------------------------------- #
def completed_bars(bars: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop any bar whose minute has not finished yet."""
    cut = bars.index.searchsorted(now - BAR, side="right")
    return bars.iloc[:cut]


def compute_orb(bars: pd.DataFrame, symbol: str, session: date,
                cfg: ORBConfig = ORBConfig()) -> ORBLevels:
    """ORB_High / ORB_Low / ORB_Range from bars in [09:30, 09:30 + window)."""
    open_ts = _session_ts(session, MARKET_OPEN)
    cutoff = open_ts + pd.Timedelta(minutes=cfg.window_minutes)
    i0, i1 = bars.index.searchsorted([open_ts, cutoff])  # O(log n) slice
    window = bars.iloc[i0:i1]

    if len(window) < cfg.min_window_bars:
        raise InsufficientDataError(
            f"{symbol}: {len(window)}/{cfg.window_minutes} opening-range bars available")

    high = float(window["High"].to_numpy().max())
    low = float(window["Low"].to_numpy().min())
    # A missing 1m bar means zero prints, so divide by the window length,
    # not the bar count: this is the true average of the 15 individual minutes.
    avg_vol = float(window["Volume"].to_numpy().sum()) / cfg.window_minutes

    return ORBLevels(
        symbol=symbol, session=session, orb_high=high, orb_low=low,
        orb_range=round(high - low, 6), avg_volume=avg_vol,
        volume_threshold=avg_vol * cfg.volume_multiplier, bars_used=len(window),
    )


def detect_breakouts(bars: pd.DataFrame, levels: ORBLevels,
                     cfg: ORBConfig = ORBConfig(),
                     start: Optional[pd.Timestamp] = None) -> list[BreakoutSignal]:
    """Flag every bar at/after `start` (default: end of ORB window) that breaks out."""
    if start is None:
        start = _session_ts(levels.session, MARKET_OPEN) + pd.Timedelta(minutes=cfg.window_minutes)
    post = bars.iloc[bars.index.searchsorted(start):]
    if post.empty:
        return []

    close = post["Close"].to_numpy()
    vol = post["Volume"].to_numpy()
    vol_ok = vol >= levels.volume_threshold
    bull = (close > levels.orb_high) & vol_ok
    bear = (close < levels.orb_low) & vol_ok
    hits = np.flatnonzero(bull | bear)

    denom = levels.avg_volume or np.nan
    ts = post.index
    return [
        BreakoutSignal(
            symbol=levels.symbol, timestamp=ts[i],
            direction="BULLISH" if bull[i] else "BEARISH",
            close=float(close[i]), volume=float(vol[i]),
            volume_ratio=float(vol[i] / denom),
            distance=float(close[i] - levels.orb_high if bull[i] else levels.orb_low - close[i]),
        )
        for i in hits
    ]


def _first_per_side(signals: list[BreakoutSignal], fired: set[str]) -> list[BreakoutSignal]:
    out = []
    for s in signals:
        if s.direction not in fired:
            fired.add(s.direction)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Session analysis + live engine
# --------------------------------------------------------------------------- #
def analyze_session(symbol: str, provider: BarProvider, session: Optional[date] = None,
                    cfg: ORBConfig = ORBConfig()) -> tuple[ORBLevels, pd.DataFrame]:
    """One-shot: ORB levels plus all breakouts so far (completed bars only)."""
    session = session or datetime.now(NY).date()
    bars = completed_bars(provider.fetch_session(symbol, session), pd.Timestamp.now(tz=NY))
    levels = compute_orb(bars, symbol, session, cfg)
    signals = detect_breakouts(bars, levels, cfg)
    if cfg.first_per_side:
        signals = _first_per_side(signals, set())
    df = pd.DataFrame([asdict(s) for s in signals])
    return levels, (df.set_index("timestamp") if not df.empty else df)


class ORBEngine:
    """Stateful live engine: locks the ORB once, then evaluates each new completed bar exactly once."""

    def __init__(self, symbol: str, provider: BarProvider,
                 cfg: ORBConfig = ORBConfig(), session: Optional[date] = None):
        self.symbol = symbol.upper()
        self.provider = provider
        self.cfg = cfg
        self.session = session or datetime.now(NY).date()
        self.cutoff = _session_ts(self.session, MARKET_OPEN) + pd.Timedelta(minutes=cfg.window_minutes)
        self.close_ts = _session_ts(self.session, MARKET_CLOSE)
        self.levels: Optional[ORBLevels] = None
        self.signals: list[BreakoutSignal] = []
        self._cursor = self.cutoff          # timestamp of next bar to evaluate
        self._fired: set[str] = set()

    def update(self, now: Optional[pd.Timestamp] = None) -> list[BreakoutSignal]:
        now = pd.Timestamp.now(tz=NY) if now is None else pd.Timestamp(now).tz_convert(NY)
        if now < self.cutoff:
            return []

        bars = completed_bars(self.provider.fetch_session(self.symbol, self.session), now)

        if self.levels is None:
            try:
                self.levels = compute_orb(bars, self.symbol, self.session, self.cfg)
                log.info("ORB locked %s", asdict(self.levels))
            except InsufficientDataError as e:
                log.warning("%s - retrying next poll (feed lag?)", e)
                return []

        new = detect_breakouts(bars, self.levels, self.cfg, start=self._cursor)
        if len(bars) and bars.index[-1] >= self._cursor:
            self._cursor = bars.index[-1] + BAR   # never re-evaluate a bar
        if self.cfg.first_per_side:
            new = _first_per_side(new, self._fired)
        self.signals.extend(new)
        return new

    def run(self, on_signal: Optional[Callable[[BreakoutSignal], None]] = None) -> None:
        on_signal = on_signal or (lambda s: log.info("BREAKOUT %s", asdict(s)))
        lag = pd.Timedelta(seconds=self.cfg.poll_lag_seconds)
        while True:
            now = pd.Timestamp.now(tz=NY)
            if now > self.close_ts + lag:
                break
            for sig in self.update(now):
                on_signal(sig)
            wake = max(now.floor("min") + BAR, self.cutoff) + lag   # first wake: 09:45:01
            time.sleep(max((wake - pd.Timestamp.now(tz=NY)).total_seconds(), 0.05))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_provider(name: str) -> BarProvider:
    if name == "alpaca":
        return AlpacaProvider(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"],
                              feed=os.environ.get("APCA_FEED", "iex"))
    return YFinanceProvider()


def main() -> None:
    p = argparse.ArgumentParser(description="15-minute Opening Range Breakout engine")
    p.add_argument("symbol")
    p.add_argument("--provider", choices=["yfinance", "alpaca"], default="yfinance")
    p.add_argument("--date", type=date.fromisoformat, help="session date YYYY-MM-DD")
    p.add_argument("--live", action="store_true", help="poll until the close")
    p.add_argument("--first-only", action="store_true", help="first breakout per side only")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ORBConfig(first_per_side=a.first_only)
    provider = _build_provider(a.provider)

    if a.live:
        ORBEngine(a.symbol, provider, cfg, a.date).run()
        return

    levels, signals = analyze_session(a.symbol.upper(), provider, a.date, cfg)
    print(f"
{levels.symbol} {levels.session}  ORB_High={levels.orb_high:.2f}  "
          f"ORB_Low={levels.orb_low:.2f}  ORB_Range=${levels.orb_range:.2f}  "
          f"vol_threshold={levels.volume_threshold:,.0f}
")
    print(signals.to_string() if not signals.empty else "No breakouts.")


if __name__ == "__main__":
    main()
