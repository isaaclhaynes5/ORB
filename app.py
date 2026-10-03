"""
ORB Scanner - Streamlit front end for orb_engine.py

    pip install -r requirements.txt
    streamlit run app.py

Alpaca (optional): put APCA_API_KEY_ID / APCA_API_SECRET_KEY (and APCA_FEED)
in .streamlit/secrets.toml and the provider appears in the sidebar.
"""
from __future__ import annotations

import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from orb_engine import (
    MARKET_OPEN, NY, AlpacaProvider, BarProvider, BreakoutSignal, InsufficientDataError,
    ORBConfig, YFinanceProvider, _first_per_side, _session_ts, completed_bars,
    compute_orb, detect_breakouts,
)

st.set_page_config(page_title="ORB Scanner", page_icon="📈", layout="wide")

BULL, BEAR, LEVEL, MUTED = "#2BB3A3", "#E8615A", "#7AA2F7", "#8A94A6"
BAND = "rgba(122,162,247,0.10)"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 500; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class DemoProvider:
    """Synthetic but realistic session (U-shaped volume, spikes). Deterministic per symbol/day.
    For today it reveals bars only up to the current minute, so live mode can be tested anytime."""

    def fetch_session(self, symbol: str, session: date) -> pd.DataFrame:
        rng = np.random.default_rng(zlib.crc32(f"{symbol}{session}".encode()))
        n = 390
        idx = pd.date_range(_session_ts(session, MARKET_OPEN), periods=n, freq="1min")
        base = 40 + rng.random() * 400
        ret = rng.normal(0, 0.0008, n)
        ret[15:] += rng.choice([-1, 1]) * 0.00012          # intraday trend after the open
        close = base * np.exp(np.cumsum(ret))
        open_ = np.r_[base, close[:-1]]
        wick = np.abs(rng.normal(0, 0.0005, n)) * close
        u = np.linspace(-1, 1, n)
        vol = (1 + 1.6 * u**2) * rng.lognormal(10, 0.35, n)
        vol *= np.where(rng.random(n) < 0.07, rng.uniform(2, 4, n), 1)  # sporadic spikes
        return pd.DataFrame(
            {"Open": open_, "High": np.maximum(open_, close) + wick,
             "Low": np.minimum(open_, close) - wick, "Close": close, "Volume": vol.round()},
            index=idx,
        )


def _alpaca_secrets() -> dict | None:
    try:
        s = st.secrets
        if "APCA_API_KEY_ID" in s and "APCA_API_SECRET_KEY" in s:
            return {"key": s["APCA_API_KEY_ID"], "secret": s["APCA_API_SECRET_KEY"],
                    "feed": s.get("APCA_FEED", "iex")}
    except Exception:
        pass
    return None


@st.cache_resource
def get_provider(name: str) -> BarProvider:
    if name == "Alpaca":
        a = _alpaca_secrets()
        return AlpacaProvider(a["key"], a["secret"], feed=a["feed"])
    if name == "Demo data":
        return DemoProvider()
    return YFinanceProvider()


@st.cache_data(ttl=50, show_spinner=False)
def fetch_all(provider_name: str, symbols: tuple[str, ...], session_iso: str) -> dict:
    """Fetch every symbol in parallel. Returns {symbol: DataFrame | error string}."""
    provider, session = get_provider(provider_name), date.fromisoformat(session_iso)

    def one(sym: str):
        try:
            return sym, provider.fetch_session(sym, session)
        except Exception as e:  # network, bad ticker, rate limit
            return sym, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        return dict(pool.map(one, symbols))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze(symbol: str, raw, session: date, cfg: ORBConfig, now: pd.Timestamp) -> dict:
    out = {"symbol": symbol, "bars": None, "levels": None, "signals": [], "note": None}
    if isinstance(raw, str):
        out["note"] = f"Couldn't load data ({raw})"
        return out
    bars = completed_bars(raw, now)
    out["bars"] = bars
    cutoff = _session_ts(session, MARKET_OPEN) + pd.Timedelta(minutes=cfg.window_minutes)
    try:
        lv = compute_orb(bars, symbol, session, cfg)
    except InsufficientDataError:
        if now < cutoff:
            out["note"] = "Range forms 09:30–09:45 ET"
        elif bars.empty:
            out["note"] = "No bars for this date. Check the symbol or pick a trading day."
        else:
            out["note"] = "Too few opening-range bars"
        return out
    sigs = detect_breakouts(bars, lv, cfg)
    out["levels"] = lv
    out["signals"] = _first_per_side(sigs, set()) if cfg.first_per_side else sigs
    return out


def summary_row(r: dict) -> dict:
    lv, bars, sigs = r["levels"], r["bars"], r["signals"]
    last = float(bars["Close"].iloc[-1]) if bars is not None and len(bars) else None
    if lv is None:
        status = r["note"]
    elif last > lv.orb_high:
        status = "Above range"
    elif last < lv.orb_low:
        status = "Below range"
    else:
        status = "Inside range"
    latest = sigs[-1] if sigs else None
    return {
        "Symbol": r["symbol"],
        "Last": last,
        "ORB high": lv.orb_high if lv else None,
        "ORB low": lv.orb_low if lv else None,
        "Range $": lv.orb_range if lv else None,
        "Status": status,
        "Breakouts": len(sigs),
        "Latest": f"{latest.direction.title()} {latest.timestamp:%H:%M}" if latest else "",
    }


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def chart(r: dict, cfg: ORBConfig) -> go.Figure:
    bars, lv, sigs = r["bars"], r["levels"], r["signals"]
    x = bars.index.tz_localize(None)  # plot ET wall-clock time
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26],
                        vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=x, open=bars["Open"], high=bars["High"], low=bars["Low"], close=bars["Close"],
        increasing_line_color=BULL, decreasing_line_color=BEAR,
        increasing_fillcolor=BULL, decreasing_fillcolor=BEAR, name="Price", showlegend=False,
    ), row=1, col=1)

    up = (bars["Close"] >= bars["Open"]).to_numpy()
    fig.add_trace(go.Bar(x=x, y=bars["Volume"], marker_color=np.where(up, BULL, BEAR),
                         marker_line_width=0, opacity=0.6, name="Volume", showlegend=False),
                  row=2, col=1)

    if lv:
        t0 = _session_ts(lv.session, MARKET_OPEN).tz_localize(None)
        t1 = t0 + pd.Timedelta(minutes=cfg.window_minutes)
        for row in (1, 2):
            fig.add_vrect(x0=t0, x1=t1, fillcolor=BAND, line_width=0, row=row, col=1)
        for y, name in ((lv.orb_high, "ORB high"), (lv.orb_low, "ORB low")):
            fig.add_hline(y=y, line=dict(color=LEVEL, width=1, dash="dot"), row=1, col=1,
                          annotation_text=f"{name} {y:.2f}", annotation_position="top left",
                          annotation_font_color=LEVEL)
        fig.add_hline(y=lv.volume_threshold, line=dict(color=MUTED, width=1, dash="dash"),
                      row=2, col=1, annotation_text=f"{cfg.volume_multiplier:g}× avg",
                      annotation_position="top left", annotation_font_color=MUTED)

        for direction, color, symbol, ycol, pad in (
            ("BULLISH", BULL, "triangle-up", "Low", -1), ("BEARISH", BEAR, "triangle-down", "High", 1)
        ):
            s = [v for v in sigs if v.direction == direction]
            if s:
                ts = pd.DatetimeIndex([v.timestamp for v in s])
                y = bars.loc[ts, ycol].to_numpy() + pad * 0.08 * lv.orb_range
                fig.add_trace(go.Scatter(
                    x=ts.tz_localize(None), y=y, mode="markers", name=direction.title(),
                    marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color="#0F1724")),
                    customdata=np.c_[[v.close for v in s], [v.volume_ratio for v in s]],
                    hovertemplate="%{x|%H:%M} " + direction.title()
                                  + "<br>Close %{customdata[0]:.2f}<br>Vol %{customdata[1]:.2f}×<extra></extra>",
                    showlegend=False,
                ), row=1, col=1)

    fig.update_layout(
        height=620, template="plotly_dark", hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
        font=dict(family="IBM Plex Sans, system-ui, sans-serif"),
    )
    fig.update_xaxes(gridcolor="rgba(138,148,166,0.12)", tickformat="%H:%M")
    fig.update_yaxes(gridcolor="rgba(138,148,166,0.12)")
    return fig


def signals_frame(sigs: list[BreakoutSignal]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Time": s.timestamp.strftime("%H:%M"), "Direction": s.direction.title(),
        "Close": s.close, "Volume": s.volume, "Vol ×avg": s.volume_ratio, "Beyond level $": s.distance,
    } for s in sigs])


def notify_new(results: list[dict], session: date) -> None:
    """Toast breakouts that appeared since the last refresh (not ones already on screen)."""
    key = f"seen-{session}"
    current = {(s.symbol, s.timestamp, s.direction) for r in results for s in r["signals"]}
    if key not in st.session_state:
        st.session_state[key] = current
        return
    for sym, ts, d in sorted(current - st.session_state[key], key=lambda t: t[1]):
        st.toast(f"{sym}: {d.lower()} breakout at {ts:%H:%M}", icon="🟢" if d == "BULLISH" else "🔴")
    st.session_state[key] |= current


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Scanner")
    providers = ["yfinance", "Demo data"] + (["Alpaca"] if _alpaca_secrets() else [])
    provider_name = st.selectbox("Data source", providers,
                                 help="Demo data works any time, including weekends.")
    raw_syms = st.text_area("Watchlist", "SPY, QQQ, AAPL, NVDA, TSLA",
                            help="Comma or space separated")
    symbols = list(dict.fromkeys(s.strip().upper() for s in raw_syms.replace(",", " ").split() if s.strip()))
    today = pd.Timestamp.now(tz=NY).date()
    session = st.date_input("Session (ET)", today, max_value=today)
    vol_mult = st.slider("Volume filter (× opening-range average)", 1.0, 3.0, 1.5, 0.1)
    first_only = st.toggle("First breakout per side only")
    live = st.toggle("Live refresh every minute", value=session == today, disabled=session != today)
    if provider_name == "Alpaca" and (_alpaca_secrets() or {}).get("feed") == "iex":
        st.caption("IEX feed sees a fraction of market volume, so the volume filter is noisier.")

cfg = ORBConfig(volume_multiplier=vol_mult, first_per_side=first_only)


@st.fragment(run_every=60 if live and session == today else None)
def board() -> None:
    now = pd.Timestamp.now(tz=NY)
    st.title("ORB Scanner")
    st.caption(f"15-minute opening range, session {session:%a %b %d, %Y}. "
               f"Updated {now:%H:%M:%S} ET from {provider_name}.")

    if not symbols:
        st.info("Add at least one ticker to the watchlist in the sidebar.")
        return

    with st.spinner("Loading bars"):
        raw = fetch_all(provider_name, tuple(symbols), session.isoformat())
    results = [analyze(s, raw[s], session, cfg, now) for s in symbols]
    if live:
        notify_new(results, session)

    st.dataframe(
        pd.DataFrame([summary_row(r) for r in results]), hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in ("Last", "ORB high", "ORB low", "Range $")},
    )

    sym = st.selectbox("Chart", symbols)
    r = next(x for x in results if x["symbol"] == sym)
    lv = r["levels"]

    if r["bars"] is None or r["bars"].empty:
        st.warning(r["note"] or "No bars yet.")
        return

    if lv:
        c = st.columns(4)
        c[0].metric("ORB high", f"{lv.orb_high:,.2f}")
        c[1].metric("ORB low", f"{lv.orb_low:,.2f}")
        c[2].metric("ORB range", f"${lv.orb_range:,.2f}", f"{lv.orb_range / lv.orb_low:.2%} of price",
                    delta_color="off")
        c[3].metric("Volume needed", f"{lv.volume_threshold:,.0f}", f"avg {lv.avg_volume:,.0f}/min",
                    delta_color="off")
    else:
        st.info(r["note"])

    st.plotly_chart(chart(r, cfg), width="stretch", config={"displayModeBar": False})

    if r["signals"]:
        st.subheader(f"{sym} breakouts")
        st.dataframe(signals_frame(r["signals"]), hide_index=True, width="stretch",
                     column_config={
                         "Close": st.column_config.NumberColumn(format="%.2f"),
                         "Volume": st.column_config.NumberColumn(format="%d"),
                         "Vol ×avg": st.column_config.NumberColumn(format="%.2f×"),
                         "Beyond level $": st.column_config.NumberColumn(format="%.2f"),
                     })
    elif lv:
        st.caption("No breakouts yet: no close beyond the range on qualifying volume.")


board()
