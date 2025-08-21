import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
from scipy.stats import norm
from datetime import datetime
from io import BytesIO
import math
import warnings
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import time

warnings.filterwarnings("ignore", message="Calling float on a single element Series is deprecated")

# ---------- Defaults ----------
DEFAULT_TICKERS = ["AMD","HIMS","NBIS","AMZN","AAPL"]
WEEKLY_YIELD_MIN_DEFAULT = 0.8  # % per week (ROC basis)
OTM_K, OTM_MIN, OTM_MAX = 0.9, 0.03, 0.12   # min IV-aware OTM buffer band
DAYS_PER_MONTH, DAYS_PER_YEAR = 30.0, 365.0
RISK_FREE = 0.00

# ---------- Helpers ----------
def find_support_resistance(prices, order=5):
    if prices.empty: 
        return None, None
    lows  = prices.iloc[argrelextrema(prices.values, np.less_equal, order=order)[0]]
    highs = prices.iloc[argrelextrema(prices.values, np.greater_equal, order=order)[0]]
    support    = float(lows.iloc[-1])  if not lows.empty  else None
    resistance = float(highs.iloc[-1]) if not highs.empty else None
    return support, resistance

def find_two_supports(prices, order=5):
    if prices.empty: 
        return None, None, None
    idx_lows  = argrelextrema(prices.values, np.less_equal, order=order)[0]
    idx_highs = argrelextrema(prices.values, np.greater_equal, order=order)[0]
    lows  = prices.iloc[idx_lows]
    highs = prices.iloc[idx_highs]
    core_support   = float(lows.iloc[-1])  if not lows.empty else None
    second_support = float(lows.iloc[-2])  if len(lows) >= 2 else None
    resistance     = float(highs.iloc[-1]) if not highs.empty else None
    return core_support, second_support, resistance

def compute_wheel_stock_score(tkr):
    """0-1 composite that prefers uptrends, lower volatility, and smaller drawdowns."""
    try:
        tk = yf.Ticker(tkr)
        hist = tk.history(period="6mo")
        if hist.empty: 
            return 0.0
        close = hist["Close"]
        price = float(close.iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        trend = 0.0
        trend += 0.5 if (price > (ma50 or price)) else 0.0
        trend += 0.5 if ((ma50 or 0) > (ma200 or 0)) else 0.0
        vol60 = float(close.pct_change().dropna().tail(60).std() * np.sqrt(252)) if len(close) > 60 else None
        roll_max = close.cummax()
        mdd = float(((close/roll_max) - 1.0).min())
        vol_term = 1.0 - min(max((vol60 or 0.6), 0.2), 0.6) / 0.6
        dd_term  = 1.0 - min(max(abs(mdd or 0.5), 0.1), 0.5) / 0.5
        score = 0.4*trend + 0.35*vol_term + 0.25*dd_term
        return round(float(score), 4)
    except Exception:
        return 0.0

def required_otm_pct(iv_decimal, dte, fallback=0.02):
    """Clamp IV-based min OTM buffer between OTM_MIN and OTM_MAX; fall back if IV missing."""
    try:
        if iv_decimal is None or not np.isfinite(iv_decimal) or dte is None or dte <= 0: 
            return fallback
        buf = OTM_K * float(iv_decimal) * math.sqrt(float(dte) / 252.0)
        return max(OTM_MIN, min(buf, OTM_MAX))
    except Exception:
        return fallback

def put_delta(S, K, T, iv, r=RISK_FREE):
    """Black–Scholes put delta (negative). Robust to NaNs."""
    if iv is None or not np.isfinite(iv) or iv <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S/K) + (r + 0.5*iv*iv) * T) / (iv * np.sqrt(T))
    return norm.cdf(d1) - 1.0

# ---------- Technical indicators ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bbands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower

@st.cache_data(ttl=900, show_spinner=False)
def fetch_expiries_for_tickers(tickers):
    expiries = set()
    for t in tickers:
        if not t:
            continue
        try:
            tk = yf.Ticker(t)
            opts = tk.options or []
            expiries.update(opts)
        except Exception:
            time.sleep(1.0)
            try:
                tk = yf.Ticker(t)
                opts = tk.options or []
                expiries.update(opts)
            except Exception:
                pass
        finally:
            time.sleep(0.4)
    return sorted(expiries)

@st.cache_data(ttl=600, show_spinner=False)
def get_history(ticker: str, lookback_days: int):
    tk = yf.Ticker(ticker)
    period = f"{max(lookback_days, 7)}d"
    hist = tk.history(period=period, interval="1d")
    return hist

def choose_lookback_by_dte(dte: int) -> int:
    if dte <= 14:
        return 90   # ~3 months
    elif dte <= 45:
        return 180  # ~6 months
    else:
        return 365  # ~12 months

def make_chart(
    ticker, dte, strike=None, net_strike=None, *,
    show_rsi=False, show_macd=False, show_bb=False,
    show_ma21=False, show_ma50=False, show_ma200=False
):
    lookback = choose_lookback_by_dte(dte or 30)
    hist = get_history(ticker, lookback)
    if hist.empty:
        return st.warning("No price history available to draw chart.")

    close = hist["Close"].copy()
    core_sup, second_sup, resistance = find_two_supports(close)

    # Precompute overlays
    ma21  = close.rolling(21).mean() if show_ma21 else None
    ma50  = close.rolling(50).mean() if show_ma50 else None
    ma200 = close.rolling(200).mean() if show_ma200 else None

    # Dynamic layout
    rows = 1 + int(show_rsi) + int(show_macd)
    row_heights = [0.6]
    titles = ["Price"]
    if show_rsi:
        row_heights.append(0.2)
        titles.append("RSI (14)")
    if show_macd:
        row_heights.append(0.2)
        titles.append("MACD (12,26,9)")

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=row_heights, subplot_titles=titles
    )

    price_row = 1
    next_row = 2
    rsi_row = next_row if show_rsi else None
    next_row = next_row + 1 if show_rsi else next_row
    macd_row = next_row if show_macd else None

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        name="Price", showlegend=False
    ), row=price_row, col=1)

    # Moving averages via toggles
    if show_ma21:
        fig.add_trace(go.Scatter(x=hist.index, y=ma21,  mode="lines", name="21d MA"), row=price_row, col=1)
    if show_ma50:
        fig.add_trace(go.Scatter(x=hist.index, y=ma50,  mode="lines", name="50d MA"), row=price_row, col=1)
    if show_ma200 and ma200 is not None and not ma200.dropna().empty:
        fig.add_trace(go.Scatter(x=hist.index, y=ma200, mode="lines", name="200d MA"), row=price_row, col=1)

    # Bollinger Bands on price
    if show_bb:
        mid, upper, lower = bbands(close, window=20, n_std=2.0)
        fig.add_trace(go.Scatter(x=hist.index, y=mid,   mode="lines", name="BB mid (20)"), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=upper, mode="lines", name="BB upper"), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=lower, mode="lines", name="BB lower"), row=price_row, col=1)

    # Horizontal levels on price
    shapes = []
    annots = []
    def hline_price(y, label):
        if y is None: 
            return
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y,
                           line=dict(width=1.5, dash="dot")))
        annots.append(dict(x=1.002, xref="paper", y=y, yref="y",
                           text=label, showarrow=False, font=dict(size=10),
                           xanchor="left", bgcolor="rgba(255,255,255,0.5)"))

    hline_price(core_sup,   "Core support")
    hline_price(second_sup, "2nd support")
    hline_price(resistance, "Resistance")
    if strike is not None:
        hline_price(strike, "Strike")
    if net_strike is not None:
        hline_price(net_strike, "Net strike")

    # RSI panel
    if show_rsi and rsi_row is not None:
        r = rsi(close, 14)
        fig.add_trace(go.Scatter(x=hist.index, y=r, mode="lines", name="RSI (14)"), row=rsi_row, col=1)
        # RSI 30/70 guides as lines (avoid shape yref issues)
        thirty = pd.Series(30, index=hist.index)
        seventy = pd.Series(70, index=hist.index)
        fig.add_trace(go.Scatter(x=hist.index, y=thirty, mode="lines", name="RSI 30", line=dict(dash="dot")), row=rsi_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=seventy, mode="lines", name="RSI 70", line=dict(dash="dot")), row=rsi_row, col=1)

    # MACD panel
    if show_macd and macd_row is not None:
        m_line, s_line, histo = macd(close)
        fig.add_trace(go.Bar(x=hist.index, y=histo, name="MACD hist"), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=m_line, mode="lines", name="MACD"), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=s_line, mode="lines", name="Signal"), row=macd_row, col=1)

    # Only show rangeslider when there is a single (price) row,
    # otherwise it can overlap the bottom indicators.
    fig.update_layout(
        height=520 if rows == 1 else (680 if rows == 2 else 860),
        margin=dict(l=40, r=120, t=40, b=40),
        shapes=shapes, annotations=annots,
        xaxis_title="Date", yaxis_title="Price",
        xaxis_rangeslider_visible=(rows == 1)
    )
    st.plotly_chart(fig, use_container_width=True)

def _fill_last_price(puts: pd.DataFrame) -> pd.Series:
    lp = pd.to_numeric(puts.get("lastPrice"), errors="coerce")
    bid = pd.to_numeric(puts.get("bid"), errors="coerce")
    ask = pd.to_numeric(puts.get("ask"), errors="coerce")
    mid = (bid + ask) / 2.0
    lp2 = lp.where(lp > 0)
    lp2 = lp2.fillna(mid.where(mid > 0))
    lp2 = lp2.fillna(bid.where(bid > 0))
    return lp2

def build_candidates(tickers, expiries, weekly_yield_min, guard, diagnostics=False):
    today = datetime.today().date()
    rows = []
    diag_rows = []

    for ticker in tickers:
        tk = yf.Ticker(ticker)
        try:
            h1d = tk.history(period="1d")
        except Exception:
            time.sleep(0.6)
            h1d = tk.history(period="1d")
        if h1d.empty: 
            continue
        price = float(h1d["Close"].iloc[-1])

        h90 = tk.history(period="90d")["Close"]
        core_sup, second_sup, resistance = find_two_supports(h90)
        w = tk.history(period="365d", interval="1wk")["Close"]
        week_low, _ = find_support_resistance(w)

        wheel_score = compute_wheel_stock_score(ticker)

        for expiry in expiries:
            try:
                chain = tk.option_chain(expiry)
                puts = chain.puts
            except Exception:
                time.sleep(0.6)
                try:
                    chain = tk.option_chain(expiry)
                    puts = chain.puts
                except Exception:
                    continue

            if puts.empty:
                continue

            puts = puts.rename(columns={"impliedVolatility": "iv"})
            puts["lp"] = _fill_last_price(puts)
            puts = puts[(puts["lp"].notna()) & (puts["lp"] > 0) & (puts["strike"] < price)]
            total = len(puts)
            if total == 0:
                continue

            dte = (pd.to_datetime(expiry).date() - today).days

            iv_series = pd.to_numeric(puts["iv"], errors="coerce")
            iv_median = iv_series.median() if iv_series.notna().any() else np.nan
            iv_used = iv_series.fillna(iv_median)
            hist_60 = tk.history(period="60d")["Close"]
            hv60 = hist_60.pct_change().dropna().std() * np.sqrt(252) if len(hist_60) >= 30 else np.nan
            iv_used = iv_used.fillna(hv60)

            req_pct = iv_used.apply(lambda v: required_otm_pct(v, dte, fallback=guard["FALLBACK_MIN_OTM"]))
            otm_pct = (price - puts["strike"]) / price
            min_otm_ok = (otm_pct >= req_pct)

            denom = puts["strike"]
            yld_pct = (puts["lp"] / denom) * 100.0
            wk_yld  = yld_pct / (dte/7.0 if dte > 0 else 1.0)
            monthly_yld = yld_pct * (DAYS_PER_MONTH / max(dte, 1))
            annual_yld  = yld_pct * (DAYS_PER_YEAR  / max(dte, 1))

            prem_ok  = puts["lp"] >= guard["MIN_PREMIUM_ABS"]
            wk_ok    = (wk_yld >= weekly_yield_min)

            if guard["USE_DELTA_BAND"]:
                T = max(dte, 1) / 365.0
                deltas = puts.apply(
                    lambda r: put_delta(price, float(r["strike"]), T,
                                        float(iv_used.loc[r.name]) if pd.notna(iv_used.loc[r.name]) else np.nan),
                    axis=1
                )
                dabs = pd.to_numeric(deltas, errors="coerce").abs()
                delta_ok = dabs.between(guard["DELTA_MIN"], guard["DELTA_MAX"], inclusive="both").fillna(False)
            else:
                delta_ok = pd.Series(True, index=puts.index)

            mask = prem_ok & wk_ok & min_otm_ok & delta_ok & yld_pct.notna()
            sel = puts[mask].copy()

            if diagnostics:
                diag_rows.append({
                    "Ticker": ticker, "Expiry": expiry, "Total puts": int(total),
                    "prem_ok": int(prem_ok.sum()),
                    "wk_ok": int(wk_ok.sum()),
                    "min_otm_ok": int(min_otm_ok.sum()),
                    "delta_ok": int(delta_ok.sum()),
                    "After all filters": int(len(sel))
                })

            if sel.empty: 
                continue

            for idx, opt in sel.iterrows():
                strike = float(opt["strike"]); premium = float(opt["lp"])
                net_strike = strike - premium
                yy = float(yld_pct.loc[idx]); wy = float(wk_yld.loc[idx])
                my = float(monthly_yld.loc[idx]); ay = float(annual_yld.loc[idx])

                rows.append({
                    "Ticker": ticker, "Price": round(price,2), "Strike": strike, "DTE": int(dte),
                    "Premium": premium,
                    "Yield %": round(yy,2), "Monthly Yield": round(my,2), "Annual Yield": round(ay,2), "Yield %/wk": round(wy,2),
                    "Net Strike": round(net_strike,2),
                    "OTM %": round(float(((price - strike) / price) * 100.0),2),
                    "Min OTM % req": round(float(req_pct.loc[idx] * 100.0),2),
                    "Wheel Stock Score": wheel_score,
                    "Expiry": expiry
                })

    df = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)

    if not df.empty:
        df["Comfort Score"] = pd.to_numeric(df["Yield %/wk"], errors="coerce").fillna(0.0).round(4)
        df["Rank Score"] = (pd.to_numeric(df["Comfort Score"], errors="coerce").fillna(0.0) *
                            pd.to_numeric(df["Wheel Stock Score"], errors="coerce").fillna(0.0)).round(6)
        df.sort_values(by=["Rank Score","Comfort Score","Yield %/wk"], ascending=False, inplace=True)

    return df, diag

def to_excel_bytes(df, diag):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Candidates", index=False)
        if not diag.empty:
            diag.to_excel(writer, sheet_name="Diagnostics", index=False)
    output.seek(0)
    return output.getvalue()

def build_signature(tickers, expiries, weekly_min, guard, overlays):
    sig = {
        "tickers": list(tickers),
        "expiries": list(expiries),
        "weekly_min": float(weekly_min),
        "MIN_PREMIUM_ABS": float(guard["MIN_PREMIUM_ABS"]),
        "USE_DELTA_BAND": bool(guard["USE_DELTA_BAND"]),
        "DELTA_MIN": float(guard["DELTA_MIN"]),
        "DELTA_MAX": float(guard["DELTA_MAX"]),
        "FALLBACK_MIN_OTM": float(guard["FALLBACK_MIN_OTM"]),
        "overlays": overlays
    }
    return json.dumps(sig, sort_keys=True)

# ---------- UI ----------
st.set_page_config(page_title="Wheel Screener — Simplified ROC", layout="wide")
st.title("Wheel Strategy Screener — Simplified (ROC only)")

if "wheel_results" not in st.session_state:
    st.session_state["wheel_results"] = None

with st.sidebar:
    st.subheader("Inputs")
    tickers_text = st.text_area("Tickers (comma-separated)",
                                value=",".join(DEFAULT_TICKERS), height=120)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    colA, colB = st.columns([1,1])
    with colA:
        refresh = st.button("Refresh expiries")
    with colB:
        clear_results = st.button("Clear results")

    if refresh:
        fetch_expiries_for_tickers.clear()

    if clear_results:
        st.session_state["wheel_results"] = None

    available_expiries = fetch_expiries_for_tickers(tickers) if tickers else []
    default_select = available_expiries[:2] if len(available_expiries) >= 2 else available_expiries
    expiries = st.multiselect("Available expiries", options=available_expiries, default=default_select)

    st.caption("Yield basis is ROC only: premium / strike.")
    weekly_min = st.number_input("Min Weekly Yield (%/wk)", value=WEEKLY_YIELD_MIN_DEFAULT, step=0.1, min_value=0.0)

    with st.expander("Guardrails & Debug"):
        min_prem_abs = st.number_input("Min premium ($)", value=0.10, step=0.05, min_value=0.0)
        use_delta = st.checkbox("Use delta band", value=False)
        dmin = st.slider("Delta min", 0.00, 0.80, 0.05, 0.01)
        dmax = st.slider("Delta max", 0.05, 0.90, 0.50, 0.01)
        fallback_min_otm = st.slider("Fallback min OTM when IV missing (%)", 0.0, 10.0, 2.0, 0.5) / 100.0
        diagnostics = st.checkbox("Capture diagnostics on run", value=False)

    # Updated: indicator toggles
    with st.expander("Chart overlays"):
        show_bb = st.checkbox("Bollinger Bands", value=False)
        show_rsi = st.checkbox("RSI (14)", value=False)
        show_macd = st.checkbox("MACD (12,26,9)", value=False)
        show_ma21 = st.checkbox("21d MA", value=False)
        show_ma50 = st.checkbox("50d MA", value=False)
        show_ma200 = st.checkbox("200d MA", value=False)

    with st.expander("Definitions"):
        st.markdown(
            """
**Wheel Stock Score**: 0–1 composite that rewards an uptrend, lower volatility, and smaller drawdowns.
**Comfort Score**: Weekly premium yield on a return-on-collateral basis (premium ÷ strike ÷ weeks to expiry × 100).
**Rank Score**: Comfort Score × Wheel Stock Score. Higher is generally better.
            """
        )

    run = st.button("Run Screener")

guard = {
    "MIN_PREMIUM_ABS": min_prem_abs,
    "USE_DELTA_BAND": use_delta,
    "DELTA_MIN": dmin,
    "DELTA_MAX": dmax,
    "FALLBACK_MIN_OTM": fallback_min_otm
}
overlay_sig = {
    "bb": bool(show_bb),
    "rsi": bool(show_rsi),
    "macd": bool(show_macd),
    "ma21": bool(show_ma21),
    "ma50": bool(show_ma50),
    "ma200": bool(show_ma200)
}
current_sig = build_signature(tickers, expiries, weekly_min, guard, overlay_sig)

if run:
    if not tickers:
        st.warning("Please enter at least one ticker.")
    elif not expiries:
        st.warning("Please pick at least one expiry.")
    else:
        try:
            with st.spinner("Fetching data and building candidates…"):
                df, diag = build_candidates(tickers, expiries, weekly_min, guard, diagnostics)
            if df.empty:
                st.error("No candidates met the filters. Consider relaxing guardrails.")
                st.session_state["wheel_results"] = None
            else:
                st.session_state["wheel_results"] = {
                    "df": df, "diag": diag,
                    "signature": current_sig,
                    "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "captured_diagnostics": bool(diagnostics)
                }
                st.success(f"Found {len(df)} option contracts.")
        except Exception as e:
            st.session_state["wheel_results"] = None
            st.error(f"Error: {e}")

results = st.session_state.get("wheel_results")

if results is not None:
    if results.get("signature") != current_sig:
        st.warning("Inputs changed since the last run. Press Run Screener to refresh. Showing last results.")

    col_cfg = {
        "Wheel Stock Score": st.column_config.NumberColumn(
            help="0–1 composite that rewards an uptrend, lower volatility, and smaller drawdowns."
        ),
        "Comfort Score": st.column_config.NumberColumn(
            help="Weekly premium yield on a return-on-collateral basis (premium ÷ strike ÷ weeks × 100)."
        ),
        "Rank Score": st.column_config.NumberColumn(
            help="Comfort Score × Wheel Stock Score."
        )
    }

    st.subheader("Candidates")
    st.dataframe(results["df"], use_container_width=True, hide_index=True, column_config=col_cfg)

    # Chart section
    st.subheader("Chart with chosen contract overlaid")
    mode = st.radio("Chart mode", ["Top-ranked per ticker", "Manual contract selector"], horizontal=True, key="chart_mode")
    df = results["df"]
    if mode == "Top-ranked per ticker":
        focus_ticker = st.selectbox("Focus ticker", options=sorted(df["Ticker"].unique()), key="focus_ticker")
        best_row = df[df["Ticker"] == focus_ticker].sort_values("Rank Score", ascending=False).iloc[0]
        dte = int(best_row["DTE"])
        strike = float(best_row["Strike"])
        net_strike = float(best_row["Net Strike"])
        st.caption(f"Top-ranked for {focus_ticker} (DTE={dte}, strike={strike:.2f}, net strike={net_strike:.2f})")
        make_chart(
            focus_ticker, dte, strike=strike, net_strike=net_strike,
            show_rsi=show_rsi, show_macd=show_macd, show_bb=show_bb,
            show_ma21=show_ma21, show_ma50=show_ma50, show_ma200=show_ma200
        )
    else:
        t_opts = sorted(df["Ticker"].unique())
        t_sel = st.selectbox("Ticker", options=t_opts, key="manual_ticker")
        df_t = df[df["Ticker"] == t_sel]
        e_opts = sorted(df_t["Expiry"].unique())
        e_sel = st.selectbox("Expiry", options=e_opts, key="manual_expiry")
        df_te = df_t[df_t["Expiry"] == e_sel]
        s_opts = sorted(df_te["Strike"].unique())
        s_sel = st.selectbox("Strike", options=s_opts, key="manual_strike")
        row = df_te[df_te["Strike"] == s_sel].sort_values("Rank Score", ascending=False).iloc[0]
        dte = int(row["DTE"])
        strike = float(row["Strike"])
        net_strike = float(row["Net Strike"])
        st.caption(f"Selected {t_sel} @ {e_sel}, strike={strike:.2f}, net strike={net_strike:.2f}")
        make_chart(
            t_sel, dte, strike=strike, net_strike=net_strike,
            show_rsi=show_rsi, show_macd=show_macd, show_bb=show_bb,
            show_ma21=show_ma21, show_ma50=show_ma50, show_ma200=show_ma200
        )

    if results.get("captured_diagnostics") and not results["diag"].empty:
        st.subheader("Diagnostics")
        st.dataframe(results["diag"], use_container_width=True, hide_index=True)
    elif st.checkbox("Show diagnostics table (last run)"):
        if results["diag"].empty:
            st.info("Diagnostics were not captured in the last run. Enable 'Capture diagnostics on run' and re-run.")

    xls = to_excel_bytes(results["df"], results["diag"])
    st.download_button("Download Excel", data=xls,
                       file_name="wheel_put_candidates_simplified.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Run the screener, then chart either the top-ranked contract per ticker or pick a specific contract via the selector.")
