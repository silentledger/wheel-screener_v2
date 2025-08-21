
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
from scipy.stats import norm
from datetime import datetime, date, timedelta
from io import BytesIO
import math
import warnings
import plotly.graph_objs as go
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

@st.cache_data(ttl=900, show_spinner=False)
def fetch_expiries_for_tickers(tickers):
    """Return a sorted set of expiries across all tickers using Yahoo's native lists.
       Includes a small delay and one retry to reduce 429s.
    """
    expiries = set()
    for t in tickers:
        if not t:
            continue
        try:
            tk = yf.Ticker(t)
            opts = tk.options or []
            expiries.update(opts)
        except Exception:
            # Retry once after a brief pause
            time.sleep(1.0)
            try:
                tk = yf.Ticker(t)
                opts = tk.options or []
                expiries.update(opts)
            except Exception:
                pass
        finally:
            time.sleep(0.4)  # gentle pacing to avoid rate limits
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

def make_chart(ticker, dte, strike=None, net_strike=None):
    lookback = choose_lookback_by_dte(dte or 30)
    hist = get_history(ticker, lookback)
    if hist.empty:
        return st.warning("No price history available to draw chart.")

    close = hist["Close"].copy()
    core_sup, second_sup, resistance = find_two_supports(close)
    ma21  = close.rolling(21).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        name="Price", showlegend=False
    ))

    # Moving averages
    fig.add_trace(go.Scatter(x=hist.index, y=ma21,  mode="lines", name="21d MA"))
    fig.add_trace(go.Scatter(x=hist.index, y=ma50,  mode="lines", name="50d MA"))
    if not ma200.dropna().empty:
        fig.add_trace(go.Scatter(x=hist.index, y=ma200, mode="lines", name="200d MA"))

    # Horizontal levels
    shapes = []
    annots = []
    def hline(y, label, color):
        if y is None: 
            return
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y,
                           line=dict(color=color, width=1.5, dash="dot")))
        annots.append(dict(x=1.002, xref="paper", y=y, yref="y",
                           text=label, showarrow=False, font=dict(size=10, color=color),
                           xanchor="left", bgcolor="rgba(255,255,255,0.5)"))

    hline(core_sup,   "Core support",  "#2ca02c")
    hline(second_sup, "2nd support",   "#98df8a")
    hline(resistance, "Resistance",    "#d62728")
    if strike is not None:
        hline(strike, "Strike",        "#1f77b4")
    if net_strike is not None:
        hline(net_strike, "Net strike", "#17becf")

    fig.update_layout(
        height=520,
        xaxis_title="Date", yaxis_title="Price",
        shapes=shapes, annotations=annots,
        margin=dict(l=40, r=120, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def _fill_last_price(puts: pd.DataFrame) -> pd.Series:
    lp = pd.to_numeric(puts.get("lastPrice"), errors="coerce")
    bid = pd.to_numeric(puts.get("bid"), errors="coerce")
    ask = pd.to_numeric(puts.get("ask"), errors="coerce")
    mid = (bid + ask) / 2.0
    # prefer lastPrice>0, else mid>0, else bid>0
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

        # Supports (kept minimal)
        h90 = tk.history(period="90d")["Close"]
        core_sup, second_sup, resistance = find_two_supports(h90)
        ma21 = float(h90.rolling(21).mean().iloc[-1]) if len(h90) >= 21 else None
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

            # Normalize columns and robust premium
            puts = puts.rename(columns={"impliedVolatility": "iv"})
            puts["lp"] = _fill_last_price(puts)
            puts = puts[(puts["lp"].notna()) & (puts["lp"] > 0) & (puts["strike"] < price)]
            total = len(puts)
            if total == 0:
                continue

            dte = (pd.to_datetime(expiry).date() - today).days

            # ---------- Robust IV hierarchy ----------
            iv_series = pd.to_numeric(puts["iv"], errors="coerce")
            iv_median = iv_series.median() if iv_series.notna().any() else np.nan
            iv_used = iv_series.fillna(iv_median)
            hist_60 = tk.history(period="60d")["Close"]
            hv60 = hist_60.pct_change().dropna().std() * np.sqrt(252) if len(hist_60) >= 30 else np.nan
            iv_used = iv_used.fillna(hv60)

            req_pct = iv_used.apply(lambda v: required_otm_pct(v, dte, fallback=guard["FALLBACK_MIN_OTM"]))
            otm_pct = (price - puts["strike"]) / price
            min_otm_ok = (otm_pct >= req_pct)

            # Yields (ROC only): premium / strike
            denom = puts["strike"]
            yld_pct = (puts["lp"] / denom) * 100.0
            wk_yld  = yld_pct / (dte/7.0 if dte > 0 else 1.0)
            monthly_yld = yld_pct * (DAYS_PER_MONTH / max(dte, 1))
            annual_yld  = yld_pct * (DAYS_PER_YEAR  / max(dte, 1))

            prem_ok  = puts["lp"] >= guard["MIN_PREMIUM_ABS"]
            wk_ok    = (wk_yld >= weekly_yield_min)

            # Optional delta band
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

    # Ranking
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

def build_signature(tickers, expiries, weekly_min, guard):
    sig = {
        "tickers": list(tickers),
        "expiries": list(expiries),
        "weekly_min": float(weekly_min),
        "MIN_PREMIUM_ABS": float(guard["MIN_PREMIUM_ABS"]),
        "USE_DELTA_BAND": bool(guard["USE_DELTA_BAND"]),
        "DELTA_MIN": float(guard["DELTA_MIN"]),
        "DELTA_MAX": float(guard["DELTA_MAX"]),
        "FALLBACK_MIN_OTM": float(guard["FALLBACK_MIN_OTM"]),
    }
    return json.dumps(sig, sort_keys=True)

# ---------- UI ----------
st.set_page_config(page_title="Wheel Screener — Simplified ROC", layout="wide")
st.title("Wheel Strategy Screener — Simplified (ROC only)")

# Session state init
if "wheel_results" not in st.session_state:
    st.session_state["wheel_results"] = None

with st.sidebar:
    st.subheader("Inputs")
    tickers_text = st.text_area("Tickers (comma-separated)",
                                value=",".join(DEFAULT_TICKERS), height=120)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    # Refresh expiries
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

    run = st.button("Run Screener")

# Build guard dict (no DTE window, no OI filter)
guard = {
    "MIN_PREMIUM_ABS": min_prem_abs,
    "USE_DELTA_BAND": use_delta,
    "DELTA_MIN": dmin,
    "DELTA_MAX": dmax,
    "FALLBACK_MIN_OTM": fallback_min_otm
}
current_sig = build_signature(tickers, expiries, weekly_min, guard)

# If user pressed Run, compute and persist results
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

# If we have prior results, show them regardless of radio toggles or other widget changes
results = st.session_state.get("wheel_results")

if results is not None:
    # Notify if inputs changed since last run
    if results.get("signature") != current_sig:
        st.warning("Inputs changed since the last run. Press Run Screener to refresh. Showing last results.")
    st.subheader("Candidates")
    st.dataframe(results["df"], use_container_width=True, hide_index=True)

    # --------- Contract selector and chart ---------
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
        make_chart(focus_ticker, dte, strike=strike, net_strike=net_strike)
    else:
        # Manual selection: Ticker -> Expiry -> Strike
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
        make_chart(t_sel, dte, strike=strike, net_strike=net_strike)

    # Diagnostics section
    if results.get("captured_diagnostics") and not results["diag"].empty:
        st.subheader("Diagnostics")
        st.dataframe(results["diag"], use_container_width=True, hide_index=True)
    elif st.checkbox("Show diagnostics table (last run)"):
        if results["diag"].empty:
            st.info("Diagnostics were not captured in the last run. Enable 'Capture diagnostics on run' and re-run.")

    # Download Excel from last results
    xls = to_excel_bytes(results["df"], results["diag"])
    st.download_button("Download Excel", data=xls,
                       file_name="wheel_put_candidates_simplified.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Run the screener, then chart either the top-ranked contract per ticker or pick a specific contract via the selector.")
