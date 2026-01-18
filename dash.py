# dashboard_app.py
# pip install streamlit pandas

import json
from pathlib import Path

import pandas as pd
import streamlit as st


REPORT_DIR_DEFAULT = r"report_outputs"

st.set_page_config(page_title="Model Backtest Dashboard", layout="wide")

st.sidebar.title("Backtest Dashboard")
report_dir = Path(st.sidebar.text_input("Report Directory", value=REPORT_DIR_DEFAULT)).expanduser()


def p(name: str) -> Path:
    return report_dir / name


def file_exists(name: str) -> bool:
    return p(name).exists()


st.sidebar.write("Directory exists:", report_dir.exists())

# Load daily
df_daily = None
if file_exists("daily_gross_net_equity.csv"):
    df_daily = pd.read_csv(p("daily_gross_net_equity.csv"))
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"], errors="coerce")
    df_daily = df_daily.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

# Date filter
if df_daily is not None and not df_daily.empty:
    d0, d1 = df_daily["trade_date"].min(), df_daily["trade_date"].max()
    sel = st.sidebar.date_input("Date Range", value=(d0.date(), d1.date()))
    if isinstance(sel, (list, tuple)) and len(sel) == 2:
        a, b = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])
        df_view = df_daily[(df_daily["trade_date"] >= a) & (df_daily["trade_date"] <= b)].copy()
    else:
        df_view = df_daily.copy()
else:
    df_view = None

show_gross = st.sidebar.checkbox("Show gross", value=True)
show_net = st.sidebar.checkbox("Show net", value=True)
# show_bench = st.sidebar.checkbox("Show benchmark", value=False)  # kept but default False

tabs = st.tabs(["Overview", "Equity/Drawdown", "IC", "Robustness", "Artifacts"])

# ===== Overview =====
with tabs[0]:
    st.header("Overview")

    # Model comparison
    if file_exists("model_comparison.csv"):
        df_cmp = pd.read_csv(p("model_comparison.csv"))
        st.subheader("Model Comparison")
        st.dataframe(df_cmp, width="stretch")
    else:
        st.info("model_comparison.csv not found (suggest saving df_cmp in the main script).")

    # KPI from daily
    if df_view is None or df_view.empty:
        st.warning("daily_gross_net_equity.csv not found or empty.")
    else:
        # Net return series
        r = df_view["net"].astype(float).fillna(0.0).values
        n = len(r)

        def sharpe(x, ann=252.0):
            s = pd.Series(x)
            sd = s.std(ddof=1)
            if sd <= 1e-12:
                return None
            return float(s.mean() / sd * (ann ** 0.5))

        def ann_ret(x, ann=252.0):
            eq = (1.0 + pd.Series(x)).cumprod()
            if len(eq) < 20 or eq.iloc[-1] <= 0:
                return None
            return float(eq.iloc[-1] ** (ann / len(eq)) - 1.0)

        def max_dd(x):
            eq = (1.0 + pd.Series(x)).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1.0
            return float(dd.min())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Samples (days)", f"{n}")
        k2.metric("Sharpe (net)", f"{sharpe(r):.4f}" if sharpe(r) is not None else "NA")
        k3.metric("Annual Return (net)", f"{ann_ret(r):.2%}" if ann_ret(r) is not None else "NA")
        k4.metric("Max Drawdown (net)", f"{max_dd(r):.2%}")
# ===== Equity/Drawdown =====
with tabs[1]:
    st.header("Equity / Drawdown")

    # Show pre-generated PNGs (fastest, minimal changes)
    # NOTE: benchmark plot is intentionally skipped.
    imgs = [
        ("equity_curve_gross_vs_net.png", "Equity (Gross vs Net)"),
        ("drawdown_net.png", "Drawdown (Net)"),
        ("rolling_sharpe_60d.png", "Rolling Sharpe (60d)"),
        ("rolling_sharpe_126d.png", "Rolling Sharpe (126d)"),
        ("rolling_sharpe_252d.png", "Rolling Sharpe (252d)"),
        ("turnover_series.png", "Turnover"),
        ("leverage_series.png", "Leverage"),
        ("risk_regime_timeline.png", "Risk Regime Timeline"),
    ]

    for fn, title in imgs:
        if file_exists(fn):
            st.subheader(title)
            st.image(str(p(fn)), width="stretch")


# ===== IC =====
with tabs[2]:
    st.header("IC / RankIC")

    if file_exists("ic_rankic_summary.json"):
        s = json.loads(p("ic_rankic_summary.json").read_text(encoding="utf-8"))
        st.subheader("IC Summary")
        st.json(s)

    if file_exists("ic_rankic_daily.csv"):
        ic = pd.read_csv(p("ic_rankic_daily.csv"))
        ic["trade_date"] = pd.to_datetime(ic["trade_date"], errors="coerce")
        ic = ic.dropna(subset=["trade_date"]).sort_values("trade_date")
        st.subheader("IC Daily")
        st.dataframe(ic.tail(200), width="stretch")
    else:
        st.info("ic_rankic_daily.csv not found.")

    if file_exists("ic_timeseries.png"):
        st.subheader("IC Time Series")
        st.image(str(p("ic_timeseries.png")), width="stretch")
    if file_exists("rankic_timeseries.png"):
        st.subheader("RankIC Time Series")
        st.image(str(p("rankic_timeseries.png")), width="stretch")

# ===== Robustness =====
with tabs[3]:
    st.header("Robustness / Sensitivity")

    files = [
        ("topk_robustness.csv", "TopK Robustness"),
        ("sample_period_sensitivity.csv", "Sample Period Sensitivity"),
        ("param_sensitivity_table.csv", "Param Sensitivity Table"),
        ("ablation_table.csv", "Ablation"),
        ("alignment_selfcheck.csv", "Alignment Self-check"),
        ("cost_sensitivity.csv", "Cost Sensitivity"),
    ]
    for fn, title in files:
        st.subheader(title)
        if file_exists(fn):
            st.dataframe(pd.read_csv(p(fn)), width="stretch")
        else:
            st.info(f"Missing: {fn}")

    if file_exists("cost_sensitivity_sharpe.png"):
        st.subheader("Cost Sensitivity Plot")
        st.image(str(p("cost_sensitivity_sharpe.png")), width="stretch")

# ===== Artifacts =====
with tabs[4]:
    st.header("Artifacts")

    if file_exists("backtest_report.html"):
        st.subheader("Backtest HTML Report")
        st.write("(HTML embedded below; if too large, provide download only.)")
        html = p("backtest_report.html").read_text(encoding="utf-8")
        st.components.v1.html(html, height=900, scrolling=True)
    else:
        st.info("Missing: backtest_report.html")

    st.subheader("File List")
    if report_dir.exists():
        items = sorted([x.name for x in report_dir.glob("*") if x.is_file()])
        st.write(items)


if __name__ == "__main__":
    # Auto-launch Streamlit only when running via "python dash.py",
    # and avoid creating Runtime twice when using "streamlit run dash.py".
    import sys
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        running_in_streamlit = get_script_run_ctx() is not None
    except Exception:
        running_in_streamlit = False

    if not running_in_streamlit:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        raise SystemExit(stcli.main())
