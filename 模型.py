# -*- coding: utf-8 -*-
import time
import json
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

try:
    import joblib
except Exception:
    from sklearn.externals import joblib  # 老 sklearn 兼容

# 绘图（不使用 seaborn）
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =========================
# 配置区
# =========================
DATA_PATH = r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features.csv"

OUT_ALPHA_OOF = r"D:\PythonFile\JCAI\data_ak\alpha_signal_oof.csv"
OUT_STACK_OOF = r"D:\PythonFile\JCAI\data_ak\alpha_stack_oof.csv"
OUT_ARTIFACT = r"D:\PythonFile\JCAI\data_ak\artifacts_oof.csv"
OUT_LASSO_LOG = r"D:\PythonFile\JCAI\data_ak\lasso_selected_log.csv"

# 报告输出目录
OUT_REPORT_DIR = r"D:\PythonFile\JCAI\data_ak\report_outputs"

# 缓存目录：首次训练写入；之后读取不再训练
CACHE_DIR = r"D:\PythonFile\JCAI\data_ak\model_cache_v5_tune_best"

ANN = 252.0
N_SPLITS = 5

# =========================
# LightGBM GPU（可选）
# =========================
USE_LGB_GPU = False
LGB_GPU_PLATFORM_ID = 0
LGB_GPU_DEVICE_ID = 0
LGB_MAX_BIN_GPU = 255

# =========================
# 固定 modernGB 默认参数（不再调 LightGBM 参数）
# =========================
MODERN_ALPHA_DEFAULT = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_samples": 50,
}

MODERN_RISK_DEFAULT = {
    "n_estimators": 350,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_samples": 50,
}

# 交易与成本
COST_BPS = 5.0
LONG_SHORT = False
CAP_WEIGHT = 0.25
POSITION_SMOOTH_ALPHA = 0.40
MAX_DAILY_TURNOVER = 0.30
REBALANCE_EVERY = 1

# =========================
# 回测集中持仓：每天只做 Top-N
# =========================
TOPK_HOLDINGS_CANDIDATES = [1, 2, 3]
TOPK_HOLDINGS_DEFAULT = 3
MIN_CROSS_SECTION = 3

# =========================
# Signal Integration
# =========================
VOL_GAMMA = 1.0
USE_PORTFOLIO_OVERLAY = True

TARGET_PORT_VOL_ANN = 0.12
PORT_VOL_WINDOW = 20
LEVERAGE_CAP = 2.0
LEVERAGE_FLOOR = 0.0

DD_THROTTLE_START = 0.08
DD_THROTTLE_END = 0.20

MIN_SPREAD = 0.002

# LASSO 特征选择
LASSO_ALPHAS = np.logspace(-4, -2, 9)
LASSO_INNER_SPLITS = 4
TOPK_FALLBACK = 60

# Regime（KMeans）
K_REGIME = 3
REGIME_MULT = {0: 1.00, 1: 0.70, 2: 0.45}

# stacking
RIDGE_ALPHA = 1.0

# =========================
# GARCH 风险模型（可选）
# =========================
USE_GARCH_RISK = True
GARCH_BLEND = 0.60
GARCH_MIN_OBS = 80
RET_CLIP = 0.20

# =========================
# 默认方向（统一乘在 alpha 上；不在打印中提“翻转”）
# =========================
SCORE_SIGN = 1.0  # 这里保持为 1.0；训练阶段会基于训练集相关性校准（calibrate_sign）

# =========================
# 自动调参（只调 ridge + RF）
# =========================
ENABLE_TUNE_BEST = True
TUNE_RANDOM_STATE = 42

TUNE_SPACE = {
    "ridge_alpha": [0.3, 1.0],
    "rf_n_estimators": [400, 600],
    "rf_min_samples_leaf": [10, 20],
    "rf_max_features": [0.5],
}

# =========================
# 工具
# =========================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _fmt_sec(sec: float) -> str:
    sec = float(sec)
    if (not np.isfinite(sec)) or sec < 0:
        return "--:--:--"
    s = int(sec)
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    return f"{h:02d}:{m:02d}:{s2:02d}"

class SimpleProgress:
    """
    无依赖进度条（适合 Windows 控制台）
    """
    def __init__(self, total: int, desc: str = ""):
        self.total = int(total) if total is not None else 0
        self.desc = str(desc)
        self.start = time.perf_counter()
        self.last = self.start
        self.done = 0

    def update(self, n: int = 1, extra: str = ""):
        self.done += int(n)
        now = time.perf_counter()
        elapsed = now - self.start
        step = now - self.last
        self.last = now

        if self.total > 0:
            pct = min(1.0, self.done / self.total)
            bar_w = 28
            fill = int(bar_w * pct)
            bar = "█" * fill + "·" * (bar_w - fill)

            rate = self.done / elapsed if elapsed > 1e-12 else 0.0
            eta = (self.total - self.done) / rate if rate > 1e-12 else np.nan

            msg = (f"\r[{bar}] {self.done}/{self.total} "
                   f"({pct*100:5.1f}%) "
                   f"elapsed={_fmt_sec(elapsed)} "
                   f"step={_fmt_sec(step)} "
                   f"eta={_fmt_sec(eta)}")
        else:
            msg = f"\relapsed={_fmt_sec(elapsed)} step={_fmt_sec(step)}"

        if self.desc:
            msg = f"{self.desc} " + msg
        if extra:
            msg += f" | {extra}"

        print(msg, end="", flush=True)

    def close(self, extra: str = ""):
        self.update(0, extra=extra)
        print("")

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def calibrate_sign(pred_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """
    用训练集相关性决定信号方向：如果 corr<0，就乘以 -1
    仅用训练数据做校准，不引入未来信息。
    """
    p = np.asarray(pred_tr, dtype=float)
    y = np.asarray(y_tr, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() < 50:
        return 1.0
    c = np.corrcoef(p[m], y[m])[0, 1]
    if (not np.isfinite(c)) or (c >= 0):
        return 1.0
    return -1.0

def sharpe_from_daily(daily, ann=252.0):
    daily = np.asarray(daily, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 20:
        return np.nan
    s = daily.std(ddof=1)
    if not np.isfinite(s) or s <= 1e-12:
        return np.nan
    return float(daily.mean() / s * np.sqrt(ann))

def ann_return_from_daily(daily, ann=252.0):
    daily = np.asarray(daily, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 20:
        return np.nan
    eq = np.cumprod(1.0 + daily)
    if eq[-1] <= 0:
        return np.nan
    return float(eq[-1] ** (ann / len(daily)) - 1.0)

def ann_vol_from_daily(daily, ann=252.0):
    daily = np.asarray(daily, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 20:
        return np.nan
    return float(daily.std(ddof=1) * np.sqrt(ann))

def max_drawdown_from_daily(daily):
    daily = np.asarray(daily, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 2:
        return np.nan
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())

def calmar_from_daily(daily, ann=252.0):
    ar = ann_return_from_daily(daily, ann=ann)
    mdd = max_drawdown_from_daily(daily)
    if not np.isfinite(ar) or not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    if abs(mdd) <= 1e-12:
        return np.nan
    return float(ar / abs(mdd))

def _select_top_bottom(g: pd.DataFrame, pred_col: str, topk: int, long_short: bool):
    if topk is None or int(topk) <= 0:
        return g
    topk = int(topk)
    gg = g.dropna(subset=[pred_col]).copy()
    if gg.empty:
        return gg

    gg = gg.sort_values(pred_col, ascending=False)
    if long_short:
        if len(gg) < 2 * topk:
            return gg.iloc[0:0]
        top = gg.head(topk)
        bot = gg.tail(topk)
        out = pd.concat([top, bot], axis=0)
        return out
    else:
        if len(gg) < topk:
            return gg.iloc[0:0]
        return gg.head(topk)

def weights_rank_longshort(g: pd.DataFrame, pred_col="pred", cap=0.3):
    if len(g) <= 1:
        p = float(g[pred_col].values[0]) if len(g) == 1 else 0.0
        return np.array([float(np.sign(p))], dtype=float)

    r = g[pred_col].rank(method="first")
    w = (r - r.mean()).values.astype(float)

    denom = np.sum(np.abs(w))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.zeros(len(g), dtype=float)

    w = w / denom
    w = np.clip(w, -float(cap), float(cap))

    denom2 = np.sum(np.abs(w))
    if not np.isfinite(denom2) or denom2 <= 1e-12:
        return np.zeros(len(g), dtype=float)

    return w / denom2

def weights_rank_longonly(g: pd.DataFrame, pred_col="pred", cap=0.3):
    if len(g) <= 1:
        return np.array([1.0], dtype=float)

    r = g[pred_col].rank(method="first").values.astype(float)
    w = r - np.min(r)

    denom = np.sum(w)
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.zeros(len(g), dtype=float)

    w = w / denom
    w = np.clip(w, 0.0, float(cap))

    denom2 = np.sum(w)
    if not np.isfinite(denom2) or denom2 <= 1e-12:
        return np.zeros(len(g), dtype=float)

    return w / denom2

def align_weights(cur_tickers, cur_w, prev_tickers, prev_w):
    cur = pd.Series(cur_w, index=cur_tickers)
    prev = pd.Series(prev_w, index=prev_tickers)
    all_tk = cur.index.union(prev.index)
    cur = cur.reindex(all_tk).fillna(0.0).values.astype(float)
    prev = prev.reindex(all_tk).fillna(0.0).values.astype(float)
    return all_tk.values, cur, prev

def one_way_turnover(cur, prev):
    return float(0.5 * np.sum(np.abs(cur - prev)))

def renorm_and_clip(w, cap, long_short=True):
    w = np.asarray(w, dtype=float)

    if w.size <= 1:
        if long_short:
            return np.clip(w, -float(cap), float(cap))
        return np.clip(w, 0.0, float(cap))

    if long_short:
        w = np.clip(w, -float(cap), float(cap))
        denom = np.sum(np.abs(w))
        if not np.isfinite(denom) or denom <= 1e-12:
            return w * 0.0
        return w / denom

    w = np.clip(w, 0.0, float(cap))
    denom = np.sum(w)
    if not np.isfinite(denom) or denom <= 1e-12:
        return w * 0.0
    return w / denom

def compute_portfolio_daily(
        df_in, pred_col="pred", ret_col="y_ret_k",
        long_short=True, cap=0.3,
        min_spread=0.0, cost_bps=5.0,
        pos_alpha=0.4, max_to=0.3,
        rebalance_every=1,
        regime_col=None, regime_mult=None,
        vol_pred_col=None,
        topk_holdings=0,
        min_cross_section=0
):
    gross, net, turnover = [], [], []
    prev_w, prev_tickers = None, None

    dates_sorted = list(df_in["trade_date"].dropna().sort_values().unique())
    date_to_step = {d: i for i, d in enumerate(dates_sorted)}

    for dt, g0 in df_in.groupby("trade_date", sort=False):
        g = g0[["ticker", pred_col, ret_col]].dropna().copy()
        if len(g) == 0:
            gross.append(0.0)
            net.append(0.0)
            turnover.append(0.0)
            continue

        if min_cross_section and len(g) < int(min_cross_section):
            if prev_w is not None:
                cur_tk = prev_tickers
                cur_w = prev_w
                r_map = pd.Series(g[ret_col].astype(float).values, index=g["ticker"].values)
                r_aligned = r_map.reindex(cur_tk).fillna(0.0).values.astype(float)
                g_ret = float(np.dot(cur_w, r_aligned))
                gross.append(g_ret)
                net.append(g_ret)
                turnover.append(0.0)
            else:
                gross.append(0.0)
                net.append(0.0)
                turnover.append(0.0)
            continue

        step = date_to_step.get(dt, 0)
        do_rebalance = (rebalance_every is None) or (rebalance_every <= 1) or (step % int(rebalance_every) == 0)

        if (not do_rebalance) and (prev_w is not None):
            cur_tk = prev_tickers
            cur_w = prev_w
            r_map = pd.Series(g[ret_col].astype(float).values, index=g["ticker"].values)
            r_aligned = r_map.reindex(cur_tk).fillna(0.0).values.astype(float)
            g_ret = float(np.dot(cur_w, r_aligned))
            gross.append(g_ret)
            net.append(g_ret)
            turnover.append(0.0)
            continue

        if min_spread and min_spread > 0:
            sp = float(np.nanmax(g[pred_col].values) - np.nanmin(g[pred_col].values))
            if sp < float(min_spread) and prev_w is not None:
                cur_tk = prev_tickers
                cur_w = prev_w
                r_map = pd.Series(g[ret_col].astype(float).values, index=g["ticker"].values)
                r_aligned = r_map.reindex(cur_tk).fillna(0.0).values.astype(float)
                g_ret = float(np.dot(cur_w, r_aligned))
                gross.append(g_ret)
                net.append(g_ret)
                turnover.append(0.0)
                continue

        g_sel = _select_top_bottom(g, pred_col=pred_col, topk=topk_holdings, long_short=long_short)
        if g_sel is None or g_sel.empty:
            if prev_w is not None:
                cur_tk = prev_tickers
                cur_w = prev_w
                r_map = pd.Series(g[ret_col].astype(float).values, index=g["ticker"].values)
                r_aligned = r_map.reindex(cur_tk).fillna(0.0).values.astype(float)
                g_ret = float(np.dot(cur_w, r_aligned))
                gross.append(g_ret)
                net.append(g_ret)
                turnover.append(0.0)
            else:
                gross.append(0.0)
                net.append(0.0)
                turnover.append(0.0)
            continue

        if long_short:
            w_target = weights_rank_longshort(g_sel, pred_col=pred_col, cap=cap)
        else:
            w_target = weights_rank_longonly(g_sel, pred_col=pred_col, cap=cap)

        if vol_pred_col is not None and (vol_pred_col in g0.columns):
            vp = g0.set_index("ticker")[vol_pred_col].reindex(g_sel["ticker"].values).fillna(0.0).values.astype(float)
            p95 = np.nanpercentile(vp, 95) if np.isfinite(np.nanpercentile(vp, 95)) else 1.0
            vp = np.clip(vp, 1e-6, p95 if p95 > 1e-6 else 1.0)
            w_target = w_target * (1.0 / vp)

        if regime_col is not None and (regime_col in g0.columns) and (regime_mult is not None):
            r_ids = g0.set_index("ticker")[regime_col].reindex(g_sel["ticker"].values).fillna(0).values.astype(int)
            mult = np.array([float(regime_mult.get(int(x), 1.0)) for x in r_ids], dtype=float)
            w_target = w_target * mult

        if prev_w is None:
            w_exec = renorm_and_clip(w_target, cap=cap, long_short=long_short)
            to = float(np.sum(np.abs(w_exec)))
            cost = float(to * (cost_bps / 10000.0))
            r = g_sel[ret_col].astype(float).values
            g_ret = float(np.dot(w_exec, r))
            gross.append(g_ret)
            net.append(g_ret - cost)
            turnover.append(to)
            prev_w = w_exec
            prev_tickers = g_sel["ticker"].values
            continue

        all_tk, cur, prev = align_weights(
            cur_tickers=g_sel["ticker"].values, cur_w=w_target,
            prev_tickers=prev_tickers, prev_w=prev_w
        )

        w_sm = (1.0 - float(pos_alpha)) * prev + float(pos_alpha) * cur
        w_sm = renorm_and_clip(w_sm, cap=cap, long_short=long_short)

        to_sm = one_way_turnover(w_sm, prev)
        if np.isfinite(to_sm) and to_sm > float(max_to) and float(max_to) > 0:
            delta = w_sm - prev
            scale = float(max_to) / float(to_sm)
            w_sm = prev + delta * scale
            w_sm = renorm_and_clip(w_sm, cap=cap, long_short=long_short)
            to_sm = one_way_turnover(w_sm, prev)

        r_map = pd.Series(g_sel[ret_col].astype(float).values, index=g_sel["ticker"].values)
        r_aligned = r_map.reindex(all_tk).fillna(0.0).values.astype(float)
        g_ret = float(np.dot(w_sm, r_aligned))

        cost = float(to_sm * (cost_bps / 10000.0))
        gross.append(g_ret)
        net.append(g_ret - cost)
        turnover.append(to_sm)

        prev_w = w_sm
        prev_tickers = all_tk

    return np.asarray(gross, float), np.asarray(net, float), np.asarray(turnover, float)

# =========================
# GARCH/EWMA 风险预测
# =========================
def _ewma_vol_forecast(r: np.ndarray, lam=0.94):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return np.nan
    v = np.nanvar(r, ddof=1) if np.isfinite(np.nanvar(r, ddof=1)) else 0.0
    for x in r:
        v = lam * v + (1.0 - lam) * (x * x)
    return float(np.sqrt(max(v, 1e-12)))

def fit_predict_garch_vol_by_ticker(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                                   ret_hist_col="_ret_hist",
                                   min_obs=80,
                                   clip=0.20):
    try:
        from arch import arch_model
        has_arch = True
    except Exception:
        has_arch = False

    df_te = df_te.reset_index(drop=True)
    te_out = np.full(len(df_te), np.nan, dtype=float)

    te_groups = df_te.groupby("ticker", sort=False)
    tr_groups = df_tr.groupby("ticker", sort=False)

    for tk, te_g in te_groups:
        te_rows = te_g.index.values.astype(int)
        if tk not in tr_groups.groups:
            continue

        r = tr_groups.get_group(tk)[ret_hist_col].values.astype(float)
        r = r[np.isfinite(r)]
        if r.size < min_obs:
            te_out[te_rows] = _ewma_vol_forecast(np.clip(r, -clip, clip))
            continue

        r = np.clip(r, -clip, clip)

        if has_arch:
            try:
                r_pct = r * 100.0
                am = arch_model(r_pct, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
                res = am.fit(disp="off", show_warning=False)
                fc = res.forecast(horizon=1, reindex=False)
                v1 = float(fc.variance.values[-1, 0])
                sig = np.sqrt(max(v1, 1e-12)) / 100.0
                te_out[te_rows] = float(sig)
            except Exception:
                te_out[te_rows] = _ewma_vol_forecast(r)
        else:
            te_out[te_rows] = _ewma_vol_forecast(r)

    return te_out

# =========================
# 防泄漏缺失填充（每折仅用训练统计）
# =========================
def compute_train_fill_stats(df_tr: pd.DataFrame, cols, by="ticker"):
    cols = [c for c in cols if c in df_tr.columns]
    cols_num = [c for c in cols if df_tr[c].dtype.kind in "biufc"]
    if len(cols_num) == 0:
        return {"cols_num": [], "global_median": {}, "by_median": {}}

    gmed = df_tr[cols_num].median(numeric_only=True).to_dict()
    by_med = df_tr.groupby(by, sort=False)[cols_num].median(numeric_only=True)
    by_med_dict = {c: by_med[c].to_dict() for c in cols_num}
    return {"cols_num": cols_num, "global_median": gmed, "by_median": by_med_dict}

def apply_fill_stats(df_: pd.DataFrame, stats, by="ticker"):
    cols_num = stats.get("cols_num", [])
    gmed = stats.get("global_median", {})
    by_med_dict = stats.get("by_median", {})

    out = df_.copy()
    if len(cols_num) == 0:
        return out

    tk = out[by]
    for c in cols_num:
        mp = by_med_dict.get(c, {})
        m = tk.map(mp).astype(float)
        m = m.fillna(float(gmed.get(c, 0.0)))
        out[c] = out[c].astype(float).fillna(m).fillna(float(gmed.get(c, 0.0)))
    return out

# =========================
# LASSO 选特征
# =========================
def lasso_select_features(X_tr, y_tr, dates_tr, alphas, inner_splits=4, topk=60):
    uniq_dates = np.array(sorted(pd.Series(dates_tr).dropna().unique()))
    if len(uniq_dates) < (inner_splits + 2):
        corr = np.nan_to_num(np.corrcoef(X_tr, y_tr, rowvar=False)[-1, :-1], nan=0.0)
        idx = np.argsort(np.abs(corr))[::-1][:topk]
        return idx, None

    tscv = TimeSeriesSplit(n_splits=inner_splits)
    best_a, best_mse = None, 1e18

    for a in alphas:
        mses = []
        for tr_didx, va_didx in tscv.split(uniq_dates):
            tr_dates = uniq_dates[tr_didx]
            va_dates = uniq_dates[va_didx]
            tr_mask = np.isin(dates_tr, tr_dates)
            va_mask = np.isin(dates_tr, va_dates)
            if tr_mask.sum() < 50 or va_mask.sum() < 50:
                continue
            mdl = Lasso(alpha=float(a), max_iter=20000, random_state=42)
            mdl.fit(X_tr[tr_mask], y_tr[tr_mask])
            pred = mdl.predict(X_tr[va_mask])
            mses.append(mean_squared_error(y_tr[va_mask], pred))
        if len(mses) == 0:
            continue
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse = m
            best_a = float(a)

    if best_a is None:
        corr = np.nan_to_num(np.corrcoef(X_tr, y_tr, rowvar=False)[-1, :-1], nan=0.0)
        idx = np.argsort(np.abs(corr))[::-1][:topk]
        return idx, None

    mdl = Lasso(alpha=best_a, max_iter=20000, random_state=42)
    mdl.fit(X_tr, y_tr)
    nz = np.where(np.abs(mdl.coef_) > 1e-12)[0]

    if len(nz) == 0:
        corr = np.nan_to_num(np.corrcoef(X_tr, y_tr, rowvar=False)[-1, :-1], nan=0.0)
        idx = np.argsort(np.abs(corr))[::-1][:topk]
        return idx, best_a

    return nz, best_a

# =========================
# Regime: KMeans + 风险排序映射
# =========================
def fit_kmeans_regime(train_df, regime_cols):
    Xtr = train_df[regime_cols].values.astype(float)
    km = KMeans(n_clusters=K_REGIME, n_init=10, random_state=42)
    km.fit(Xtr)

    centers = km.cluster_centers_
    col_to_i = {c: i for i, c in enumerate(regime_cols)}

    def pick_first(names):
        for n in names:
            if n in col_to_i:
                return centers[:, col_to_i[n]]
        return np.zeros(len(centers), dtype=float)

    c_rv = pick_first(["rv_20_L1_z120", "rv_20_L1_z60"])
    c_turn = pick_first(["log_turn_L1_z120", "log_turn_L1_z60"])
    c_spread = pick_first([
        "spread_rel_L1_z120", "spread_rel_L1_z60",
        "range_hl_L1_z120", "range_hl_L1_z60",
        "bb_bw_L1_z120", "bb_bw_L1_z60",
    ])
    c_ofi = pick_first(["ofi_L1_z120", "ofi_L1_z60"])

    risk_score = c_rv + c_spread + 0.25 * np.abs(c_ofi) - c_turn
    order = np.argsort(risk_score)
    mapping = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    return km, mapping

def predict_regime(km, mapping, df_te, regime_cols):
    Xte = df_te[regime_cols].values.astype(float)
    te_cluster = km.predict(Xte)
    te_regime = np.vectorize(lambda x: mapping.get(int(x), 1))(te_cluster).astype(int)
    return te_regime

# =========================
# 模型构造（modern GB 优先 LGBM/XGB）
# =========================
def make_gb_modern(task="alpha", random_state=42):
    defaults = MODERN_ALPHA_DEFAULT if task == "alpha" else MODERN_RISK_DEFAULT

    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=int(defaults["n_estimators"]),
            learning_rate=float(defaults["learning_rate"]),
            num_leaves=int(defaults["num_leaves"]),
            max_depth=int(defaults["max_depth"]),
            subsample=float(defaults["subsample"]),
            colsample_bytree=float(defaults["colsample_bytree"]),
            min_child_samples=int(defaults["min_child_samples"]),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            device_type=("gpu" if USE_LGB_GPU else "cpu"),
            gpu_platform_id=int(LGB_GPU_PLATFORM_ID),
            gpu_device_id=int(LGB_GPU_DEVICE_ID),
            max_bin=int(LGB_MAX_BIN_GPU),
        )
    except Exception:
        pass

    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=int(defaults["n_estimators"]),
            learning_rate=float(defaults["learning_rate"]),
            max_depth=int(defaults["max_depth"]),
            subsample=float(defaults["subsample"]),
            colsample_bytree=float(defaults["colsample_bytree"]),
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    except Exception:
        pass

    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=int(defaults["n_estimators"]),
        learning_rate=float(defaults["learning_rate"]),
        max_depth=int(defaults["max_depth"]),
        subsample=float(defaults["subsample"]),
        random_state=random_state
    )
def gb_backend_tag(model) -> str:
    """
    返回具体使用的 GB 后端与类名：
      - LightGBM::LGBMRegressor
      - XGBoost::XGBRegressor
      - sklearn::GradientBoostingRegressor
    """
    if model is None:
        return "unknown"
    mod = (model.__class__.__module__ or "").lower()
    cls = model.__class__.__name__
    if "lightgbm" in mod:
        return f"LightGBM::{cls}"
    if "xgboost" in mod:
        return f"XGBoost::{cls}"
    return f"sklearn::{cls}"

def gb_backend_tag(model) -> str:
    """
    返回具体使用的 GB 后端与类名：
      - LightGBM::LGBMRegressor
      - XGBoost::XGBRegressor
      - sklearn::GradientBoostingRegressor
    """
    if model is None:
        return "unknown"
    mod = (model.__class__.__module__ or "").lower()
    cls = model.__class__.__name__
    if "lightgbm" in mod:
        return f"LightGBM::{cls}"
    if "xgboost" in mod:
        return f"XGBoost::{cls}"
    return f"sklearn::{cls}"


def make_gbdt_sklearn(task="alpha", random_state=42):
    from sklearn.ensemble import GradientBoostingRegressor
    if task == "alpha":
        return GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=random_state
        )
    return GradientBoostingRegressor(
        n_estimators=700, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=random_state
    )


def make_risk_regressor(random_state=42):
    return make_gb_modern(task="risk", random_state=random_state)

# =========================
# 报告/绘图：核心函数
# =========================
# =========================
# 报告/绘图：核心函数
# =========================
def _format_date_xaxis(ax):
    """
    解决日期横轴过密：自动刻度 + 简洁格式 + 旋转
    """
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_horizontalalignment("right")

def equity_curve(daily_ret: np.ndarray):
    r = np.asarray(daily_ret, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    return np.cumprod(1.0 + r)

def daily_stats(daily_ret: np.ndarray):
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {}
    s = pd.Series(r)
    out = {
        "n": int(r.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if r.size > 1 else np.nan,
        "skew": float(s.skew()) if r.size > 2 else np.nan,
        "kurt": float(s.kurtosis()) if r.size > 3 else np.nan,
        "min": float(s.min()),
        "max": float(s.max()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "q50": float(s.quantile(0.50)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
    }
    q05 = out["q05"]
    tail = r[r <= q05]
    out["cvar05"] = float(np.mean(tail)) if tail.size > 0 else np.nan
    return out

def period_perf_table(daily_df: pd.DataFrame, ret_col: str, freq: str, ann=252.0):
    x = daily_df[[ret_col]].copy()
    x = x.sort_index()
    grp = x.groupby(pd.Grouper(freq=freq))
    out_rows = []
    for dt, g in grp:
        if g.empty:
            continue
        r = g[ret_col].values.astype(float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        period_ret = float(np.prod(1.0 + r) - 1.0)
        sh = sharpe_from_daily(r, ann=ann)
        mdd = max_drawdown_from_daily(r)
        out_rows.append({
            "period": dt,
            "n_days": int(r.size),
            "return": period_ret,
            "sharpe": float(sh) if np.isfinite(sh) else np.nan,
            "maxdd": float(mdd) if np.isfinite(mdd) else np.nan,
        })
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["period"] = pd.to_datetime(out["period"])
    return out

def compute_ic_rankic(df_oof: pd.DataFrame, alpha_col="alpha", ret_col="y_ret_k", min_n=2):
    rows = []
    for dt, g in df_oof.groupby("trade_date", sort=False):
        gg = g[[alpha_col, ret_col]].dropna()
        if len(gg) < int(min_n):
            rows.append({"trade_date": dt, "ic": np.nan, "rankic": np.nan, "n": int(len(gg))})
            continue
        ic = gg[alpha_col].corr(gg[ret_col], method="pearson")
        ric = gg[alpha_col].corr(gg[ret_col], method="spearman")
        rows.append({"trade_date": dt, "ic": float(ic) if np.isfinite(ic) else np.nan,
                     "rankic": float(ric) if np.isfinite(ric) else np.nan,
                     "n": int(len(gg))})
    out = pd.DataFrame(rows).sort_values("trade_date")
    out["ic_rolling20"] = out["ic"].rolling(20, min_periods=5).mean()
    out["rankic_rolling20"] = out["rankic"].rolling(20, min_periods=5).mean()

    ic_s = out["ic"].dropna().values
    ric_s = out["rankic"].dropna().values

    def _ir(x):
        if x.size < 20:
            return np.nan
        sd = np.std(x, ddof=1)
        if not np.isfinite(sd) or sd <= 1e-12:
            return np.nan
        return float(np.mean(x) / sd * np.sqrt(ANN))

    summary = {
        "ic_mean": float(np.nanmean(ic_s)) if ic_s.size else np.nan,
        "ic_std": float(np.nanstd(ic_s, ddof=1)) if ic_s.size > 1 else np.nan,
        "ic_ir": _ir(ic_s),
        "rankic_mean": float(np.nanmean(ric_s)) if ric_s.size else np.nan,
        "rankic_std": float(np.nanstd(ric_s, ddof=1)) if ric_s.size > 1 else np.nan,
        "rankic_ir": _ir(ric_s),
        "days": int(out["trade_date"].nunique())
    }
    return out, summary
def plot_equity(dates, gross, net, out_png):
    eq_g = equity_curve(gross)
    eq_n = equity_curve(net)
    plt.figure()
    plt.plot(dates, eq_g, label="gross")
    plt.plot(dates, eq_n, label="net")
    plt.title("Equity Curve (Gross vs Net)")
    plt.legend()
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def compute_equal_weight_benchmark(df_all: pd.DataFrame, ret_col="y_ret_k"):
    """
    等权基准：每天对全市场(样本内ticker集合)等权平均收益
    返回：DataFrame(trade_date, bench_ret)
    """
    g = df_all[["trade_date", ret_col]].dropna().copy()
    if g.empty:
        return pd.DataFrame(columns=["trade_date", "bench_ret"])
    out = g.groupby("trade_date", sort=True)[ret_col].mean().reset_index()
    out = out.rename(columns={ret_col: "bench_ret"})
    return out

def plot_equity_vs_benchmark(dates, net, bench_ret, out_png):
    net = np.asarray(net, dtype=float)
    bench_ret = np.asarray(bench_ret, dtype=float)
    eq_n = equity_curve(net)
    eq_b = equity_curve(bench_ret)
    plt.figure()
    plt.plot(dates, eq_n, label="strategy_net")
    plt.plot(dates, eq_b, label="benchmark_eqw")
    plt.title("Equity Curve (Net vs Equal-Weight Benchmark)")
    plt.legend()
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_series(dates, y, out_png, title, ylabel=None):
    y = np.asarray(y, dtype=float)
    plt.figure()
    plt.plot(dates, y)
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def sample_period_sensitivity(df_daily: pd.DataFrame, out_dir: str, ann=252.0):
    """
    样本期间敏感性：不同起始/不同近N年/近6个月
    输出：
      - sample_period_sensitivity.csv
      - rolling_sharpe_126d.png, rolling_sharpe_252d.png（若未生成可复用现有绘图）
    """
    ensure_dir(out_dir)
    d = df_daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d = d.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    if d.empty or "net" not in d.columns:
        out_csv = str(Path(out_dir) / "sample_period_sensitivity.csv")
        pd.DataFrame([]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out_csv

    end_dt = d["trade_date"].max()

    def _eval(sub: pd.DataFrame):
        r = sub["net"].values.astype(float)
        return {
            "n_days": int(np.isfinite(r).sum()),
            "sharpe_net": sharpe_from_daily(r, ann=ann),
            "ann_ret_net": ann_return_from_daily(r, ann=ann),
            "ann_vol_net": ann_vol_from_daily(r, ann=ann),
            "mdd_net": max_drawdown_from_daily(r),
            "calmar_net": calmar_from_daily(r, ann=ann),
        }

    specs = [
        ("full", None),
        ("last_3y", end_dt - pd.DateOffset(years=3)),
        ("last_2y", end_dt - pd.DateOffset(years=2)),
        ("last_1y", end_dt - pd.DateOffset(years=1)),
        ("last_6m", end_dt - pd.DateOffset(months=6)),
    ]

    rows = []
    for name, start_dt in specs:
        if start_dt is None:
            sub = d
        else:
            sub = d[d["trade_date"] >= start_dt].copy()
        m = _eval(sub)
        rows.append({"period": name, "start": (sub["trade_date"].min() if len(sub) else pd.NaT), "end": end_dt, **m})

    out = pd.DataFrame(rows)
    out_csv = str(Path(out_dir) / "sample_period_sensitivity.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 滚动稳定性（126D/252D）
    r = d["net"].values.astype(float)
    rs_126, _ = rolling_metrics(r, window=126, ann=ann)
    rs_252, _ = rolling_metrics(r, window=252, ann=ann)

    plot_rolling(d["trade_date"], rs_126, str(Path(out_dir) / "rolling_sharpe_126d.png"), "Rolling Sharpe (126d, Net)")
    plot_rolling(d["trade_date"], rs_252, str(Path(out_dir) / "rolling_sharpe_252d.png"), "Rolling Sharpe (252d, Net)")

    return out_csv

def generate_backtest_report(out_dir: str,
                            main_name: str,
                            tuned_params: dict,
                            best_topk: int,
                            df_cmp: pd.DataFrame,
                            ic_sum: dict):
    """
    输出：backtest_report.html（把关键表格/指标/图片链接在一起）
    """
    ensure_dir(out_dir)
    html_path = Path(out_dir) / "backtest_report.html"

    # 关键文件（相对路径，方便直接打开）
    imgs = [
        ("Equity Gross vs Net", "equity_curve_gross_vs_net.png"),
        ("Equity vs Benchmark", "equity_vs_benchmark.png"),
        ("Drawdown Net", "drawdown_net.png"),
        ("Rolling Sharpe 60D", "rolling_sharpe_60d.png"),
        ("Rolling Sharpe 126D", "rolling_sharpe_126d.png"),
        ("Rolling Sharpe 252D", "rolling_sharpe_252d.png"),
        ("Turnover", "turnover_series.png"),
        ("Leverage", "leverage_series.png"),
        ("IC", "ic_timeseries.png"),
        ("RankIC", "rankic_timeseries.png"),
        ("Cost Sensitivity", "cost_sensitivity_sharpe.png"),
    ]

    tables = [
        ("Model Comparison (Top 20)", df_cmp.head(20) if df_cmp is not None else pd.DataFrame()),
    ]

    def _img_tag(title, fn):
        p = Path(out_dir) / fn
        if not p.exists():
            return f"<h3>{title}</h3><p>(missing: {fn})</p>"
        return f'<h3>{title}</h3><img src="{fn}" style="max-width:1200px;width:100%;border:1px solid #ddd;">'

    parts = []
    parts.append("<html><head><meta charset='utf-8'>"
                 "<title>Backtest Report</title>"
                 "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;} "
                 "table{border-collapse:collapse;} th,td{border:1px solid #ccc;padding:6px 10px;} "
                 "h1{margin-top:0;} .mono{font-family:Consolas,Menlo,monospace;}</style>"
                 "</head><body>")

    parts.append("<h1>Backtest Report</h1>")
    parts.append(f"<p><b>Main Model</b>: <span class='mono'>{main_name}</span></p>")
    parts.append(f"<p><b>Best TopK Holdings</b>: {int(best_topk)}</p>")
    parts.append("<h2>Key Summary</h2>")
    parts.append("<pre class='mono'>" + json.dumps({
        "tuned_params": tuned_params,
        "ic_rankic_summary": ic_sum
    }, ensure_ascii=False, indent=2) + "</pre>")

    parts.append("<h2>Charts</h2>")
    for title, fn in imgs:
        parts.append(_img_tag(title, fn))

    parts.append("<h2>Tables</h2>")
    for title, tdf in tables:
        parts.append(f"<h3>{title}</h3>")
        if tdf is None or len(tdf) == 0:
            parts.append("<p>(empty)</p>")
        else:
            parts.append(tdf.to_html(index=False, float_format=lambda x: f"{x:.6g}"))

    parts.append("<h2>Raw Outputs</h2>")
    outputs = [
        "daily_gross_net_equity.csv",
        "monthly_performance.csv",
        "quarterly_performance.csv",
        "stability_by_year.csv",
        "stability_by_regime.csv",
        "cost_sensitivity.csv",
        "ic_rankic_daily.csv",
        "param_sensitivity_table.csv",
        "ablation_table.csv",
        "alignment_selfcheck.csv",
        "topk_robustness.csv",
        "sample_period_sensitivity.csv",
    ]
    parts.append("<ul>")
    for fn in outputs:
        if (Path(out_dir) / fn).exists():
            parts.append(f'<li><a href="{fn}">{fn}</a></li>')
        else:
            parts.append(f"<li>{fn} (missing)</li>")
    parts.append("</ul>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return str(html_path)


def plot_hist(daily_ret, title, out_png, bins=50):
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    plt.figure()
    plt.hist(r, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_cost_sensitivity(cost_df: pd.DataFrame, out_png: str):
    plt.figure()
    plt.plot(cost_df["cost_bps"], cost_df["sharpe_net"], marker="o")
    plt.title("Cost Sensitivity (Sharpe_net vs cost_bps)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_ic_series(ic_df: pd.DataFrame, out_png: str):
    plt.figure()
    plt.plot(ic_df["trade_date"], ic_df["ic"], label="IC")
    plt.plot(ic_df["trade_date"], ic_df["ic_rolling20"], label="IC_rolling20")
    plt.title("IC Time Series")
    plt.legend()
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_rankic_series(ic_df: pd.DataFrame, out_png: str):
    plt.figure()
    plt.plot(ic_df["trade_date"], ic_df["rankic"], label="RankIC")
    plt.plot(ic_df["trade_date"], ic_df["rankic_rolling20"], label="RankIC_rolling20")
    plt.title("RankIC Time Series")
    plt.legend()
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def drawdown_series(daily_ret: np.ndarray):
    eq = equity_curve(daily_ret)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak <= 0, 1.0, peak) - 1.0
    return dd

def plot_drawdown(dates, daily_ret, out_png, title="Drawdown Curve"):
    dd = drawdown_series(daily_ret)
    plt.figure()
    plt.plot(dates, dd)
    plt.title(title)
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def rolling_metrics(daily_ret: np.ndarray, window=60, ann=252.0):
    r = pd.Series(np.asarray(daily_ret, dtype=float))
    mu = r.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = r.rolling(window, min_periods=max(10, window // 3)).std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(ann)
    vol = sd * np.sqrt(ann)
    return sharpe.values.astype(float), vol.values.astype(float)

def plot_rolling(dates, y, out_png, title):
    plt.figure()
    plt.plot(dates, y)
    plt.title(title)
    _format_date_xaxis(plt.gca())
    plt.tight_layout()

    plt.savefig(out_png, dpi=150)
    plt.close()

def apply_signal_integration(df_trade: pd.DataFrame, alpha_col="alpha", vol_col="vol_pred", gamma=1.0):
    out = df_trade.copy()
    v = out[vol_col].astype(float).values
    p95 = np.nanpercentile(v, 95) if np.isfinite(np.nanpercentile(v, 95)) else 1.0
    v = np.clip(v, 1e-6, p95 if p95 > 1e-6 else 1.0)
    out["integrated_score"] = out[alpha_col].astype(float).values / (v ** float(gamma))
    return out

def apply_portfolio_overlay(daily_gross: np.ndarray, daily_net: np.ndarray,
                            target_vol_ann=0.08, window=20,
                            lev_cap=2.0, lev_floor=0.0,
                            dd_start=0.06, dd_end=0.15,
                            ann=252.0):
    g = np.asarray(daily_gross, dtype=float)
    n = np.asarray(daily_net, dtype=float)

    s = pd.Series(n)
    rv = s.rolling(window, min_periods=max(10, window // 2)).std(ddof=1) * np.sqrt(ann)
    rv = rv.shift(1)
    lev = (float(target_vol_ann) / rv).replace([np.inf, -np.inf], np.nan).fillna(1.0).values.astype(float)
    lev = np.clip(lev, float(lev_floor), float(lev_cap))

    eq = equity_curve(n)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak <= 0, 1.0, peak) - 1.0
    dd_depth = -dd

    throttle = np.ones_like(lev, dtype=float)
    a = float(dd_start)
    b = float(dd_end)
    if b > a > 0:
        x = (dd_depth - a) / (b - a)
        x = np.clip(x, 0.0, 1.0)
        throttle = 1.0 - x

    lev2 = np.clip(lev * throttle, float(lev_floor), float(lev_cap))
    return g * lev2, n * lev2, lev2

def signal_quantile_spread(df_trade: pd.DataFrame, score_col="integrated_score", ret_col="y_ret_k", q=10):
    rows = []
    for dt, g in df_trade.groupby("trade_date", sort=False):
        gg = g[[score_col, ret_col]].dropna()
        if len(gg) < q:
            continue
        gg = gg.sort_values(score_col)
        gg["bucket"] = pd.qcut(gg[score_col], q, labels=False, duplicates="drop")
        if gg["bucket"].nunique() < 2:
            continue
        top = gg.loc[gg["bucket"] == gg["bucket"].max(), ret_col].mean()
        bot = gg.loc[gg["bucket"] == gg["bucket"].min(), ret_col].mean()
        rows.append({"trade_date": dt, "spread": float(top - bot)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["trade_date", "spread"])
    out = out.sort_values("trade_date")
    return out

def run_param_sensitivity(df_trade_base: pd.DataFrame, out_dir: str):
    grids = {
        "cap_weight": [0.15, 0.25, 0.35],
        "pos_alpha": [0.25, 0.40, 0.60],
        "max_to": [0.10, 0.30, 0.50],
        "rebalance_every": [1, 5],
        "gamma": [0.5, 1.0, 1.5],
        "topk": [1, 2, 3],
    }

    rows = []
    for cap in grids["cap_weight"]:
        for pa in grids["pos_alpha"]:
            for mt in grids["max_to"]:
                for rb in grids["rebalance_every"]:
                    for gm in grids["gamma"]:
                        for tk in grids["topk"]:
                            df_trade = apply_signal_integration(df_trade_base, alpha_col="alpha", vol_col="vol_pred", gamma=gm)
                            gross, net, to = compute_portfolio_daily(
                                df_trade, pred_col="integrated_score", ret_col="y_ret_k",
                                long_short=LONG_SHORT, cap=float(cap),
                                min_spread=MIN_SPREAD, cost_bps=COST_BPS,
                                pos_alpha=float(pa), max_to=float(mt),
                                rebalance_every=int(rb),
                                regime_col="regime", regime_mult=REGIME_MULT,
                                vol_pred_col=None,
                                topk_holdings=int(tk),
                                min_cross_section=MIN_CROSS_SECTION,
                            )

                            if USE_PORTFOLIO_OVERLAY:
                                gross2, net2, _ = apply_portfolio_overlay(
                                    gross, net,
                                    target_vol_ann=TARGET_PORT_VOL_ANN,
                                    window=PORT_VOL_WINDOW,
                                    lev_cap=LEVERAGE_CAP,
                                    lev_floor=LEVERAGE_FLOOR,
                                    dd_start=DD_THROTTLE_START,
                                    dd_end=DD_THROTTLE_END,
                                    ann=ANN
                                )
                            else:
                                gross2, net2 = gross, net

                            rows.append({
                                "cap_weight": float(cap),
                                "pos_alpha": float(pa),
                                "max_to": float(mt),
                                "rebalance_every": int(rb),
                                "gamma": float(gm),
                                "topk": int(tk),
                                "sharpe_net": sharpe_from_daily(net2, ann=ANN),
                                "ann_ret_net": ann_return_from_daily(net2, ann=ANN),
                                "mdd_net": max_drawdown_from_daily(net2),
                                "turnover_mean": float(np.nanmean(to)) if np.isfinite(np.nanmean(to)) else np.nan,
                            })

    df_sens = pd.DataFrame(rows).sort_values("sharpe_net", ascending=False)
    out_csv = str(Path(out_dir) / "param_sensitivity_table.csv")
    df_sens.to_csv(out_csv, index=False, encoding="utf-8-sig")

    topn = df_sens.head(30).reset_index(drop=True)
    plt.figure()
    plt.plot(np.arange(len(topn)), topn["sharpe_net"].values)
    plt.title("Top-30 Sharpe in Param Sensitivity")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "param_sensitivity_top30.png"), dpi=150)
    plt.close()

    return out_csv

# =========================
# 方向诊断（不在输出中出现 flip 字样）
# =========================
def run_direction_diagnostics(df_oof: pd.DataFrame, out_dir: str, topk: int):
    """
    快速排雷：对比两种方向方案
    输出：
      - direction_check.csv（IC/RankIC + 策略 Sharpe 对照）
      - ic_rankic_summary_{alpha,alpha_schemeB,integrated,integrated_schemeB}.json
    """
    ensure_dir(out_dir)

    ic_a, sum_a = compute_ic_rankic(df_oof, alpha_col="alpha", ret_col="y_ret_k",
                                   min_n=max(2, MIN_CROSS_SECTION))
    df_flip = df_oof.copy()
    df_flip["alpha"] = -df_flip["alpha"].astype(float)
    ic_af, sum_af = compute_ic_rankic(df_flip, alpha_col="alpha", ret_col="y_ret_k",
                                     min_n=max(2, MIN_CROSS_SECTION))

    df_int = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)
    ic_i, sum_i = compute_ic_rankic(df_int, alpha_col="integrated_score", ret_col="y_ret_k",
                                   min_n=max(2, MIN_CROSS_SECTION))
    df_int_f = df_int.copy()
    df_int_f["integrated_score"] = -df_int_f["integrated_score"].astype(float)
    ic_if, sum_if = compute_ic_rankic(df_int_f, alpha_col="integrated_score", ret_col="y_ret_k",
                                      min_n=max(2, MIN_CROSS_SECTION))

    def _sharpe_for_score(df_trade: pd.DataFrame, score_col: str):
        gross, net, to = compute_portfolio_daily(
            df_trade, pred_col=score_col, ret_col="y_ret_k",
            long_short=LONG_SHORT, cap=CAP_WEIGHT,
            min_spread=MIN_SPREAD, cost_bps=COST_BPS,
            pos_alpha=POSITION_SMOOTH_ALPHA, max_to=MAX_DAILY_TURNOVER,
            rebalance_every=REBALANCE_EVERY,
            regime_col="regime", regime_mult=REGIME_MULT,
            vol_pred_col=None,
            topk_holdings=int(topk),
            min_cross_section=MIN_CROSS_SECTION,
        )
        if USE_PORTFOLIO_OVERLAY:
            _, net2, _ = apply_portfolio_overlay(
                gross, net,
                target_vol_ann=TARGET_PORT_VOL_ANN,
                window=PORT_VOL_WINDOW,
                lev_cap=LEVERAGE_CAP,
                lev_floor=LEVERAGE_FLOOR,
                dd_start=DD_THROTTLE_START,
                dd_end=DD_THROTTLE_END,
                ann=ANN
            )
        else:
            net2 = net
        return sharpe_from_daily(net2, ann=ANN)

    sh_pos = _sharpe_for_score(df_int, "integrated_score")
    df_int_f2 = df_int.copy()
    df_int_f2["integrated_score"] = -df_int_f2["integrated_score"].astype(float)
    sh_neg = _sharpe_for_score(df_int_f2, "integrated_score")

    rows = [
        {"name": "alpha_schemeA", "ic_mean": sum_a.get("ic_mean"), "rankic_mean": sum_a.get("rankic_mean"),
         "ic_ir": sum_a.get("ic_ir"), "rankic_ir": sum_a.get("rankic_ir"), "strategy_sharpe_net": np.nan},
        {"name": "alpha_schemeB", "ic_mean": sum_af.get("ic_mean"), "rankic_mean": sum_af.get("rankic_mean"),
         "ic_ir": sum_af.get("ic_ir"), "rankic_ir": sum_af.get("rankic_ir"), "strategy_sharpe_net": np.nan},
        {"name": "integrated_schemeA", "ic_mean": sum_i.get("ic_mean"), "rankic_mean": sum_i.get("rankic_mean"),
         "ic_ir": sum_i.get("ic_ir"), "rankic_ir": sum_i.get("rankic_ir"), "strategy_sharpe_net": sh_pos},
        {"name": "integrated_schemeB", "ic_mean": sum_if.get("ic_mean"), "rankic_mean": sum_if.get("rankic_mean"),
         "ic_ir": sum_if.get("ic_ir"), "rankic_ir": sum_if.get("rankic_ir"), "strategy_sharpe_net": sh_neg},
    ]
    df_out = pd.DataFrame(rows)

    out_csv = str(Path(out_dir) / "direction_check.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    Path(str(Path(out_dir) / "ic_rankic_summary_alpha.json")).write_text(json.dumps(sum_a, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(Path(out_dir) / "ic_rankic_summary_alpha_schemeB.json")).write_text(json.dumps(sum_af, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(Path(out_dir) / "ic_rankic_summary_integrated.json")).write_text(json.dumps(sum_i, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(Path(out_dir) / "ic_rankic_summary_integrated_schemeB.json")).write_text(json.dumps(sum_if, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_csv, df_out

# =========================
# 论文/排雷：消融 + 对齐自检 + TopK稳健性
# =========================
def _eval_strategy_pipeline(df_base: pd.DataFrame,
                            score_col: str,
                            ret_col: str,
                            topk: int,
                            use_overlay: bool,
                            cost_bps: float,
                            use_integration: bool,
                            ann: float = 252.0):
    """
    统一评估入口（用于 ablation / alignment / topk robustness）
    df_base 需要包含：ticker, trade_date, y_ret_k(或 ret_col), alpha, vol_pred, regime
    """
    dfx = df_base.copy()

    # score
    if use_integration:
        df_trade = apply_signal_integration(dfx, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)
        pred_col = "integrated_score" if score_col == "integrated_score" else score_col
    else:
        df_trade = dfx.copy()
        if score_col not in df_trade.columns:
            df_trade[score_col] = df_trade["alpha"].astype(float)
        pred_col = score_col

    if ret_col not in df_trade.columns:
        raise ValueError(f"ret_col not found: {ret_col}")

    gross, net, to = compute_portfolio_daily(
        df_trade, pred_col=pred_col, ret_col=ret_col,
        long_short=LONG_SHORT, cap=CAP_WEIGHT,
        min_spread=MIN_SPREAD, cost_bps=cost_bps,
        pos_alpha=POSITION_SMOOTH_ALPHA, max_to=MAX_DAILY_TURNOVER,
        rebalance_every=REBALANCE_EVERY,
        regime_col="regime", regime_mult=REGIME_MULT,
        vol_pred_col=None,
        topk_holdings=int(topk),
        min_cross_section=MIN_CROSS_SECTION,
    )

    if use_overlay and USE_PORTFOLIO_OVERLAY:
        gross2, net2, lev = apply_portfolio_overlay(
            gross, net,
            target_vol_ann=TARGET_PORT_VOL_ANN,
            window=PORT_VOL_WINDOW,
            lev_cap=LEVERAGE_CAP,
            lev_floor=LEVERAGE_FLOOR,
            dd_start=DD_THROTTLE_START,
            dd_end=DD_THROTTLE_END,
            ann=ann
        )
    else:
        gross2, net2 = gross, net
        lev = np.ones_like(net2, dtype=float)

    out = {
        "sharpe_net": sharpe_from_daily(net2, ann=ann),
        "ann_ret_net": ann_return_from_daily(net2, ann=ann),
        "ann_vol_net": ann_vol_from_daily(net2, ann=ann),
        "mdd_net": max_drawdown_from_daily(net2),
        "calmar_net": calmar_from_daily(net2, ann=ann),
        "turnover_mean": float(np.nanmean(to)) if np.isfinite(np.nanmean(to)) else np.nan,
        "n_days": int(np.isfinite(net2).sum()),
    }
    return out, gross2, net2, to, lev


def run_ablation(df_oof: pd.DataFrame, out_dir: str, topk: int, ann: float = 252.0):
    """
    消融实验（4组）：
      1) alpha + no overlay
      2) alpha + overlay
      3) integrated_score + no overlay
      4) integrated_score + overlay
    输出：ablation_table.csv
    """
    ensure_dir(out_dir)

    rows = []
    specs = [
        ("alpha_only", False, False, "alpha"),
        ("alpha_only_overlay", True, False, "alpha"),
        ("risk_adjusted", False, True, "integrated_score"),
        ("risk_adjusted_overlay", True, True, "integrated_score"),
    ]

    for name, use_overlay, use_integration, score_col in specs:
        if use_integration:
            df_score = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)
            alpha_col = "integrated_score"
        else:
            df_score = df_oof.copy()
            alpha_col = "alpha"

        _, ic_sum = compute_ic_rankic(df_score, alpha_col=alpha_col, ret_col="y_ret_k",
                                      min_n=max(2, MIN_CROSS_SECTION))

        perf, _, _, _, _ = _eval_strategy_pipeline(
            df_base=df_oof,
            score_col=score_col,
            ret_col="y_ret_k",
            topk=int(topk),
            use_overlay=bool(use_overlay),
            cost_bps=COST_BPS,
            use_integration=bool(use_integration),
            ann=ann
        )

        rows.append({
            "variant": name,
            "topk": int(topk),
            "use_integration": bool(use_integration),
            "use_overlay": bool(use_overlay),
            "sharpe_net": perf.get("sharpe_net"),
            "ann_ret_net": perf.get("ann_ret_net"),
            "ann_vol_net": perf.get("ann_vol_net"),
            "mdd_net": perf.get("mdd_net"),
            "calmar_net": perf.get("calmar_net"),
            "turnover_mean": perf.get("turnover_mean"),
            "ic_mean": ic_sum.get("ic_mean"),
            "ic_ir": ic_sum.get("ic_ir"),
            "rankic_mean": ic_sum.get("rankic_mean"),
            "rankic_ir": ic_sum.get("rankic_ir"),
            "days": ic_sum.get("days"),
        })

    df_out = pd.DataFrame(rows)
    out_csv = str(Path(out_dir) / "ablation_table.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv, df_out


def run_alignment_selfcheck(df_oof: pd.DataFrame, out_dir: str, topk: int, ann: float = 252.0):
    """
    对齐自检：同一份信号，换不同的收益对齐（base / lead1 / lag1）
    需要 df_oof 包含：y_ret_k_lead1, y_ret_k_lag1
    输出：alignment_selfcheck.csv
    """
    ensure_dir(out_dir)

    variants = [
        ("base_t_to_t1", "y_ret_k"),
        ("lead1_t1_to_t2", "y_ret_k_lead1"),
        ("lag1_tminus1_to_t", "y_ret_k_lag1"),
    ]

    rows = []
    df_trade = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)

    for name, rcol in variants:
        if rcol not in df_trade.columns:
            rows.append({
                "variant": name,
                "ret_col": rcol,
                "topk": int(topk),
                "sharpe_net": np.nan,
                "ann_ret_net": np.nan,
                "ann_vol_net": np.nan,
                "mdd_net": np.nan,
                "turnover_mean": np.nan,
                "ic_mean": np.nan,
                "rankic_mean": np.nan,
                "note": f"missing {rcol}",
            })
            continue

        _, ic_sum = compute_ic_rankic(df_trade, alpha_col="integrated_score", ret_col=rcol,
                                      min_n=max(2, MIN_CROSS_SECTION))

        perf, _, _, _, _ = _eval_strategy_pipeline(
            df_base=df_trade,
            score_col="integrated_score",
            ret_col=rcol,
            topk=int(topk),
            use_overlay=True,
            cost_bps=COST_BPS,
            use_integration=False,
            ann=ann
        )

        rows.append({
            "variant": name,
            "ret_col": rcol,
            "topk": int(topk),
            "sharpe_net": perf.get("sharpe_net"),
            "ann_ret_net": perf.get("ann_ret_net"),
            "ann_vol_net": perf.get("ann_vol_net"),
            "mdd_net": perf.get("mdd_net"),
            "turnover_mean": perf.get("turnover_mean"),
            "ic_mean": ic_sum.get("ic_mean"),
            "rankic_mean": ic_sum.get("rankic_mean"),
            "note": "",
        })

    df_out = pd.DataFrame(rows)
    out_csv = str(Path(out_dir) / "alignment_selfcheck.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv, df_out


def run_topk_robustness(df_oof: pd.DataFrame, out_dir: str, topk_list=None, ann: float = 252.0):
    """
    TopK稳健性：固定一套信号/风控，评估不同 topk
    输出：topk_robustness.csv
    """
    ensure_dir(out_dir)
    if topk_list is None:
        topk_list = [1, 2, 3, 5, 10]

    rows = []
    df_trade = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)
    for tk in topk_list:
        perf, _, _, _, _ = _eval_strategy_pipeline(
            df_base=df_trade,
            score_col="integrated_score",
            ret_col="y_ret_k",
            topk=int(tk),
            use_overlay=True,
            cost_bps=COST_BPS,
            use_integration=False,
            ann=ann
        )

        rows.append({
            "topk": int(tk),
            "sharpe_net": perf.get("sharpe_net"),
            "ann_ret_net": perf.get("ann_ret_net"),
            "ann_vol_net": perf.get("ann_vol_net"),
            "mdd_net": perf.get("mdd_net"),
            "calmar_net": perf.get("calmar_net"),
            "turnover_mean": perf.get("turnover_mean"),
        })

    df_out = pd.DataFrame(rows).sort_values("topk")
    out_csv = str(Path(out_dir) / "topk_robustness.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv, df_out

# =========================
# 评估：返回 dict（最后统一对比）
# =========================
def compute_report(df_all, name, alpha_oof, vol_oof, regime_oof,
                   ann=252.0, cost_bps=5.0, topk_holdings=0):
    mask = np.isfinite(alpha_oof) & np.isfinite(vol_oof) & np.isfinite(regime_oof)
    if mask.sum() < 50:
        return None

    df_oof = df_all.loc[mask, ["ticker", "trade_date", "y_ret_k"]].copy()
    df_oof["alpha"] = float(SCORE_SIGN) * alpha_oof[mask]
    df_oof["vol_pred"] = vol_oof[mask]
    df_oof["regime"] = regime_oof[mask].astype(int)

    pnl_point = np.sign(df_oof["alpha"].values) * df_oof["y_ret_k"].values.astype(float)
    sharpe_point = sharpe_from_daily(pnl_point, ann=ann)

    df_trade = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)

    gross, net, to = compute_portfolio_daily(
        df_trade, pred_col="integrated_score", ret_col="y_ret_k",
        long_short=LONG_SHORT, cap=CAP_WEIGHT,
        min_spread=MIN_SPREAD, cost_bps=cost_bps,
        pos_alpha=POSITION_SMOOTH_ALPHA, max_to=MAX_DAILY_TURNOVER,
        rebalance_every=REBALANCE_EVERY,
        regime_col="regime", regime_mult=REGIME_MULT,
        vol_pred_col=None,
        topk_holdings=int(topk_holdings),
        min_cross_section=MIN_CROSS_SECTION,
    )

    if USE_PORTFOLIO_OVERLAY:
        gross2, net2, _ = apply_portfolio_overlay(
            gross, net,
            target_vol_ann=TARGET_PORT_VOL_ANN,
            window=PORT_VOL_WINDOW,
            lev_cap=LEVERAGE_CAP,
            lev_floor=LEVERAGE_FLOOR,
            dd_start=DD_THROTTLE_START,
            dd_end=DD_THROTTLE_END,
            ann=ANN
        )
    else:
        gross2, net2 = gross, net

    mse = mean_squared_error(df_all.loc[mask, "y_ret_k"].values.astype(float), df_oof["alpha"].values.astype(float))
    sh_n = sharpe_from_daily(net2, ann=ann)

    return {
        "model": name,
        "rows": int(mask.sum()),
        "mse": float(mse),
        "sharpe_point": float(sharpe_point) if np.isfinite(sharpe_point) else np.nan,
        "sharpe_net": float(sh_n) if np.isfinite(sh_n) else np.nan,
        "ann_ret_net": float(ann_return_from_daily(net2, ann=ann)),
        "ann_vol_net": float(ann_vol_from_daily(net2, ann=ann)),
        "mdd_net": float(max_drawdown_from_daily(net2)),
        "calmar_net": float(calmar_from_daily(net2, ann=ann)),
        "turnover_mean": float(np.nanmean(to)) if np.isfinite(np.nanmean(to)) else np.nan,
        "topk_holdings": int(topk_holdings),
    }

# =========================
# 训练/缓存
# =========================
def load_cached_oof(cache_dir: str):
    p = Path(cache_dir) / "oof_arrays.joblib"
    if not p.exists():
        return None
    return joblib.load(p)

def save_cached_oof(cache_dir: str, pack: dict, meta: dict):
    ensure_dir(cache_dir)
    Path(cache_dir, "cache_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(pack, Path(cache_dir, "oof_arrays.joblib"))

def cache_valid(meta_path: Path, oof_path: Path, data_sig: str, cfg_sig: str):
    if not meta_path.exists() or not oof_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (meta.get("data_sig") == data_sig) and (meta.get("cfg_sig") == cfg_sig) and (meta.get("n_splits") == N_SPLITS)
    except Exception:
        return False

# =========================
# 最佳模型调参
# =========================
def grid_from_space(space: dict):
    keys = list(space.keys())
    grids = [space[k] for k in keys]
    out = []
    def rec(i, cur):
        if i == len(keys):
            out.append(cur.copy())
            return
        k = keys[i]
        for v in grids[i]:
            cur[k] = v
            rec(i + 1, cur)
    rec(0, {})
    return out

def tune_best_params(df: pd.DataFrame, X_alpha_cols, REGIME_COLS, X_risk_cols, dates: np.ndarray,
                     base_fill_cols: list, cache_dir: str):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    candidates = grid_from_space(TUNE_SPACE)

    best_score = -1e18
    best_params = None

    total_steps = int(len(candidates) * N_SPLITS)
    pbar = SimpleProgress(total_steps, desc="TUNE")

    def idx_from_dates(sel_dates):
        return np.where(df["trade_date"].isin(sel_dates).values)[0]

    for cand_i, cand in enumerate(candidates, start=1):
        oof_alpha = np.full(len(df), np.nan)
        oof_vol = np.full(len(df), np.nan)
        oof_reg = np.full(len(df), np.nan)

        for fold, (tr_didx, te_didx) in enumerate(tscv.split(dates), start=1):
            tr_dates = dates[tr_didx]
            te_dates = dates[te_didx]
            tr_idx = idx_from_dates(tr_dates)
            te_idx = idx_from_dates(te_dates)

            df_tr_raw = df.iloc[tr_idx].copy()
            df_te_raw = df.iloc[te_idx].copy()

            fill_cols = sorted(set(base_fill_cols + ["_ret_hist"]))
            fill_stats = compute_train_fill_stats(df_tr_raw, cols=fill_cols, by="ticker")
            df_tr = apply_fill_stats(df_tr_raw, fill_stats, by="ticker")
            df_te = apply_fill_stats(df_te_raw, fill_stats, by="ticker")

            km, mapping = fit_kmeans_regime(df_tr, REGIME_COLS)
            te_reg = predict_regime(km, mapping, df_te, REGIME_COLS)
            oof_reg[te_idx] = te_reg

            Xtr_full = df_tr[X_alpha_cols].values.astype(float)
            ytr = df_tr["y_ret_k"].values.astype(float)
            dates_tr = df_tr["trade_date"].values
            sel_idx, _ = lasso_select_features(Xtr_full, ytr, dates_tr,
                                               alphas=LASSO_ALPHAS, inner_splits=LASSO_INNER_SPLITS, topk=TOPK_FALLBACK)
            sel_cols = [X_alpha_cols[i] for i in sel_idx.tolist()]

            Xtr = df_tr[sel_cols].values.astype(float)
            Xte = df_te[sel_cols].values.astype(float)

            modern = make_gb_modern(task="alpha", random_state=TUNE_RANDOM_STATE)
            modern.fit(Xtr, ytr)
            pred_modern = modern.predict(Xte)

            rf = RandomForestRegressor(
                n_estimators=int(cand.get("rf_n_estimators", 500)),
                min_samples_leaf=int(cand.get("rf_min_samples_leaf", 10)),
                max_features=cand.get("rf_max_features", 1.0),
                n_jobs=-1,
                random_state=TUNE_RANDOM_STATE
            )
            rf.fit(Xtr, ytr)
            pred_rf = rf.predict(Xte)

            pred_modern_tr = modern.predict(Xtr)
            pred_rf_tr = rf.predict(Xtr)
            alpha_tr = 0.5 * pred_modern_tr + 0.5 * pred_rf_tr
            sgn = calibrate_sign(alpha_tr, ytr)

            alpha_te = sgn * (0.5 * pred_modern + 0.5 * pred_rf)
            oof_alpha[te_idx] = alpha_te

            Xtr_r = df_tr[X_risk_cols].values.astype(float)
            Xte_r = df_te[X_risk_cols].values.astype(float)
            ytr_v = df_tr["y_vol_k"].values.astype(float)

            risk = make_risk_regressor(random_state=TUNE_RANDOM_STATE)
            risk.fit(Xtr_r, ytr_v)
            vol_ml = risk.predict(Xte_r).astype(float)

            if USE_GARCH_RISK:
                vol_garch = fit_predict_garch_vol_by_ticker(
                    df_tr=df_tr, df_te=df_te, ret_hist_col="_ret_hist",
                    min_obs=GARCH_MIN_OBS, clip=RET_CLIP
                ).astype(float)
                vol_pred = (1.0 - float(GARCH_BLEND)) * vol_ml + float(GARCH_BLEND) * vol_garch
            else:
                vol_pred = vol_ml

            p95 = np.nanpercentile(vol_pred, 95) if np.isfinite(np.nanpercentile(vol_pred, 95)) else 1.0
            vol_pred = np.clip(vol_pred, 1e-6, p95 if p95 > 1e-6 else 1.0)
            oof_vol[te_idx] = vol_pred

            pbar.update(1, extra=f"cand={cand_i}/{len(candidates)} fold={fold}/{N_SPLITS}")

        rep = compute_report(df, "tmp", oof_alpha, oof_vol, oof_reg,
                             ann=ANN, cost_bps=COST_BPS, topk_holdings=TOPK_HOLDINGS_DEFAULT)
        score = rep["sharpe_net"] if rep is not None and np.isfinite(rep.get("sharpe_net", np.nan)) else -1e18
        if score > best_score:
            best_score = score
            best_params = cand.copy()

    if best_params is None:
        best_params = {}

    pbar.close(extra=f"best_sharpe_net={best_score:.6f}" if np.isfinite(best_score) else "done")
    Path(cache_dir, "tuned_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_params

# =========================
# 训练主流程
# =========================
def train_oof_all(df: pd.DataFrame, X_alpha_cols, REGIME_COLS, X_risk_cols, dates: np.ndarray, cache_dir: str, tuned_params: dict):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    oof_alpha_modern = np.full(len(df), np.nan)
    oof_alpha_gbdt = np.full(len(df), np.nan)
    oof_alpha_rf = np.full(len(df), np.nan)

    oof_stack_modern = np.full(len(df), np.nan)
    oof_stack_gbdt = np.full(len(df), np.nan)
    oof_stack_modern_tuned = np.full(len(df), np.nan)

    oof_vol_pred = np.full(len(df), np.nan)
    oof_regime = np.full(len(df), np.nan)

    selected_cols_each_fold = []
    pbar = SimpleProgress(N_SPLITS, desc="TRAIN")

    def idx_from_dates(sel_dates):
        return np.where(df["trade_date"].isin(sel_dates).values)[0]

    for fold, (tr_didx, te_didx) in enumerate(tscv.split(dates), start=1):
        tr_dates = dates[tr_didx]
        te_dates = dates[te_didx]
        tr_idx = idx_from_dates(tr_dates)
        te_idx = idx_from_dates(te_dates)

        df_tr_raw = df.iloc[tr_idx].copy()
        df_te_raw = df.iloc[te_idx].copy()

        fill_cols = sorted(set(X_alpha_cols + REGIME_COLS + X_risk_cols + ["_ret_hist"]))
        fill_stats = compute_train_fill_stats(df_tr_raw, cols=fill_cols, by="ticker")
        df_tr = apply_fill_stats(df_tr_raw, fill_stats, by="ticker")
        df_te = apply_fill_stats(df_te_raw, fill_stats, by="ticker")

        km, mapping = fit_kmeans_regime(df_tr, REGIME_COLS)
        te_reg = predict_regime(km, mapping, df_te, REGIME_COLS)
        oof_regime[te_idx] = te_reg

        Xtr_full = df_tr[X_alpha_cols].values.astype(float)
        ytr = df_tr["y_ret_k"].values.astype(float)
        dates_tr = df_tr["trade_date"].values

        sel_idx, best_lasso_a = lasso_select_features(
            Xtr_full, ytr, dates_tr,
            alphas=LASSO_ALPHAS, inner_splits=LASSO_INNER_SPLITS, topk=TOPK_FALLBACK
        )
        sel_cols = [X_alpha_cols[i] for i in sel_idx.tolist()]
        selected_cols_each_fold.append({"fold": fold, "n_sel": len(sel_cols), "best_lasso_alpha": best_lasso_a})

        Xtr = df_tr[sel_cols].values.astype(float)
        Xte = df_te[sel_cols].values.astype(float)

        t0 = time.time()

        modern = make_gb_modern(task="alpha", random_state=42)
        modern.fit(Xtr, ytr)
        pred_modern_tr = modern.predict(Xtr)
        sgn_modern = calibrate_sign(pred_modern_tr, ytr)
        pred_modern = sgn_modern * modern.predict(Xte)
        oof_alpha_modern[te_idx] = pred_modern

        gbdt = make_gbdt_sklearn(task="alpha", random_state=42)
        gbdt.fit(Xtr, ytr)
        pred_gbdt_tr = gbdt.predict(Xtr)
        sgn_gbdt = calibrate_sign(pred_gbdt_tr, ytr)
        pred_gbdt = sgn_gbdt * gbdt.predict(Xte)
        oof_alpha_gbdt[te_idx] = pred_gbdt

        rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=10, n_jobs=-1, random_state=42)
        rf.fit(Xtr, ytr)
        pred_rf_tr = rf.predict(Xtr)
        sgn_rf = calibrate_sign(pred_rf_tr, ytr)
        pred_rf = sgn_rf * rf.predict(Xte)
        oof_alpha_rf[te_idx] = pred_rf

        modern_t = make_gb_modern(task="alpha", random_state=42)
        modern_t.fit(Xtr, ytr)
        pred_modern_t_tr = modern_t.predict(Xtr)
        sgn_modern_t = calibrate_sign(pred_modern_t_tr, ytr)
        pred_modern_t = sgn_modern_t * modern_t.predict(Xte)

        rf_t = RandomForestRegressor(
            n_estimators=int((tuned_params or {}).get("rf_n_estimators", 500)),
            min_samples_leaf=int((tuned_params or {}).get("rf_min_samples_leaf", 10)),
            max_features=(tuned_params or {}).get("rf_max_features", 1.0),
            n_jobs=-1,
            random_state=42
        )
        rf_t.fit(Xtr, ytr)
        pred_rf_t_tr = rf_t.predict(Xtr)
        sgn_rf_t = calibrate_sign(pred_rf_t_tr, ytr)
        pred_rf_t = sgn_rf_t * rf_t.predict(Xte)

        print(f"[Fold {fold}] base(alpha) done, time={time.time() - t0:.2f}s, sel_features={len(sel_cols)}")

        inner_dates = np.array(sorted(df_tr["trade_date"].unique()))
        inner_tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(inner_dates) // 50)))

        def _stack_fit_predict(baseA_builder, baseB_builder, predA_te, predB_te, ridge_alpha):
            meta_oof_a = np.full(len(df_tr), np.nan)
            meta_oof_b = np.full(len(df_tr), np.nan)

            for _, (itr_didx2, iva_didx2) in enumerate(inner_tscv.split(inner_dates), start=1):
                itr_dates2 = inner_dates[itr_didx2]
                iva_dates2 = inner_dates[iva_didx2]
                itr_idx2 = np.where(df_tr["trade_date"].isin(itr_dates2).values)[0]
                iva_idx2 = np.where(df_tr["trade_date"].isin(iva_dates2).values)[0]
                if len(itr_idx2) < 100 or len(iva_idx2) < 50:
                    continue

                X_itr = df_tr.iloc[itr_idx2][sel_cols].values.astype(float)
                y_itr = df_tr.iloc[itr_idx2]["y_ret_k"].values.astype(float)
                X_iva = df_tr.iloc[iva_idx2][sel_cols].values.astype(float)

                mA = baseA_builder()
                mA.fit(X_itr, y_itr)
                meta_oof_a[iva_idx2] = mA.predict(X_iva)

                mB = baseB_builder()
                mB.fit(X_itr, y_itr)
                meta_oof_b[iva_idx2] = mB.predict(X_iva)

            msk = np.isfinite(meta_oof_a) & np.isfinite(meta_oof_b)
            if msk.sum() < 200:
                tr_pred = 0.5 * meta_oof_a[msk] + 0.5 * meta_oof_b[msk]
                tr_y = df_tr.loc[msk, "y_ret_k"].values.astype(float)
                sgn = calibrate_sign(tr_pred, tr_y)
                return sgn * (0.5 * predA_te + 0.5 * predB_te)

            meta_X = np.vstack([meta_oof_a[msk], meta_oof_b[msk]]).T
            meta_y = df_tr.loc[msk, "y_ret_k"].values.astype(float)

            meta = Ridge(alpha=float(ridge_alpha), random_state=42)
            meta.fit(meta_X, meta_y)

            tr_pred = meta.predict(meta_X)
            sgn = calibrate_sign(tr_pred, meta_y)

            te_X = np.vstack([predA_te, predB_te]).T
            return sgn * meta.predict(te_X)

        oof_stack_modern[te_idx] = _stack_fit_predict(
            baseA_builder=lambda: make_gb_modern(task="alpha", random_state=42),
            baseB_builder=lambda: RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=10, n_jobs=-1, random_state=42),
            predA_te=pred_modern,
            predB_te=pred_rf,
            ridge_alpha=RIDGE_ALPHA
        )

        oof_stack_gbdt[te_idx] = _stack_fit_predict(
            baseA_builder=lambda: make_gbdt_sklearn(task="alpha", random_state=42),
            baseB_builder=lambda: RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=10, n_jobs=-1, random_state=42),
            predA_te=pred_gbdt,
            predB_te=pred_rf,
            ridge_alpha=RIDGE_ALPHA
        )

        tuned_ridge = float((tuned_params or {}).get("ridge_alpha", RIDGE_ALPHA))
        oof_stack_modern_tuned[te_idx] = _stack_fit_predict(
            baseA_builder=lambda: make_gb_modern(task="alpha", random_state=42),
            baseB_builder=lambda: RandomForestRegressor(
                n_estimators=int((tuned_params or {}).get("rf_n_estimators", 500)),
                min_samples_leaf=int((tuned_params or {}).get("rf_min_samples_leaf", 10)),
                max_features=(tuned_params or {}).get("rf_max_features", 1.0),
                n_jobs=-1,
                random_state=42
            ),
            predA_te=pred_modern_t,
            predB_te=pred_rf_t,
            ridge_alpha=tuned_ridge
        )

        Xtr_r = df_tr[X_risk_cols].values.astype(float)
        Xte_r = df_te[X_risk_cols].values.astype(float)
        ytr_v = df_tr["y_vol_k"].values.astype(float)

        risk = make_risk_regressor(random_state=42)
        risk.fit(Xtr_r, ytr_v)
        vol_ml = risk.predict(Xte_r).astype(float)

        if USE_GARCH_RISK:
            vol_garch = fit_predict_garch_vol_by_ticker(
                df_tr=df_tr, df_te=df_te,
                ret_hist_col="_ret_hist",
                min_obs=GARCH_MIN_OBS,
                clip=RET_CLIP
            ).astype(float)
            vol_pred = (1.0 - float(GARCH_BLEND)) * vol_ml + float(GARCH_BLEND) * vol_garch
        else:
            vol_pred = vol_ml

        vol_pred = np.asarray(vol_pred, dtype=float)
        p95 = np.nanpercentile(vol_pred, 95) if np.isfinite(np.nanpercentile(vol_pred, 95)) else 1.0
        vol_pred = np.clip(vol_pred, 1e-6, p95 if p95 > 1e-6 else 1.0)
        oof_vol_pred[te_idx] = vol_pred

        pbar.update(1, extra=f"fold={fold}/{N_SPLITS} sel={len(sel_cols)}")

    pbar.close(extra="folds_done")
    pd.DataFrame(selected_cols_each_fold).to_csv(OUT_LASSO_LOG, index=False, encoding="utf-8-sig")
    print("saved:", OUT_LASSO_LOG)

    pack = {
        "oof_alpha_modern": oof_alpha_modern,
        "oof_alpha_gbdt": oof_alpha_gbdt,
        "oof_alpha_rf": oof_alpha_rf,
        "oof_stack_modern": oof_stack_modern,
        "oof_stack_gbdt": oof_stack_gbdt,
        "oof_stack_modern_tuned": oof_stack_modern_tuned,
        "oof_vol_pred": oof_vol_pred,
        "oof_regime": oof_regime,
    }
    return pack, selected_cols_each_fold

def pick_best_topk_for_main(df_oof: pd.DataFrame, cost_bps: float):
    best = {"topk": TOPK_HOLDINGS_DEFAULT, "sh": -1e18, "gross": None, "net": None, "to": None}

    df_base = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)
    for tk in TOPK_HOLDINGS_CANDIDATES:
        gross, net, to = compute_portfolio_daily(
            df_base, pred_col="integrated_score", ret_col="y_ret_k",
            long_short=LONG_SHORT, cap=CAP_WEIGHT,
            min_spread=MIN_SPREAD, cost_bps=cost_bps,
            pos_alpha=POSITION_SMOOTH_ALPHA, max_to=MAX_DAILY_TURNOVER,
            rebalance_every=REBALANCE_EVERY,
            regime_col="regime", regime_mult=REGIME_MULT,
            vol_pred_col=None,
            topk_holdings=int(tk),
            min_cross_section=MIN_CROSS_SECTION,
        )
        if USE_PORTFOLIO_OVERLAY:
            _, net2, _ = apply_portfolio_overlay(
                gross, net,
                target_vol_ann=TARGET_PORT_VOL_ANN,
                window=PORT_VOL_WINDOW,
                lev_cap=LEVERAGE_CAP,
                lev_floor=LEVERAGE_FLOOR,
                dd_start=DD_THROTTLE_START,
                dd_end=DD_THROTTLE_END,
                ann=ANN
            )
        else:
            net2 = net

        sh = sharpe_from_daily(net2, ann=ANN)
        if np.isfinite(sh) and sh > best["sh"]:
            best = {"topk": int(tk), "sh": float(sh), "gross": gross, "net": net, "to": to}

    return best
# ====== ONLINE EXPORT / PREDICT ======
FINAL_MODEL_PATH = str(Path(CACHE_DIR) / "final_model.joblib")

def train_and_export_final_model(df: pd.DataFrame,
                                 X_alpha_cols: list,
                                 REGIME_COLS: list,
                                 X_risk_cols: list,
                                 tuned_params: dict,
                                 out_path: str = None):
    """
    用全量数据训练“线上可推理”的最终模型，并保存为 joblib
    """
    if out_path is None:
        out_path = FINAL_MODEL_PATH

    # 1) fill_stats（全量）
    fill_cols = sorted(set(X_alpha_cols + REGIME_COLS + X_risk_cols + ["_ret_hist"]))
    fill_stats = compute_train_fill_stats(df, cols=fill_cols, by="ticker")
    df_f = apply_fill_stats(df.copy(), fill_stats, by="ticker")

    # 2) regime（全量）
    km, mapping = fit_kmeans_regime(df_f, REGIME_COLS)

    # 3) alpha：全量做一次 LASSO 选特征
    Xtr_full = df_f[X_alpha_cols].values.astype(float)
    ytr = df_f["y_ret_k"].values.astype(float)
    dates_tr = df_f["trade_date"].values
    sel_idx, best_lasso_a = lasso_select_features(
        Xtr_full, ytr, dates_tr,
        alphas=LASSO_ALPHAS, inner_splits=LASSO_INNER_SPLITS, topk=TOPK_FALLBACK
    )
    sel_cols = [X_alpha_cols[i] for i in sel_idx.tolist()]
    Xtr = df_f[sel_cols].values.astype(float)

    # 4) alpha base models：modern + rf
    modern = make_gb_modern(task="alpha", random_state=42)
    modern.fit(Xtr, ytr)
    pred_modern_tr = modern.predict(Xtr)
    sgn_modern = calibrate_sign(pred_modern_tr, ytr)

    rf = RandomForestRegressor(
        n_estimators=int((tuned_params or {}).get("rf_n_estimators", 500)),
        min_samples_leaf=int((tuned_params or {}).get("rf_min_samples_leaf", 10)),
        max_features=(tuned_params or {}).get("rf_max_features", 1.0),
        n_jobs=-1,
        random_state=42
    )
    rf.fit(Xtr, ytr)
    pred_rf_tr = rf.predict(Xtr)
    sgn_rf = calibrate_sign(pred_rf_tr, ytr)

    # 5) meta ridge（全量拟合最终 meta）
    ridge_a = float((tuned_params or {}).get("ridge_alpha", RIDGE_ALPHA))
    meta = Ridge(alpha=ridge_a, random_state=42)
    meta_X = np.vstack([sgn_modern * pred_modern_tr, sgn_rf * pred_rf_tr]).T
    meta.fit(meta_X, ytr)
    pred_meta_tr = meta.predict(meta_X)
    sgn_meta = calibrate_sign(pred_meta_tr, ytr)

    # 6) risk model（全量）
    Xtr_r = df_f[X_risk_cols].values.astype(float)
    ytr_v = df_f["y_vol_k"].values.astype(float)
    risk = make_risk_regressor(random_state=42)
    risk.fit(Xtr_r, ytr_v)

    # risk clip 用全量的 ML 预测 p95
    vol_ml_tr = risk.predict(Xtr_r).astype(float)
    vol_p95 = float(np.nanpercentile(vol_ml_tr, 95)) if np.isfinite(np.nanpercentile(vol_ml_tr, 95)) else 1.0
    if not (vol_p95 > 1e-6):
        vol_p95 = 1.0

    # 7) 给线上 GARCH 留历史 ret（按 ticker 保存最近 GARCH_MIN_OBS 条）
    ret_hist_by_ticker = {}
    for tk, g in df_f.groupby("ticker", sort=False):
        arr = g["_ret_hist"].dropna().values.astype(float)
        if arr.size >= max(10, int(GARCH_MIN_OBS)):
            ret_hist_by_ticker[str(tk)] = arr[-int(GARCH_MIN_OBS):].tolist()

    pack = {
        "version": "final_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tuned_params": tuned_params or {},

        "fill_stats": fill_stats,
        "X_alpha_cols": X_alpha_cols,
        "REGIME_COLS": REGIME_COLS,
        "X_risk_cols": X_risk_cols,

        "km": km,
        "mapping": mapping,

        "sel_cols": sel_cols,
        "best_lasso_alpha": best_lasso_a,

        "modern": modern,
        "rf": rf,
        "meta": meta,

        "sgn_modern": float(sgn_modern),
        "sgn_rf": float(sgn_rf),
        "sgn_meta": float(sgn_meta),

        "risk": risk,
        "vol_p95": float(vol_p95),

        "use_garch_risk": bool(USE_GARCH_RISK),
        "garch_blend": float(GARCH_BLEND),
        "garch_min_obs": int(GARCH_MIN_OBS),
        "ret_clip": float(RET_CLIP),
        "ret_hist_by_ticker": ret_hist_by_ticker,
    }

    joblib.dump(pack, out_path)
    return out_path, pack


def _make_garch_df_from_hist(pack: dict, tickers: list) -> pd.DataFrame:
    rows = []
    base_date = pd.Timestamp("2000-01-01")
    for t in tickers:
        arr = pack["ret_hist_by_ticker"].get(str(t), [])
        for i, r in enumerate(arr):
            rows.append({"ticker": t, "trade_date": base_date + pd.Timedelta(days=i), "_ret_hist": float(r)})
    return pd.DataFrame(rows)


def predict_with_final_model(pack: dict, df_in: pd.DataFrame, topk: int = 3, gamma: float = None):
    """
    df_in: 必须含 ticker, trade_date, 以及 pack 里需要的特征列（sel_cols/REGIME_COLS/X_risk_cols）
    返回：all_df, topk_df
    """
    d = df_in.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["ticker", "trade_date"]).sort_values(["trade_date", "ticker"]).reset_index(drop=True)

    if "_ret_hist" not in d.columns:
        d["_ret_hist"] = np.nan

    # fill
    d = apply_fill_stats(d, pack["fill_stats"], by="ticker")

    # regime
    d["regime"] = predict_regime(pack["km"], pack["mapping"], d, pack["REGIME_COLS"]).astype(int)

    # alpha
    X = d[pack["sel_cols"]].values.astype(float)
    pA = pack["sgn_modern"] * pack["modern"].predict(X)
    pB = pack["sgn_rf"] * pack["rf"].predict(X)
    meta_X = np.vstack([pA, pB]).T
    alpha = pack["sgn_meta"] * pack["meta"].predict(meta_X)
    d["alpha"] = alpha.astype(float)

    # risk
    Xr = d[pack["X_risk_cols"]].values.astype(float)
    vol_ml = pack["risk"].predict(Xr).astype(float)
    vol_pred = vol_ml

    if bool(pack.get("use_garch_risk", False)) and len(pack.get("ret_hist_by_ticker", {})) > 0:
        tickers = d["ticker"].astype(str).tolist()
        if len(tickers) <= 200:
            df_tr_g = _make_garch_df_from_hist(pack, tickers)
            if not df_tr_g.empty:
                df_te_g = d[["ticker", "trade_date", "_ret_hist"]].copy()
                df_tr_g["_ret_hist"] = df_tr_g["_ret_hist"].clip(-pack["ret_clip"], pack["ret_clip"])
                df_te_g["_ret_hist"] = df_te_g["_ret_hist"].clip(-pack["ret_clip"], pack["ret_clip"])
                try:
                    vol_garch = fit_predict_garch_vol_by_ticker(
                        df_tr=df_tr_g, df_te=df_te_g,
                        ret_hist_col="_ret_hist",
                        min_obs=int(pack["garch_min_obs"]),
                        clip=float(pack["ret_clip"])
                    ).astype(float)
                    b = float(pack["garch_blend"])
                    vol_pred = (1.0 - b) * vol_ml + b * vol_garch
                except Exception:
                    vol_pred = vol_ml

    p95 = float(pack.get("vol_p95", 1.0))
    vol_pred = np.clip(np.asarray(vol_pred, float), 1e-6, p95 if p95 > 1e-6 else 1.0)
    d["vol_pred"] = vol_pred.astype(float)

    # integrate
    g = float(gamma) if gamma is not None else float(VOL_GAMMA)
    d2 = apply_signal_integration(d[["ticker", "trade_date", "alpha", "vol_pred", "regime"]].copy(),
                                  alpha_col="alpha", vol_col="vol_pred", gamma=g)

    d2 = d2.sort_values(["trade_date", "integrated_score"], ascending=[True, False]).reset_index(drop=True)
    d2["rank_in_date"] = d2.groupby("trade_date")["integrated_score"].rank(ascending=False, method="first").astype(int)
    top = d2[d2["rank_in_date"] <= int(topk)].copy()
    return d2, top

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    ensure_dir(OUT_REPORT_DIR)
    ensure_dir(CACHE_DIR)

    df = pd.read_csv(DATA_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ticker", "trade_date"]).copy()
    df = df.sort_values(["trade_date", "ticker"]).reset_index(drop=True)

    # 构造“已知历史收益”列：ret_hist(t) = y_ret_k(t-1)
    df["_ret_hist"] = df.groupby("ticker", sort=False)["y_ret_k"].shift(1).astype(float)
    df["_ret_hist"] = df["_ret_hist"].clip(lower=-RET_CLIP, upper=RET_CLIP)

    # 对齐自检用：不同收益对齐（不参与训练）
    df["y_ret_k_lag1"] = df.groupby("ticker", sort=False)["y_ret_k"].shift(1).astype(float)
    df["y_ret_k_lead1"] = df.groupby("ticker", sort=False)["y_ret_k"].shift(-1).astype(float)

    key_cols = ["ticker", "trade_date"]
    label_cols = ["y_ret_k", "y_up_k", "y_vol_k"]
    feat_cols = [c for c in df.columns if c not in key_cols + label_cols + ["_ret_hist", "y_ret_k_lag1", "y_ret_k_lead1"]]

    X_alpha_cols = [c for c in feat_cols if ("_L1_z60" in c) or ("_L1_z120" in c)]
    if len(X_alpha_cols) < 5:
        raise ValueError("alpha 特征列不足，请检查 CSV 是否包含 *_L1_z60/_L1_z120 等列。")

    regime_cols_try = [
        "rv_20_L1_z120", "log_turn_L1_z120", "spread_rel_L1_z120", "depth_imb_L1_z120", "ofi_L1_z120", "adx_L1_z120",
        "rv_20_L1_z60",  "log_turn_L1_z60",  "spread_rel_L1_z60",  "depth_imb_L1_z60",  "ofi_L1_z60",  "adx_L1_z60",
        "range_hl_L1_z120", "bb_bw_L1_z120",
        "range_hl_L1_z60",  "bb_bw_L1_z60",
    ]
    REGIME_COLS = [c for c in regime_cols_try if c in df.columns]
    if len(REGIME_COLS) < 3:
        raise ValueError("regime 特征列不足，请检查 CSV 是否包含 rv_20/log_turn/ADX 等列。")

    risk_cols_try = [
        "rv_20_L1_z120", "vol_20_L1_z120", "vol_10_L1_z120", "atr_L1_z120", "bb_bw_L1_z120", "log_turn_L1_z120",
        "spread_rel_L1_z120", "depth_imb_L1_z120", "ofi_L1_z120", "range_hl_L1_z120", "adx_L1_z120",
        "rv_20_L1_z60", "vol_20_L1_z60", "vol_10_L1_z60", "atr_L1_z60", "bb_bw_L1_z60", "log_turn_L1_z60",
        "spread_rel_L1_z60", "depth_imb_L1_z60", "ofi_L1_z60", "range_hl_L1_z60", "adx_L1_z60",
    ]
    X_risk_cols = [c for c in risk_cols_try if c in df.columns]
    if len(X_risk_cols) < 4:
        X_risk_cols = REGIME_COLS

    dates = np.array(sorted(df["trade_date"].unique()))

    data_sig = sha1_of_file(DATA_PATH)
    cfg_sig = hashlib.sha1(json.dumps({
        "CACHE_VERSION": 9,
        "N_SPLITS": N_SPLITS,
        "LASSO_ALPHAS": [float(x) for x in LASSO_ALPHAS],
        "TOPK_FALLBACK": TOPK_FALLBACK,
        "K_REGIME": K_REGIME,
        "RIDGE_ALPHA": RIDGE_ALPHA,
        "USE_GARCH_RISK": bool(USE_GARCH_RISK),
        "GARCH_BLEND": float(GARCH_BLEND),
        "GARCH_MIN_OBS": int(GARCH_MIN_OBS),
        "RET_CLIP": float(RET_CLIP),
        "ENABLE_TUNE_BEST": bool(ENABLE_TUNE_BEST),
        "TUNE_SPACE": TUNE_SPACE,
        "REGIME_COLS": REGIME_COLS,
        "RISK_COLS": X_risk_cols,
        "X_ALPHA_COLS_N": len(X_alpha_cols),
        "MODERN_ALPHA_DEFAULT": MODERN_ALPHA_DEFAULT,
        "MODERN_RISK_DEFAULT": MODERN_RISK_DEFAULT,
        "TOPK_HOLDINGS_CANDIDATES": TOPK_HOLDINGS_CANDIDATES,
        "MIN_CROSS_SECTION": MIN_CROSS_SECTION,
        "MIN_SPREAD": MIN_SPREAD,
        "TARGET_PORT_VOL_ANN": TARGET_PORT_VOL_ANN,
        "DD_THROTTLE_START": DD_THROTTLE_START,
        "DD_THROTTLE_END": DD_THROTTLE_END,
        "VOL_GAMMA": VOL_GAMMA,
        "USE_PORTFOLIO_OVERLAY": USE_PORTFOLIO_OVERLAY,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    meta_path = Path(CACHE_DIR) / "cache_meta.json"
    oof_path = Path(CACHE_DIR) / "oof_arrays.joblib"
    cache_hit = cache_valid(meta_path, oof_path, data_sig, cfg_sig)

    tuned_params = {}
    tuned_path = Path(CACHE_DIR) / "tuned_params.json"

    # 记录/恢复：modernGB 实际用的后端（alpha / risk）
    modern_gb_backend_alpha = "unknown"
    modern_gb_backend_risk = "unknown"

    if cache_hit and meta_path.exists():
        try:
            _meta = json.loads(meta_path.read_text(encoding="utf-8"))
            modern_gb_backend_alpha = str(_meta.get("modern_gb_backend_alpha", "unknown"))
            modern_gb_backend_risk = str(_meta.get("modern_gb_backend_risk", "unknown"))
        except Exception:
            modern_gb_backend_alpha = "unknown"
            modern_gb_backend_risk = "unknown"

    if not cache_hit:
        # 本次训练将实际 backend 记录进 meta（与训练时环境一致）
        try:
            modern_gb_backend_alpha = gb_backend_tag(make_gb_modern(task="alpha", random_state=42))
        except Exception:
            modern_gb_backend_alpha = "unknown"
        try:
            modern_gb_backend_risk = gb_backend_tag(make_gb_modern(task="risk", random_state=42))
        except Exception:
            modern_gb_backend_risk = "unknown"

        t_all0 = time.perf_counter()
        base_fill_cols = sorted(set(X_alpha_cols + REGIME_COLS + X_risk_cols))

        if ENABLE_TUNE_BEST:
            print("\n==================== TUNE BEST MODEL ====================")
            tuned_params = tune_best_params(
                df=df,
                X_alpha_cols=X_alpha_cols,
                REGIME_COLS=REGIME_COLS,
                X_risk_cols=X_risk_cols,
                dates=dates,
                base_fill_cols=base_fill_cols,
                cache_dir=CACHE_DIR
            )
        else:
            tuned_params = {}

        pack, selected_log = train_oof_all(
            df=df,
            X_alpha_cols=X_alpha_cols,
            REGIME_COLS=REGIME_COLS,
            X_risk_cols=X_risk_cols,
            dates=dates,
            cache_dir=CACHE_DIR,
            tuned_params=tuned_params
        )

        meta = {
            "version": 8,
            "data_sig": data_sig,
            "cfg_sig": cfg_sig,
            "n_splits": N_SPLITS,
            "tuned_params": tuned_params,
            "has_tuned": bool(ENABLE_TUNE_BEST),
            "selected_cols_each_fold": selected_log,
            "modern_gb_backend_alpha": modern_gb_backend_alpha,
            "modern_gb_backend_risk": modern_gb_backend_risk,
        }
        save_cached_oof(CACHE_DIR, pack, meta)
        cache_hit = False
        print(f"\nTOTAL TRAIN TIME = {_fmt_sec(time.perf_counter() - t_all0)}")
    else:
        if tuned_path.exists():
            try:
                tuned_params = json.loads(tuned_path.read_text(encoding="utf-8"))
            except Exception:
                tuned_params = {}

        # cache 命中但 meta 没有 backend 时，尽量现场探测一次（兜底）
        if modern_gb_backend_alpha == "unknown":
            try:
                modern_gb_backend_alpha = gb_backend_tag(make_gb_modern(task="alpha", random_state=42))
            except Exception:
                modern_gb_backend_alpha = "unknown"
        if modern_gb_backend_risk == "unknown":
            try:
                modern_gb_backend_risk = gb_backend_tag(make_gb_modern(task="risk", random_state=42))
            except Exception:
                modern_gb_backend_risk = "unknown"

    oof_pack = load_cached_oof(CACHE_DIR)
    if oof_pack is None:
        raise RuntimeError("缓存缺失：请删除 CACHE_DIR 重新运行。")

    oof_alpha_modern = oof_pack.get("oof_alpha_modern")
    oof_alpha_gbdt = oof_pack.get("oof_alpha_gbdt")
    oof_alpha_rf = oof_pack.get("oof_alpha_rf")
    oof_stack_modern = oof_pack.get("oof_stack_modern")
    oof_stack_gbdt = oof_pack.get("oof_stack_gbdt")
    oof_stack_modern_tuned = oof_pack.get("oof_stack_modern_tuned")
    oof_vol_pred = oof_pack.get("oof_vol_pred")
    oof_regime = oof_pack.get("oof_regime")

    reports = []
    for tk in TOPK_HOLDINGS_CANDIDATES:
        reports.append(compute_report(df, f"alpha_modernGB({modern_gb_backend_alpha})[topk={tk}]", oof_alpha_modern, oof_vol_pred, oof_regime, ann=ANN, cost_bps=COST_BPS, topk_holdings=tk))
        reports.append(compute_report(df, f"alpha_GBDT(sklearn)[topk={tk}]", oof_alpha_gbdt, oof_vol_pred, oof_regime, ann=ANN, cost_bps=COST_BPS, topk_holdings=tk))
        reports.append(compute_report(df, f"alpha_RF[topk={tk}]", oof_alpha_rf, oof_vol_pred, oof_regime, ann=ANN, cost_bps=COST_BPS, topk_holdings=tk))
        reports.append(compute_report(df, f"stack_modernGB+RF->Ridge[topk={tk}]", oof_stack_modern, oof_vol_pred, oof_regime, ann=ANN, cost_bps=COST_BPS, topk_holdings=tk))
        reports.append(compute_report(df, f"stack_GBDT+RF->Ridge[topk={tk}]", oof_stack_gbdt, oof_vol_pred, oof_regime, ann=ANN, cost_bps=COST_BPS, topk_holdings=tk))
        if oof_stack_modern_tuned is not None:
            reports.append(compute_report(df, f"stack_modernGB+RF->Ridge[TUNED][topk={tk}]", oof_stack_modern_tuned, oof_vol_pred, oof_regime, ann=ANN, cost_bps=COST_BPS, topk_holdings=tk))

    reports = [r for r in reports if r is not None]
    df_cmp = pd.DataFrame(reports).sort_values(["sharpe_net", "sharpe_point"], ascending=False)

    print("\n==================== CACHE ====================")
    print(f"CACHE_HIT={cache_hit}")
    print(f"CACHE_DIR={CACHE_DIR}")
    print(f"MODERN_GB_BACKEND_ALPHA={modern_gb_backend_alpha}")
    print(f"MODERN_GB_BACKEND_RISK={modern_gb_backend_risk}")
    print("\n==================== MODEL COMPARISON (sorted by sharpe_net) ====================")
    print(df_cmp.to_string(index=False))

    # 追加：保存模型对比表，供 dashboard 读取
    out_cmp_csv = str(Path(OUT_REPORT_DIR) / "model_comparison.csv")
    df_cmp.to_csv(out_cmp_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_cmp_csv)

    best_model_str = str(df_cmp.iloc[0]["model"]) if (df_cmp is not None and len(df_cmp) > 0) else ""


    def _alpha_by_model_name(m: str):
        m = str(m)
        if "stack_modernGB+RF->Ridge[TUNED]" in m:
            return oof_stack_modern_tuned, "stack_modernGB+RF->Ridge[TUNED]"
        if "stack_modernGB+RF->Ridge" in m:
            return oof_stack_modern, "stack_modernGB+RF->Ridge"
        if "stack_GBDT+RF->Ridge" in m:
            return oof_stack_gbdt, "stack_GBDT+RF->Ridge"
        if "alpha_RF" in m:
            return oof_alpha_rf, "alpha_RF"
        if "alpha_GBDT(sklearn)" in m:
            return oof_alpha_gbdt, "alpha_GBDT(sklearn)"
        if "alpha_modernGB" in m:
            return oof_alpha_modern, f"alpha_modernGB({modern_gb_backend_alpha})"
        return None, "UNKNOWN"

    main_alpha, main_name = _alpha_by_model_name(best_model_str)

    if main_alpha is None or (not np.isfinite(main_alpha).any()):
        main_alpha = oof_stack_modern_tuned if oof_stack_modern_tuned is not None else oof_stack_modern
        main_name = "stack_modernGB+RF->Ridge[TUNED]" if oof_stack_modern_tuned is not None else "stack_modernGB+RF->Ridge"
    if main_alpha is None or (not np.isfinite(main_alpha).any()):
        main_alpha = oof_stack_modern
        main_name = "stack_modernGB+RF->Ridge"
    if main_alpha is None or (not np.isfinite(main_alpha).any()):
        main_alpha = oof_stack_gbdt
        main_name = "stack_GBDT+RF->Ridge"
    if main_alpha is None or (not np.isfinite(main_alpha).any()):
        main_alpha = oof_alpha_gbdt
        main_name = "alpha_GBDT(sklearn)"

    print(f"\n==================== EXPORT MAIN MODEL ====================")
    print(f"MAIN_MODEL={main_name}")
    print(f"MODERN_GB_BACKEND_ALPHA={modern_gb_backend_alpha}")
    print(f"MODERN_GB_BACKEND_RISK={modern_gb_backend_risk}")
    if tuned_params:
        print(f"TUNED_PARAMS={tuned_params}")

    # 额外导出：线上推理用的最终模型包（joblib）
    final_path, _ = train_and_export_final_model(
        df=df,
        X_alpha_cols=X_alpha_cols,
        REGIME_COLS=REGIME_COLS,
        X_risk_cols=X_risk_cols,
        tuned_params=tuned_params,
        out_path=FINAL_MODEL_PATH
    )
    print(f"FINAL_MODEL_SAVED={final_path}")

    mask = np.isfinite(main_alpha) & np.isfinite(oof_vol_pred) & np.isfinite(oof_regime)
    df_oof = df.loc[mask, ["ticker", "trade_date", "y_ret_k"]].copy()
    df_oof["alpha"] = float(SCORE_SIGN) * main_alpha[mask]
    df_oof["vol_pred"] = oof_vol_pred[mask]
    df_oof["regime"] = oof_regime[mask].astype(int)

    df_oof.to_csv(OUT_ARTIFACT, index=False, encoding="utf-8-sig")
    print("saved:", OUT_ARTIFACT, "rows=", len(df_oof))

    df_alpha = df_oof[["ticker", "trade_date", "alpha", "vol_pred", "regime", "y_ret_k"]].copy()
    df_alpha.to_csv(OUT_ALPHA_OOF, index=False, encoding="utf-8-sig")
    print("saved:", OUT_ALPHA_OOF, "rows=", len(df_alpha))

    df_stack = df.loc[mask, ["ticker", "trade_date", "y_ret_k"]].copy()
    df_stack["oof_modern_gb"] = (oof_alpha_modern[mask] if oof_alpha_modern is not None else np.nan)
    df_stack["oof_gbdt"] = (oof_alpha_gbdt[mask] if oof_alpha_gbdt is not None else np.nan)
    df_stack["oof_rf"] = (oof_alpha_rf[mask] if oof_alpha_rf is not None else np.nan)
    df_stack["oof_stack_modern"] = (oof_stack_modern[mask] if oof_stack_modern is not None else np.nan)
    df_stack["oof_stack_gbdt"] = (oof_stack_gbdt[mask] if oof_stack_gbdt is not None else np.nan)
    if oof_stack_modern_tuned is not None:
        df_stack["oof_stack_modern_tuned"] = oof_stack_modern_tuned[mask]
    df_stack["oof_vol_pred"] = oof_vol_pred[mask]
    df_stack["oof_regime"] = oof_regime[mask]
    df_stack.to_csv(OUT_STACK_OOF, index=False, encoding="utf-8-sig")
    print("saved:", OUT_STACK_OOF, "rows=", len(df_stack))

    # 给后续“对齐自检”补齐不同收益对齐列（不影响已导出的CSV）
    if "y_ret_k_lag1" in df.columns:
        df_oof["y_ret_k_lag1"] = df.loc[mask, "y_ret_k_lag1"].values.astype(float)
    if "y_ret_k_lead1" in df.columns:
        df_oof["y_ret_k_lead1"] = df.loc[mask, "y_ret_k_lead1"].values.astype(float)

    # =========================
    # ======= 报告输出区 =======
    # =========================
    best_topk_pack = pick_best_topk_for_main(df_oof, cost_bps=COST_BPS)
    BEST_TOPK = best_topk_pack["topk"]
    print(f"\n==================== BEST TOPK HOLDINGS ====================")
    print(f"BEST_TOPK={BEST_TOPK} (candidates={TOPK_HOLDINGS_CANDIDATES}) sharpe_net={best_topk_pack['sh']:.6f}")

    df_trade = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)

    gross, net, to = compute_portfolio_daily(
        df_trade, pred_col="integrated_score", ret_col="y_ret_k",
        long_short=LONG_SHORT, cap=CAP_WEIGHT,
        min_spread=MIN_SPREAD, cost_bps=COST_BPS,
        pos_alpha=POSITION_SMOOTH_ALPHA, max_to=MAX_DAILY_TURNOVER,
        rebalance_every=REBALANCE_EVERY,
        regime_col="regime", regime_mult=REGIME_MULT,
        vol_pred_col=None,
        topk_holdings=BEST_TOPK,
        min_cross_section=MIN_CROSS_SECTION,
    )

    if USE_PORTFOLIO_OVERLAY:
        gross_ov, net_ov, lev = apply_portfolio_overlay(
            gross, net,
            target_vol_ann=TARGET_PORT_VOL_ANN,
            window=PORT_VOL_WINDOW,
            lev_cap=LEVERAGE_CAP,
            lev_floor=LEVERAGE_FLOOR,
            dd_start=DD_THROTTLE_START,
            dd_end=DD_THROTTLE_END,
            ann=ANN
        )
    else:
        gross_ov, net_ov = gross, net
        lev = np.ones_like(net_ov, dtype=float)

    dates_daily = np.array(sorted(df_trade["trade_date"].unique()))
    df_daily = pd.DataFrame({
        "trade_date": dates_daily,
        "gross": gross_ov,
        "net": net_ov,
        "turnover": to,
        "lev_overlay": lev,
        "eq_gross": equity_curve(gross_ov),
        "eq_net": equity_curve(net_ov),
    })
    out_daily_csv = str(Path(OUT_REPORT_DIR) / "daily_gross_net_equity.csv")
    df_daily.to_csv(out_daily_csv, index=False, encoding="utf-8-sig")

    out_eq_png = str(Path(OUT_REPORT_DIR) / "equity_curve_gross_vs_net.png")
    plot_equity(df_daily["trade_date"], df_daily["gross"].values, df_daily["net"].values, out_eq_png)

    # === Performance Charts: Benchmark / Turnover / Leverage ===
    bench = compute_equal_weight_benchmark(df.loc[mask, ["trade_date", "y_ret_k"]].copy(), ret_col="y_ret_k")
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench = bench.dropna(subset=["trade_date"]).sort_values("trade_date")

    df_daily = df_daily.merge(bench, on="trade_date", how="left")
    df_daily["bench_ret"] = df_daily["bench_ret"].fillna(0.0).astype(float)

    out_bench_png = str(Path(OUT_REPORT_DIR) / "equity_vs_benchmark.png")
    plot_equity_vs_benchmark(df_daily["trade_date"], df_daily["net"].values, df_daily["bench_ret"].values,
                             out_bench_png)

    out_to_png = str(Path(OUT_REPORT_DIR) / "turnover_series.png")
    plot_series(df_daily["trade_date"], df_daily["turnover"].values, out_to_png, "Turnover Series", ylabel="turnover")

    out_lev_png = str(Path(OUT_REPORT_DIR) / "leverage_series.png")
    plot_series(df_daily["trade_date"], df_daily["lev_overlay"].values, out_lev_png, "Leverage Series (Overlay)",
                ylabel="leverage")

    # === 分布统计 ===
    stats_g = daily_stats(df_daily["gross"].values)
    stats_n = daily_stats(df_daily["net"].values)
    df_stats = pd.DataFrame([{"series": "gross", **stats_g}, {"series": "net", **stats_n}])
    out_stats_csv = str(Path(OUT_REPORT_DIR) / "daily_return_distribution_stats.csv")
    df_stats.to_csv(out_stats_csv, index=False, encoding="utf-8-sig")

    plot_hist(df_daily["gross"].values, "Daily Return Histogram (Gross)",
              str(Path(OUT_REPORT_DIR) / "hist_daily_gross.png"))
    plot_hist(df_daily["net"].values, "Daily Return Histogram (Net)", str(Path(OUT_REPORT_DIR) / "hist_daily_net.png"))

    # === 样本期间敏感性 + 滚动稳定性（新增） ===
    out_sample_csv = sample_period_sensitivity(df_daily, OUT_REPORT_DIR, ann=ANN)

    dd = df_daily.set_index("trade_date")
    mon = period_perf_table(dd, ret_col="net", freq="M", ann=ANN)
    qtr = period_perf_table(dd, ret_col="net", freq="Q", ann=ANN)

    out_mon_csv = str(Path(OUT_REPORT_DIR) / "monthly_performance.csv")
    out_qtr_csv = str(Path(OUT_REPORT_DIR) / "quarterly_performance.csv")
    mon.to_csv(out_mon_csv, index=False, encoding="utf-8-sig")
    qtr.to_csv(out_qtr_csv, index=False, encoding="utf-8-sig")

    df_daily["year"] = pd.to_datetime(df_daily["trade_date"]).dt.year
    by_year_rows = []
    for y, g in df_daily.groupby("year", sort=True):
        r = g["net"].values.astype(float)
        by_year_rows.append({
            "year": int(y),
            "n_days": int(np.isfinite(r).sum()),
            "sharpe_net": sharpe_from_daily(r, ann=ANN),
            "ann_ret_net": ann_return_from_daily(r, ann=ANN),
            "ann_vol_net": ann_vol_from_daily(r, ann=ANN),
            "mdd_net": max_drawdown_from_daily(r),
            "calmar_net": calmar_from_daily(r, ann=ANN),
        })
    df_by_year = pd.DataFrame(by_year_rows)
    out_year_csv = str(Path(OUT_REPORT_DIR) / "stability_by_year.csv")
    df_by_year.to_csv(out_year_csv, index=False, encoding="utf-8-sig")

    daily_regime = df_oof.groupby("trade_date")["regime"].agg(lambda s: int(s.value_counts().idxmax()) if len(s.dropna()) else 0).reset_index()
    df_daily = df_daily.merge(daily_regime, on="trade_date", how="left").rename(columns={"regime": "daily_regime_mode"})

    by_reg_rows = []
    for rid, g in df_daily.groupby("daily_regime_mode", sort=True):
        r = g["net"].values.astype(float)
        by_reg_rows.append({
            "daily_regime_mode": int(rid),
            "n_days": int(np.isfinite(r).sum()),
            "sharpe_net": sharpe_from_daily(r, ann=ANN),
            "ann_ret_net": ann_return_from_daily(r, ann=ANN),
            "ann_vol_net": ann_vol_from_daily(r, ann=ANN),
            "mdd_net": max_drawdown_from_daily(r),
            "calmar_net": calmar_from_daily(r, ann=ANN),
        })
    df_by_reg = pd.DataFrame(by_reg_rows)
    out_reg_csv = str(Path(OUT_REPORT_DIR) / "stability_by_regime.csv")
    df_by_reg.to_csv(out_reg_csv, index=False, encoding="utf-8-sig")

    plt.figure()
    for rid, g in df_daily.groupby("daily_regime_mode", sort=True):
        eq = equity_curve(g["net"].values.astype(float))
        plt.plot(g["trade_date"], eq, label=f"regime={int(rid)}")
    plt.title("Equity Curve by Regime (Net, daily_mode)")
    plt.legend()
    _format_date_xaxis(plt.gca())
    plt.tight_layout()
    plt.savefig(str(Path(OUT_REPORT_DIR) / "equity_by_regime.png"), dpi=150)

    plt.close()

    cost_list = [0.0, 5.0, 10.0, 20.0]
    cost_rows = []
    for cb in cost_list:
        g2, n2, t2 = compute_portfolio_daily(
            df_trade, pred_col="integrated_score", ret_col="y_ret_k",
            long_short=LONG_SHORT, cap=CAP_WEIGHT,
            min_spread=MIN_SPREAD, cost_bps=float(cb),
            pos_alpha=POSITION_SMOOTH_ALPHA, max_to=MAX_DAILY_TURNOVER,
            rebalance_every=REBALANCE_EVERY,
            regime_col="regime", regime_mult=REGIME_MULT,
            vol_pred_col=None,
            topk_holdings=BEST_TOPK,
            min_cross_section=MIN_CROSS_SECTION,
        )
        if USE_PORTFOLIO_OVERLAY:
            _, n2o, _ = apply_portfolio_overlay(
                g2, n2,
                target_vol_ann=TARGET_PORT_VOL_ANN,
                window=PORT_VOL_WINDOW,
                lev_cap=LEVERAGE_CAP,
                lev_floor=LEVERAGE_FLOOR,
                dd_start=DD_THROTTLE_START,
                dd_end=DD_THROTTLE_END,
                ann=ANN
            )
        else:
            n2o = n2
        cost_rows.append({
            "cost_bps": float(cb),
            "sharpe_net": sharpe_from_daily(n2o, ann=ANN),
            "ann_ret_net": ann_return_from_daily(n2o, ann=ANN),
            "ann_vol_net": ann_vol_from_daily(n2o, ann=ANN),
            "mdd_net": max_drawdown_from_daily(n2o),
            "turnover_mean": float(np.nanmean(t2)) if np.isfinite(np.nanmean(t2)) else np.nan,
        })
    df_cost = pd.DataFrame(cost_rows)
    out_cost_csv = str(Path(OUT_REPORT_DIR) / "cost_sensitivity.csv")
    df_cost.to_csv(out_cost_csv, index=False, encoding="utf-8-sig")
    plot_cost_sensitivity(df_cost, str(Path(OUT_REPORT_DIR) / "cost_sensitivity_sharpe.png"))

    df_oof_ic = apply_signal_integration(df_oof, alpha_col="alpha", vol_col="vol_pred", gamma=VOL_GAMMA)
    ic_df, ic_sum = compute_ic_rankic(df_oof_ic, alpha_col="integrated_score", ret_col="y_ret_k",
                                      min_n=max(2, MIN_CROSS_SECTION))

    out_ic_csv = str(Path(OUT_REPORT_DIR) / "ic_rankic_daily.csv")
    ic_df.to_csv(out_ic_csv, index=False, encoding="utf-8-sig")

    out_ic_sum = str(Path(OUT_REPORT_DIR) / "ic_rankic_summary.json")
    Path(out_ic_sum).write_text(json.dumps(ic_sum, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_ic_series(ic_df, str(Path(OUT_REPORT_DIR) / "ic_timeseries.png"))
    plot_rankic_series(ic_df, str(Path(OUT_REPORT_DIR) / "rankic_timeseries.png"))

    plot_drawdown(df_daily["trade_date"], df_daily["gross"].values,
                  str(Path(OUT_REPORT_DIR) / "drawdown_gross.png"),
                  title="Drawdown (Gross)")
    plot_drawdown(df_daily["trade_date"], df_daily["net"].values,
                  str(Path(OUT_REPORT_DIR) / "drawdown_net.png"),
                  title="Drawdown (Net)")

    roll_sh, roll_vol = rolling_metrics(df_daily["net"].values, window=60, ann=ANN)
    plot_rolling(df_daily["trade_date"], roll_sh, str(Path(OUT_REPORT_DIR) / "rolling_sharpe_60d.png"),
                 "Rolling Sharpe (60d, Net)")
    plot_rolling(df_daily["trade_date"], roll_vol, str(Path(OUT_REPORT_DIR) / "rolling_vol_60d.png"),
                 "Rolling Vol (60d, Net, annualized)")

    df_spread = signal_quantile_spread(df_trade, score_col="integrated_score", ret_col="y_ret_k", q=10)
    df_spread.to_csv(str(Path(OUT_REPORT_DIR) / "signal_quantile_spread_q10.csv"), index=False, encoding="utf-8-sig")

    if not df_spread.empty:
        spread_sum = {
            "days": int(df_spread["trade_date"].nunique()),
            "mean_spread": float(np.nanmean(df_spread["spread"].values.astype(float))),
            "std_spread": float(np.nanstd(df_spread["spread"].values.astype(float), ddof=1)),
            "sharpe_spread": sharpe_from_daily(df_spread["spread"].values.astype(float), ann=ANN),
        }
        Path(str(Path(OUT_REPORT_DIR) / "signal_quantile_spread_q10_summary.json")).write_text(
            json.dumps(spread_sum, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        plt.figure()
        plt.plot(df_spread["trade_date"], equity_curve(df_spread["spread"].values), label="Top-Bottom Spread Equity")
        plt.title("Signal Quantile Spread (q=10): Top - Bottom")
        plt.legend()
        _format_date_xaxis(plt.gca())
        plt.tight_layout()
        plt.savefig(str(Path(OUT_REPORT_DIR) / "signal_quantile_spread_equity.png"), dpi=150)

        plt.close()

    out_sens_csv = run_param_sensitivity(df_trade_base=df_oof, out_dir=OUT_REPORT_DIR)

    # ========= 论文/排雷补充输出：消融 / 对齐自检 / TopK稳健性 =========
    out_ablation_csv, _ = run_ablation(df_oof, OUT_REPORT_DIR, topk=BEST_TOPK, ann=ANN)
    out_align_csv, _ = run_alignment_selfcheck(df_oof, OUT_REPORT_DIR, topk=BEST_TOPK, ann=ANN)
    out_topk_csv, _ = run_topk_robustness(df_oof, OUT_REPORT_DIR, topk_list=[1, 2, 3, 5, 10], ann=ANN)
    # === Backtest Report (HTML) ===
    out_html = generate_backtest_report(
        out_dir=OUT_REPORT_DIR,
        main_name=main_name,
        tuned_params=tuned_params,
        best_topk=BEST_TOPK,
        df_cmp=df_cmp,
        ic_sum=ic_sum
    )

    print("\n==================== REPORT OUTPUTS ====================")
    print("REPORT_DIR:", OUT_REPORT_DIR)
    print("saved:", out_daily_csv)
    print("saved:", out_eq_png)
    print("saved:", str(Path(OUT_REPORT_DIR) / "equity_vs_benchmark.png"))
    print("saved:", str(Path(OUT_REPORT_DIR) / "turnover_series.png"))
    print("saved:", str(Path(OUT_REPORT_DIR) / "leverage_series.png"))
    print("saved:", out_stats_csv)
    print("saved:", out_mon_csv)
    print("saved:", out_qtr_csv)
    print("saved:", out_year_csv)
    print("saved:", out_reg_csv)
    print("saved:", out_cost_csv)
    print("saved:", out_ic_csv)
    print("saved:", out_ic_sum)
    print("saved:", str(Path(OUT_REPORT_DIR) / "drawdown_net.png"))
    print("saved:", str(Path(OUT_REPORT_DIR) / "rolling_sharpe_60d.png"))
    print("saved:", str(Path(OUT_REPORT_DIR) / "rolling_sharpe_126d.png"))
    print("saved:", str(Path(OUT_REPORT_DIR) / "rolling_sharpe_252d.png"))
    print("saved:", str(Path(OUT_REPORT_DIR) / "signal_quantile_spread_equity.png"))
    print("saved:", out_sens_csv)
    print("saved:", out_ablation_csv)
    print("saved:", out_align_csv)
    print("saved:", out_topk_csv)
    print("saved:", str(Path(OUT_REPORT_DIR) / "sample_period_sensitivity.csv"))
    print("saved:", out_html)

    print("\n==================== IC/RANKIC SUMMARY ====================")
    print(json.dumps(ic_sum, ensure_ascii=False, indent=2))

    input("保留结果暂定用.................................................")
