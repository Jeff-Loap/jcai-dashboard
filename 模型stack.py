import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# 可选（若已安装）
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

# =========================
# GPU 开关（XGB: CUDA，LGBM: OpenCL GPU）
# =========================
USE_GPU = True
EARLY_STOP = 50

def make_model(name, ModelCls, params, use_gpu=True):
    p = dict(params)

    if use_gpu and name == "XGB":
        p.update({
            "tree_method": "hist",
            "device": "cuda",
        })
        p.pop("gpu_id", None)
        p.pop("predictor", None)

    if use_gpu and name == "LGBM":
        p.update({
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "max_bin": 255,
            "force_col_wise": True,
        })

    return ModelCls(**p), p

def fit_one(mdl, name, X_tr, y_tr, X_te, y_te):
    if name == "XGB":
        try:
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False,
                early_stopping_rounds=EARLY_STOP
            )
        except TypeError:
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False
            )

    elif name == "LGBM":
        try:
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                eval_metric="l2",
                early_stopping_rounds=EARLY_STOP
            )
        except TypeError:
            try:
                import lightgbm as lgb
                mdl.fit(
                    X_tr, y_tr,
                    eval_set=[(X_te, y_te)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
                )
            except Exception:
                mdl.fit(X_tr, y_tr)
    else:
        mdl.fit(X_tr, y_tr)

    return mdl

def fit_with_gpu_fallback(name, ModelCls, params, X_tr, y_tr, X_te, y_te):
    if USE_GPU and name in ("XGB", "LGBM"):
        try:
            mdl, used = make_model(name, ModelCls, params, use_gpu=True)
            return fit_one(mdl, name, X_tr, y_tr, X_te, y_te), used, True
        except Exception as e:
            msg = str(e).lower()
            if ("gpu" in msg) or ("cuda" in msg) or ("opencl" in msg) or ("device" in msg):
                print(f"[{name}] GPU 不可用，回退 CPU。err={e}")
            else:
                raise

    mdl, used = make_model(name, ModelCls, params, use_gpu=False)
    return fit_one(mdl, name, X_tr, y_tr, X_te, y_te), used, False


# ===== 1) 读取数据 =====
df = pd.read_csv(r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features.csv")
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()

key_cols = ["ticker", "trade_date"]
label_cols = ["y_ret_k", "y_up_k", "y_vol_k"]
X_cols = [c for c in df.columns if c not in key_cols + label_cols]

df = df.sort_values(["trade_date", "ticker"]).reset_index(drop=True)

# ===== 1.1) 贴合你文件的 y 分布：winsorize（比固定 [-0.1, 0.1] 更合适）=====
y_raw = df["y_ret_k"].values.astype(float)
WINSOR_Q = 0.01
lo, hi = np.quantile(y_raw, [WINSOR_Q, 1.0 - WINSOR_Q])
y = np.clip(y_raw, lo, hi)
df["_y"] = y

# ===== 1.2) 特征矩阵 =====
use_interaction = False
if use_interaction:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X = poly.fit_transform(df[X_cols].values)
else:
    X = df[X_cols].values


# ===== 2) 定义模型类 + 参数网格 =====
models = {}
param_grids = {}

models["GBDT"] = GradientBoostingRegressor
param_grids["GBDT"] = [
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "random_state": 42},
    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "random_state": 42},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8, "random_state": 42},
]

if XGBRegressor is not None:
    models["XGB"] = XGBRegressor
    param_grids["XGB"] = [
        {
            "n_estimators": 800, "learning_rate": 0.05, "max_depth": 4,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2,
            "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": -1,
            "random_state": 42, "eval_metric": "rmse",
        },
        {
            "n_estimators": 1200, "learning_rate": 0.05, "max_depth": 4,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2,
            "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": -1,
            "random_state": 42, "eval_metric": "rmse",
        },
        {
            "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 5,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2,
            "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": -1,
            "random_state": 42, "eval_metric": "rmse",
        },
    ]

if LGBMRegressor is not None:
    models["LGBM"] = LGBMRegressor
    param_grids["LGBM"] = [
        {
            "n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
            "objective": "regression", "n_jobs": -1, "random_state": 42,
        },
        {
            "n_estimators": 2500, "learning_rate": 0.05, "num_leaves": 31,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
            "objective": "regression", "n_jobs": -1, "random_state": 42,
        },
        {
            "n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
            "objective": "regression", "n_jobs": -1, "random_state": 42,
        },
    ]


# ===== 3) 按 trade_date 做 TimeSeriesSplit =====
dates = np.array(sorted(df["trade_date"].unique()))
tscv_dates = TimeSeriesSplit(n_splits=5)

def iter_date_splits():
    for tr_didx, te_didx in tscv_dates.split(dates):
        tr_dates = dates[tr_didx]
        te_dates = dates[te_didx]
        tr_idx = np.where(df["trade_date"].isin(tr_dates).values)[0]
        te_idx = np.where(df["trade_date"].isin(te_dates).values)[0]
        yield tr_idx, te_idx


# ===== 4) 组合收益：适配你文件（10 只票、每天数量不稳定）=====
LONG_SHORT = True           # True=多空，False=只做多
TOP_FRAC = 0.2              # 每天选前/后 20%（对 10 只票≈2，对 5~6 只票≈1）
K_MIN = 1                   # 最少选 1
K_MAX = 3                   # 最多选 3（你的 universe=10，别选太大）
MIN_TICKERS_LS = 4          # 多空至少要 >=4 才能 1v1
PRED_SMOOTH_WIN = 1         # 你票池太小，先别平滑；需要再改 3
MIN_SPREAD = 0.0            # 先不过滤；需要再改小正数

def add_pred_smooth(df_in, pred_col="pred", window=PRED_SMOOTH_WIN):
    if window is None or window <= 1:
        return df_in[pred_col].values
    d = df_in.sort_values(["ticker", "trade_date"])
    s = d.groupby("ticker", sort=False)[pred_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    out = pd.Series(index=d.index, data=s).reindex(df_in.index)
    return out.values

def compute_daily_ret_port(df_in, pred_col="pred", ret_col="_y",
                           long_short=True, top_frac=0.2, k_min=1, k_max=3,
                           min_tickers_ls=4, min_spread=0.0):
    daily = []
    for dt, g in df_in.groupby("trade_date", sort=False):
        gg = g[[pred_col, ret_col]].dropna()
        n = len(gg)
        if n == 0:
            daily.append(0.0)
            continue

        if long_short:
            if n < int(min_tickers_ls):
                daily.append(0.0)
                continue
            kk = int(round(n * float(top_frac)))
            kk = max(int(k_min), kk)
            kk = min(int(k_max), kk)
            kk = min(kk, n // 2)   # 关键：保证多空不重叠
            if kk < 1:
                daily.append(0.0)
                continue

            longs = gg.nlargest(kk, pred_col)
            shorts = gg.nsmallest(kk, pred_col)

            if min_spread and min_spread > 0:
                spread = float(longs[pred_col].mean() - shorts[pred_col].mean())
                if spread < float(min_spread):
                    daily.append(0.0)
                    continue

            r = float(longs[ret_col].mean() - shorts[ret_col].mean())
            daily.append(r)
        else:
            kk = int(round(n * float(top_frac)))
            kk = max(int(k_min), kk)
            kk = min(int(k_max), kk)
            kk = min(kk, n)
            if kk < 1:
                daily.append(0.0)
                continue
            longs = gg.nlargest(kk, pred_col)
            r = float(longs[ret_col].mean())
            daily.append(r)

    return np.asarray(daily, dtype=float)

def sharpe_from_daily(daily, ann=252.0):
    daily = np.asarray(daily, dtype=float)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 20:
        return np.nan
    s = daily.std(ddof=1)
    if not np.isfinite(s) or s <= 0:
        return np.nan
    return daily.mean() / s * np.sqrt(ann)


# ===== 5) 外层时间序列交叉验证（产生 OOF） + 参数搜索 =====
results = {}

for name, ModelCls in models.items():
    print(f"\n==================== {name} ====================")
    model_param_results = []

    for param_id, params in enumerate(param_grids[name], start=1):
        print(f"\n--- {name} param set #{param_id} ---")
        print("params:", params)

        oof = np.full(len(df), np.nan)
        fold_id = 0
        used_gpu_any = False

        for tr_idx, te_idx in iter_date_splits():
            fold_id += 1
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            t0 = time.time()
            mdl, used_params, used_gpu = fit_with_gpu_fallback(
                name, ModelCls, params, X_tr, y_tr, X_te, y_te
            )
            used_gpu_any = used_gpu_any or used_gpu

            print(
                f"[{name}] param#{param_id} fold {fold_id}: "
                f"train_size={len(tr_idx)}, valid_size={len(te_idx)}, gpu={used_gpu}, "
                f"time={time.time()-t0:.2f}s"
            )
            oof[te_idx] = mdl.predict(X_te)

        mask = ~np.isnan(oof)
        mse = mean_squared_error(y[mask], oof[mask])

        df_valid = df.iloc[mask].copy()
        df_valid["pred"] = oof[mask]
        df_valid["pred_s"] = add_pred_smooth(df_valid, pred_col="pred", window=PRED_SMOOTH_WIN)

        # point 口径（保留，但不拿它做主指标）
        pnl_point = np.sign(df_valid["pred_s"].values) * df_valid["_y"].values
        ann = 252.0
        s_point = pnl_point.std(ddof=1)
        sharpe_point = (pnl_point.mean() / s_point * np.sqrt(ann)) if s_point > 0 else np.nan

        # 组合口径（适配小票池/不稳定票数）
        daily_ret = compute_daily_ret_port(
            df_valid,
            pred_col="pred_s",
            ret_col="_y",
            long_short=LONG_SHORT,
            top_frac=TOP_FRAC,
            k_min=K_MIN,
            k_max=K_MAX,
            min_tickers_ls=MIN_TICKERS_LS,
            min_spread=MIN_SPREAD
        )
        sharpe_port = sharpe_from_daily(daily_ret, ann=ann)

        record = dict(params)
        record.update({
            "MSE": mse,
            "Sharpe_point": sharpe_point,
            "Sharpe_port": sharpe_port,
            "used_gpu": int(used_gpu_any)
        })
        model_param_results.append(record)

    df_params = pd.DataFrame(model_param_results)
    print(f"\n参数结果表 - {name}:")
    print(df_params)
    df_params.to_csv(f"D:/PythonFile/JCAI/data_ak/{name}_param_log.csv", index=False)

    # 选参优先 Sharpe_port
    if df_params["Sharpe_port"].notna().any():
        metric_col = "Sharpe_port"
        best_idx = df_params[metric_col].idxmax()
    elif df_params["Sharpe_point"].notna().any():
        metric_col = "Sharpe_point"
        best_idx = df_params[metric_col].idxmax()
    else:
        metric_col = "MSE"
        best_idx = df_params[metric_col].idxmin()

    best_row = df_params.loc[best_idx]
    print(f"\n最佳参数（按 {metric_col} 选出） - {name}:")
    print(best_row.to_dict())

    best_params = {k: best_row[k] for k in param_grids[name][0].keys()}
    int_params = ["n_estimators", "max_depth", "num_leaves", "min_child_samples", "random_state"]
    for p in int_params:
        if p in best_params and not pd.isna(best_params[p]):
            best_params[p] = int(best_params[p])

    best_row_dict = best_row.to_dict()
    best_row_dict["best_params"] = best_params
    best_row_dict["selected_by"] = metric_col
    results[name] = best_row_dict

print("\nSummary best params by model:")
print(results)


# ===== 6) stacking：base OOF + meta（用 OOF 日期再做一层 time split，避免你之前少一整段）=====
stack_oofs = []
stack_mask = None

for name, ModelCls in models.items():
    best_params = results[name]["best_params"]
    print(f"\n[Stacking] 使用 {name} 的最佳参数重新生成 OOF：", best_params)

    oof = np.full(len(df), np.nan)
    used_gpu_any = False

    for tr_idx, te_idx in iter_date_splits():
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        mdl, used_params, used_gpu = fit_with_gpu_fallback(
            name, ModelCls, best_params, X_tr, y_tr, X_te, y_te
        )
        used_gpu_any = used_gpu_any or used_gpu
        oof[te_idx] = mdl.predict(X_te)

    mask = ~np.isnan(oof)
    stack_mask = mask if stack_mask is None else (stack_mask & mask)
    stack_oofs.append((name, oof))
    print(f"[Stacking] {name} used_gpu={used_gpu_any}")

if stack_oofs and stack_mask is not None and stack_mask.any():
    # base oof 特征
    X_stack_all = np.vstack([o[1] for o in stack_oofs]).T  # (len(df), n_models)

    # 仅在有 base OOF 的日期上再做一层 time split（避免你之前 meta 少一大段）
    oof_dates = np.array(sorted(df.loc[stack_mask, "trade_date"].unique()))
    meta_splits = 9 if len(oof_dates) >= 600 else 5
    tscv_meta = TimeSeriesSplit(n_splits=meta_splits)

    meta_oof = np.full(len(df), np.nan)

    for tr_didx, te_didx in tscv_meta.split(oof_dates):
        tr_dates = oof_dates[tr_didx]
        te_dates = oof_dates[te_didx]

        tr_idx = np.where(stack_mask & df["trade_date"].isin(tr_dates).values)[0]
        te_idx = np.where(stack_mask & df["trade_date"].isin(te_dates).values)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        meta = Ridge(alpha=1.0)
        meta.fit(X_stack_all[tr_idx], y[tr_idx])
        meta_oof[te_idx] = meta.predict(X_stack_all[te_idx])

    mask_meta = stack_mask & ~np.isnan(meta_oof)

    stack_pred = meta_oof[mask_meta]
    y_stack = y[mask_meta]
    mse_stack = mean_squared_error(y_stack, stack_pred)

    df_valid_stack = df.iloc[mask_meta].copy()
    df_valid_stack["pred"] = stack_pred
    df_valid_stack["pred_s"] = add_pred_smooth(df_valid_stack, pred_col="pred", window=PRED_SMOOTH_WIN)

    pnl_point = np.sign(df_valid_stack["pred_s"].values) * df_valid_stack["_y"].values
    ann = 252.0
    s_point = pnl_point.std(ddof=1)
    sharpe_point_stack = (pnl_point.mean() / s_point * np.sqrt(ann)) if s_point > 0 else np.nan

    daily_ret_stack = compute_daily_ret_port(
        df_valid_stack,
        pred_col="pred_s",
        ret_col="_y",
        long_short=LONG_SHORT,
        top_frac=TOP_FRAC,
        k_min=K_MIN,
        k_max=K_MAX,
        min_tickers_ls=MIN_TICKERS_LS,
        min_spread=MIN_SPREAD
    )
    sharpe_port_stack = sharpe_from_daily(daily_ret_stack, ann=ann)

    alpha_out = df_valid_stack[["ticker", "trade_date", "pred", "_y"]].copy()
    alpha_out.to_csv(r"D:\PythonFile\JCAI\data_ak\alpha_signal_oof.csv", index=False)
    print("saved:", r"D:\PythonFile\JCAI\data_ak\alpha_signal_oof.csv", "rows=", len(alpha_out))

    print("\n[Stacking] 结果：")
    print(f"MSE={mse_stack:.6f}, Sharpe_point={sharpe_point_stack:.4f}, Sharpe_port={sharpe_port_stack:.4f}")
else:
    print("\n[Stacking] 可用于 stacking 的基模型不足，未计算。")

print("\n==================== Final Summary ====================")
print(f"Base models run = {len(results)} , models = {list(results.keys())}")

rows = []
for m, info in results.items():
    rows.append({
        "Model": m,
        "selected_by": info.get("selected_by", ""),
        "MSE": float(info.get("MSE", np.nan)),
        "Sharpe_point": float(info.get("Sharpe_point", np.nan)),
        "Sharpe_port": float(info.get("Sharpe_port", np.nan)),
        "used_gpu": int(info.get("used_gpu", 0)),
    })

df_final = pd.DataFrame(rows, columns=["Model", "selected_by", "MSE", "Sharpe_point", "Sharpe_port", "used_gpu"])
print(df_final.to_string(index=False))
