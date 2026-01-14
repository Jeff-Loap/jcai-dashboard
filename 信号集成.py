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


def sharpe_from_daily(daily, ann=252.0):
    daily = np.asarray(daily, dtype=float)
    s = daily.std(ddof=1)
    if len(daily) < 20 or s <= 0:
        return np.nan
    return float(daily.mean() / s * np.sqrt(ann))


# ===== 1) 读取数据 =====
df = pd.read_csv(r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features.csv")
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()

key_cols = ["ticker", "trade_date"]
label_cols = ["y_ret_k", "y_up_k", "y_vol_k"]
X_cols = [c for c in df.columns if c not in key_cols + label_cols]

# 关键：按日期聚合形成横截面（避免按行切分导致单ticker落入fold）
df = df.sort_values(["trade_date", "ticker"]).reset_index(drop=True)

# 对未来收益做简单去极值，有助于模型稳定
y_raw = df["y_ret_k"].values.astype(float)
y_clipped = np.clip(y_raw, -0.10, 0.10)
df["_y"] = y_clipped


# ===== 1.1) 只选“效果最好”的 Top3 ticker 再做 stacking（可关）=====
USE_TOPK_TICKERS = True
TOPK = 3
PICK_FRAC = 0.70  # 用最早 70% 的日期来挑 ticker（避免用到最后的未来）

if USE_TOPK_TICKERS:
    dates_all = np.array(sorted(df["trade_date"].dropna().unique()))
    cut = dates_all[max(0, int(len(dates_all) * PICK_FRAC) - 1)]
    df_pick = df[df["trade_date"] <= cut].copy()

    rows_pick = []
    for tkr, sub in df_pick.groupby("ticker", sort=False):
        daily = sub["_y"].values.astype(float)
        s = np.std(daily, ddof=1) if len(daily) > 1 else 0.0
        if len(daily) < 20 or s <= 0:
            continue
        sh = float(np.mean(daily) / s * np.sqrt(252.0))
        rows_pick.append((tkr, sh, int(len(daily))))

    if len(rows_pick) < TOPK:
        raise RuntimeError(f"可用 ticker 不足 {TOPK} 个（挑选窗口内有效样本太少）")

    pick_df = pd.DataFrame(rows_pick, columns=["ticker", "bh_sharpe", "n"]).sort_values("bh_sharpe", ascending=False)
    top_tickers = pick_df.head(TOPK)["ticker"].tolist()
    print("[TopK pick] cutoff_date=", str(cut), "Top tickers=", top_tickers)
    print(pick_df.head(TOPK).to_string(index=False))

    df = df[df["ticker"].isin(top_tickers)].copy()
    df = df.sort_values(["trade_date", "ticker"]).reset_index(drop=True)

# 最终训练用的 y / X
y = df["_y"].values.astype(float)

# 是否使用二阶交互特征
use_interaction = False  # 需要时改成 True
if use_interaction:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X = poly.fit_transform(df[X_cols].values)
else:
    X = df[X_cols].values


# ===== 2) 定义模型类 + 参数网格 =====
models = {}
param_grids = {}

# GBDT
models["GBDT"] = GradientBoostingRegressor
param_grids["GBDT"] = [
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "random_state": 42},
    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "random_state": 42},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8, "random_state": 42},
]

# XGBoost
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

# LightGBM
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


# ===== 4) 组合收益：多头-only（每天买预测最高的 1 只）=====
def compute_daily_ret_long_only_top1(df_in, pred_col="pred", ret_col="_y"):
    g = df_in.groupby("trade_date", sort=False)
    idx_long = g[pred_col].idxmax()
    daily = df_in.loc[idx_long, ret_col].values.astype(float)
    return daily


# ===== 5) 外层时间序列交叉验证（产生 OOF） + 参数搜索 =====
results = {}

for name, ModelCls in models.items():
    print(f"\n==================== {name} ====================")
    model_param_results = []

    for param_id, params in enumerate(param_grids[name], start=1):
        print(f"\n--- {name} param set #{param_id} ---")
        print("params:", params)

        mdl = ModelCls(**params)
        oof = np.full(len(df), np.nan)

        fold_id = 0
        for tr_idx, te_idx in iter_date_splits():
            fold_id += 1
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]

            t0 = time.time()
            if name == "XGB":
                mdl.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            elif name == "LGBM":
                try:
                    mdl.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="l2", early_stopping_rounds=100)
                except TypeError:
                    mdl.fit(X_tr, y_tr)
            else:
                mdl.fit(X_tr, y_tr)

            print(
                f"[{name}] param#{param_id} fold {fold_id}: "
                f"train_size={len(tr_idx)}, valid_size={len(te_idx)}, time={time.time()-t0:.2f}s"
            )
            oof[te_idx] = mdl.predict(X_te)

        mask = ~np.isnan(oof)
        mse = mean_squared_error(y[mask], oof[mask])

        pnl_point = np.sign(oof[mask]) * y[mask]
        sharpe_point = (pnl_point.mean() / pnl_point.std(ddof=1) * np.sqrt(252.0)) if pnl_point.std(ddof=1) > 0 else np.nan

        df_valid = df.iloc[mask].copy()
        df_valid["pred"] = oof[mask]
        daily_ret = compute_daily_ret_long_only_top1(df_valid, pred_col="pred", ret_col="_y")
        sharpe_port = sharpe_from_daily(daily_ret, ann=252.0)

        record = dict(params)
        record.update({"MSE": mse, "Sharpe_point": sharpe_point, "Sharpe_port": sharpe_port})
        model_param_results.append(record)

    df_params = pd.DataFrame(model_param_results)
    df_params.to_csv(f"D:/PythonFile/JCAI/data_ak/{name}_param_log.csv", index=False)

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


# ===== 6) stacking（Ridge meta）=====
stack_oofs = []
stack_mask = None

for name, ModelCls in models.items():
    best_params = results[name]["best_params"]
    print(f"\n[Stacking] 使用 {name} 的最佳参数重新生成 OOF：", best_params)

    mdl = ModelCls(**best_params)
    oof = np.full(len(df), np.nan)

    for tr_idx, te_idx in iter_date_splits():
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        if name == "XGB":
            mdl.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        elif name == "LGBM":
            try:
                mdl.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="l2", early_stopping_rounds=100)
            except TypeError:
                mdl.fit(X_tr, y_tr)
        else:
            mdl.fit(X_tr, y_tr)

        oof[te_idx] = mdl.predict(X_te)

    mask = ~np.isnan(oof)
    stack_mask = mask if stack_mask is None else (stack_mask & mask)
    stack_oofs.append((name, oof))

if stack_oofs and stack_mask is not None and stack_mask.any():
    X_stack = np.vstack([o[1][stack_mask] for o in stack_oofs]).T
    y_stack = y[stack_mask]

    meta = Ridge(alpha=1.0)
    meta.fit(X_stack, y_stack)
    stack_pred = meta.predict(X_stack)

    mse_stack = mean_squared_error(y_stack, stack_pred)

    pnl_point = np.sign(stack_pred) * y_stack
    sharpe_point_stack = (pnl_point.mean() / pnl_point.std(ddof=1) * np.sqrt(252.0)) if pnl_point.std(ddof=1) > 0 else np.nan

    df_valid_stack = df.iloc[stack_mask].copy()
    df_valid_stack["pred"] = stack_pred

    alpha_out = df_valid_stack[["ticker", "trade_date", "pred", "_y"]].copy()
    alpha_out.to_csv(r"D:\PythonFile\JCAI\data_ak\alpha_signal_oof.csv", index=False)
    print("saved:", r"D:\PythonFile\JCAI\data_ak\alpha_signal_oof.csv", "rows=", len(alpha_out))

    daily_ret_stack = compute_daily_ret_long_only_top1(df_valid_stack, pred_col="pred", ret_col="_y")
    sharpe_port_stack = sharpe_from_daily(daily_ret_stack, ann=252.0)

    print("\n[Stacking] 结果：")
    print(f"MSE={mse_stack:.6f}, Sharpe_point={sharpe_point_stack:.4f}, Sharpe_port={sharpe_port_stack:.4f}")
else:
    print("\n[Stacking] 可用于 stacking 的基模型不足，未计算。")
