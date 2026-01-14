import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# ===============================
# 0) 特征工程（Factor Selection & Feature Engineering）
# ===============================
def feature_engineering_from_file(in_path: str,
                                  out_dir: str = "./outputs",
                                  out_name: str = "dataquant_features.csv",
                                  ts_col: str = "trade_date",
                                  id_col: str = "ticker",
                                  price_cols=("open","high","low","close"),
                                  z_windows=(60,120)) -> Tuple[str, dict]:
    """
    严格按提案方向构造特征：
      1) 统一滞后 _L1（防泄漏）
      2) 分资产 winsorize(1%/99%)
      3) 分资产滚动Z标准化（z60/z120）
      4) 导出 Alpha 全量特征 + 标签(y_ret_k, y_up_k, y_vol_k)
    返回：导出CSV路径、以及分块DataFrame字典（X_alpha/X_regime/X_risk/factor_pool/labels）
    """
    df = pd.read_csv(in_path)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values([id_col, ts_col]).copy()
    g = df.groupby(id_col, group_keys=False)

    # —— 自动识别：仅使用文件中已有的内生因子/技术指标/替代特征 ——
    base_tech = [
        "ret_1","ret_5","ret_10","ret_20",
        "vol_10","vol_20",
        "sma_5","sma_10","sma_20",
        "ema_12","ema_26","ma_gap_5_20","ema_gap_12_26",
        "macd","macd_signal","macd_hist","rsi","kdj_k","kdj_d",
        "atr","pdi","mdi","adx",
        "bb_mid","bb_up","bb_dn","bb_bw","bb_pb",
        "obv","vpt","mfi","cci",
        "vwap_win","z_close_20","z_turnover_20",
        "sentiment_score","bid1","ask1","bid1_size","ask1_size"
    ]
    present = [c for c in base_tech if c in df.columns]

    # —— 基础价量派生（若缺则补齐） ——
    o,h,l,c = price_cols
    if "range_hl" not in df.columns and all(x in df.columns for x in (h,l,c)):
        df["range_hl"] = (df[h] - df[l]) / df[c].replace(0, np.nan)
        present.append("range_hl")
    if "gap_oc" not in df.columns and all(x in df.columns for x in (o,c)):
        prev_c = g[c].shift(1)
        df["gap_oc"] = df[o] / prev_c - 1.0
        present.append("gap_oc")
    if "log_turn" not in df.columns and "turnover" in df.columns:
        df["log_turn"] = np.log1p(df["turnover"])
        present.append("log_turn")
    if "rv_20" not in df.columns and c in df.columns:
        lr_tmp = g[c].apply(lambda s: np.log(s).diff())
        df["rv_20"] = g[lr_tmp.name].apply(lambda s: np.sqrt((s**2).rolling(20, min_periods=10).sum()))
        present.append("rv_20")

    # ——（可选）盘口微观结构：若存在一档盘口则生成价差/深度/OFI ——
    if all(x in df.columns for x in ["bid1","ask1","bid1_size","ask1_size"]):
        b,a,bs,as_ = "bid1","ask1","bid1_size","ask1_size"
        mid = (df[a] + df[b]) / 2.0
        df["spread_rel"] = (df[a]-df[b]) / ((df[a]+df[b])/2.0)
        df["depth_imb"]  = (df[bs]-df[as_]) / (df[bs]+df[as_]).replace(0, np.nan)
        df["ofi"] = g.apply(
            lambda x: (np.sign(mid.loc[x.index].diff()).fillna(0) *
                       (x[bs].add(x[as_]).diff().fillna(0)) +
                       (x[bs].diff().fillna(0) - x[as_].diff().fillna(0)))
        ).reset_index(level=0, drop=True)
        present += ["spread_rel","depth_imb","ofi"]

    # ——（可选）情绪替代特征：存在即用 ——
    if "sentiment_score" in df.columns:
        df["sent_mean_20"] = g["sentiment_score"].apply(lambda s: s.rolling(20, min_periods=10).mean())
        df["sent_std_20"]  = g["sentiment_score"].apply(lambda s: s.rolling(20, min_periods=10).std())
        df["sent_x_turn"]  = df["sentiment_score"] * df.get("log_turn", 0)
        present += ["sent_mean_20","sent_std_20","sent_x_turn"]

    # —— 标签（Return Prediction & Risk Forecasting） ——
    fut_c = g[c].shift(-1)
    df["y_ret_k"] = (fut_c / df[c]) - 1.0
    df["y_up_k"]  = (df["y_ret_k"] > 0).astype(int)
    lr = g[c].apply(lambda s: np.log(s).diff())
    df["_r2"] = lr.pow(2)
    df["y_vol_k"] = g["_r2"].apply(lambda s: np.sqrt(s.shift(0).rolling(20, min_periods=10).sum()))

    # —— 统一滞后（_L1）——
    for p in present:
        df[f"{p}_L1"] = g[p].shift(1)

    feat_cols = [f"{p}_L1" for p in present]

    # —— winsorize（1%/99%）按资产 ——
    for col in feat_cols:
        q1  = g[col].transform(lambda s: s.quantile(0.01))
        q99 = g[col].transform(lambda s: s.quantile(0.99))
        df[col] = df[col].clip(q1, q99)

    # —— 滚动Z（z60 / z120）按资产 ——（批量 concat，避免碎片化）
    for w in z_windows:
        zdict = {}
        for col in feat_cols:
            m  = g[col].transform(lambda s, ww=w: s.rolling(ww, min_periods=max(5, ww//5)).mean())
            sd = g[col].transform(lambda s, ww=w: s.rolling(ww, min_periods=max(5, ww//5)).std())
            zname = f"{col}_z{w}"
            zdict[zname] = (df[col] - m) / sd.replace(0, np.nan)
        df = pd.concat([df, pd.DataFrame(zdict, index=df.index)], axis=1)
    df = df.copy()  # defragment

    # —— 切片：X_alpha / X_regime / X_risk / factor_pool ——
    alpha_feats = [c for c in df.columns if any(c.endswith(f"_z{w}") for w in z_windows)]
    regime_core = [x for x in ["rv_20_L1_z60","spread_rel_L1_z60","log_turn_L1_z60","ofi_L1_z60","adx_L1_z60"] if x in df.columns]
    risk_candidates = [x for x in alpha_feats if any(k in x for k in ["rv_","atr_","range_hl","log_turn","spread_rel","depth_imb","ofi","adx"])]

    base_cols = [id_col, ts_col]
    X_alpha     = df[base_cols + alpha_feats].dropna()
    X_regime    = df[base_cols + regime_core].dropna() if regime_core else df[base_cols].iloc[0:0]
    X_risk      = df[base_cols + risk_candidates].dropna()
    factor_pool = X_alpha.copy()
    y_df        = df[base_cols + ["y_ret_k","y_up_k","y_vol_k"]].dropna()

    # —— 导出：Alpha 特征 + 标签 ——
    os.makedirs(out_dir, exist_ok=True)
    csv_df  = X_alpha.merge(y_df, on=base_cols, how="inner")
    out_path = str(Path(out_dir) / out_name)
    csv_df.to_csv(out_path, index=False)

    return out_path, {
        "X_alpha":  X_alpha.reset_index(drop=True),
        "X_regime": X_regime.reset_index(drop=True),
        "X_risk":   X_risk.reset_index(drop=True),
        "factor_pool": factor_pool.reset_index(drop=True),
        "y_ret_k":  y_df[["y_ret_k","y_up_k"]].reset_index(drop=True),
        "y_vol_k":  y_df[["y_vol_k"]].reset_index(drop=True),
    }

# ===============================
# 1) Return Prediction: Gradient Boosting
# ===============================
from sklearn.ensemble import GradientBoostingRegressor

def train_gbdt_alpha(X_alpha: pd.DataFrame, y_ret: pd.DataFrame,
                     id_col="ticker", ts_col="trade_date",
                     out_scores="./outputs/gbdt_scores.csv") -> pd.DataFrame:
    """GBDT 直接做短期收益预测（与提案一致）。输出 OOF 预测 alpha_score_gbdt。"""
    base_cols = [id_col, ts_col]
    df = X_alpha.merge(y_ret[base_cols+["y_ret_k"]], on=base_cols, how="inner").dropna(subset=["y_ret_k"]).sort_values([ts_col, id_col])
    feats = [c for c in X_alpha.columns if c not in base_cols]
    X = df[feats].values
    y = df["y_ret_k"].values

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros(len(df))
    model = GradientBoostingRegressor(random_state=42)
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])

    out = df[base_cols].copy()
    out["alpha_score_gbdt"] = oof
    Path(out_scores).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_scores, index=False)
    print(f"[GBDT] scores -> {Path(out_scores).resolve()}")
    return out

# ===============================
# 2) Market Regime Classification: KMeans
# ===============================
def fit_kmeans_regime(X_regime: pd.DataFrame,
                      feature_cols=None,
                      n_clusters=3,
                      random_state=42):
    """
    用 z60 的波动/流动性/价差等特征做KMeans，并按簇心的风险维度排序映射 low/mid/high。
    """
    if X_regime.empty:
        raise ValueError("X_regime is empty; ensure regime features exist (e.g., rv_20_L1_z60).")
    from sklearn.cluster import KMeans

    base_cols = ["ticker","trade_date"]
    if feature_cols is None:
        feature_cols = [c for c in X_regime.columns if c not in base_cols]
    X = X_regime[feature_cols].values

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(X)

    centers = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    score = pd.Series(0.0, index=centers.index)
    if "rv_20_L1_z60" in centers:      score += centers["rv_20_L1_z60"]
    if "spread_rel_L1_z60" in centers: score += centers["spread_rel_L1_z60"]
    if "log_turn_L1_z60" in centers:   score += (-centers["log_turn_L1_z60"])

    order = score.sort_values().index.tolist()  # 小=低风险 → 大=高风险
    risk_names = ["low","mid","high"]
    mapping = { int(k): risk_names[i] for i,k in enumerate(order) }

    labels = km.predict(X)
    out = X_regime[base_cols].copy()
    out["regime_id"] = labels
    out["regime_risk_label"] = out["regime_id"].map(mapping)
    return km, mapping, out

# ===============================
# 3) 因素选择：LASSO（含空集回退）
# ===============================
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer

def lasso_select(factor_pool: pd.DataFrame, y_ret: pd.DataFrame,
                 id_col="ticker", ts_col="trade_date",
                 alphas=None, cv=5, random_state=42,
                 out_path="./outputs/lasso_selected_features.txt") -> List[str]:
    base_cols = [id_col, ts_col]
    df = factor_pool.merge(y_ret[base_cols+["y_ret_k"]], on=base_cols, how="inner").dropna(subset=["y_ret_k"]).sort_values([ts_col, id_col])
    X = df.drop(columns=base_cols+["y_ret_k"]).values
    y = df["y_ret_k"].values
    X = SimpleImputer(strategy="median").fit_transform(X)

    # —— 这里已按你的要求修改：时序CV + 收敛参数 ——（keywords: lasso_select）
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv)
    model = LassoCV(
        alphas=alphas if alphas is not None else np.logspace(-5, 0, 30),
        cv=tscv,
        random_state=random_state,
        max_iter=20000,
        tol=1e-5,
        n_jobs=-1
    )
    model.fit(X, y)

    cols = [c for c in df.columns if c not in base_cols+["y_ret_k"]]
    selected = [c for c, w in zip(cols, model.coef_) if abs(w) > 1e-12]

    # 回退：如全0，则用 |corr| Top-K
    if len(selected) == 0:
        X_df = pd.DataFrame(X, columns=cols)
        corr = X_df.apply(lambda s: np.corrcoef(s, y)[0,1] if np.std(s)>0 else 0.0)
        corr = corr.replace([np.nan, np.inf, -np.inf], 0.0).abs().sort_values(ascending=False)
        K = min(50, max(10, int(0.1 * len(cols))))
        selected = corr.head(K).index.tolist()
        print(f"[LASSO] no nonzero coef; fallback to top-{K} by |corr|.")

    # —— 这里已按你的要求新增：保存系数文件 ——（keywords: save_coef）
    coef_path = out_path
    coef_txt = Path(out_path).with_name("lasso_selected_with_coef.txt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(coef_path, "w", encoding="utf-8") as f:
        for c in selected:
            f.write(c + "\n")
    with open(coef_txt, "w", encoding="utf-8") as f:
        for c, w in zip(cols, model.coef_):
            if abs(w) > 1e-12:
                f.write(f"{c}\t{w:.8g}\n")

    print(f"[LASSO] selected {len(selected)} features -> {Path(coef_path).resolve()}")
    print(f"[LASSO] coef saved -> {Path(coef_txt).resolve()}")
    return selected

# ===============================
# 4) 信号集成：Stacking（RF+GBDT → Ridge）
# ===============================
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

def train_stack_alpha(X_alpha: pd.DataFrame, y_ret: pd.DataFrame,
                      id_col="ticker", ts_col="trade_date",
                      selected_features: List[str] = None,
                      n_splits=5,
                      out_scores="./outputs/model_scores.csv") -> pd.DataFrame:
    base_cols = [id_col, ts_col]
    df = X_alpha.merge(y_ret[base_cols+["y_ret_k"]], on=base_cols, how="inner").dropna(subset=["y_ret_k"]).sort_values([ts_col, id_col])

    all_feats = [c for c in X_alpha.columns if c not in base_cols]
    feats = all_feats if not selected_features else [c for c in all_feats if c in selected_features]
    X = df[feats].values
    y = df["y_ret_k"].values

    base_estimators = [
        ("rf",   RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)),
        ("gbrt", GradientBoostingRegressor(random_state=42))
    ]
    meta = RidgeCV(alphas=np.logspace(-3, 3, 13))
    stack = StackingRegressor(estimators=base_estimators, final_estimator=meta, passthrough=False, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(df))
    for tr, te in tscv.split(X):
        stack.fit(X[tr], y[tr])
        oof[te] = stack.predict(X[te])

    out = df[base_cols].copy()
    out["alpha_score"] = oof
    Path(out_scores).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_scores, index=False)
    print(f"[STACK] model scores -> {Path(out_scores).resolve()}")
    return out

# ===============================
# 5) 风险预测：波动率回归
# ===============================
def train_risk_model(X_risk: pd.DataFrame, y_vol: pd.DataFrame,
                     id_col="ticker", ts_col="trade_date",
                     n_splits=5,
                     out_path="./outputs/risk_forecast.csv") -> pd.DataFrame:
    base_cols = [id_col, ts_col]
    df = X_risk.merge(y_vol[base_cols+["y_vol_k"]], on=base_cols, how="inner").dropna(subset=["y_vol_k"]).sort_values([ts_col, id_col])

    feats = [c for c in X_risk.columns if c not in base_cols]
    X = df[feats].values
    y = df["y_vol_k"].values

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros(len(df))
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])

    out = df[base_cols].copy()
    out["vol_pred"] = oof
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[RISK] forecast -> {Path(out_path).resolve()}")
    return out

# ===============================
# 主程序：读取文件 -> 特征工程 -> KMeans -> LASSO -> GBDT/Stacking -> 风险回归
# ===============================
if __name__ == "__main__":
    in_path = "internal_derived_dataset_clean.csv"  # 你指定的输入文件
    out_csv, res = feature_engineering_from_file(in_path,
                                                 out_dir="./outputs",
                                                 out_name="dataquant_features.csv",
                                                 ts_col="trade_date",
                                                 id_col="ticker")

    base_csv = Path(out_csv).resolve()
    print(f"Feature CSV saved to: {base_csv}")

    # —— Regime：KMeans + 导出 with_regime ——
    X_regime = res["X_regime"]
    if not X_regime.empty:
        km, mapping, regime_df = fit_kmeans_regime(X_regime)
        df_alpha = pd.read_csv(base_csv, parse_dates=["trade_date"])
        df_alpha["trade_date"] = pd.to_datetime(df_alpha["trade_date"], errors="coerce").dt.normalize()
        regime_df["trade_date"] = pd.to_datetime(regime_df["trade_date"], errors="coerce").dt.normalize()
        merged = df_alpha.merge(regime_df, on=["ticker","trade_date"], how="left")
        out_with_regime = base_csv.with_name(base_csv.stem + "_with_regime.csv")
        merged.to_csv(out_with_regime, index=False)
        print(f"Regime CSV saved to: {out_with_regime}")
        print(f"Regime mapping (cluster -> risk): {mapping}")
    else:
        print("Skip regime export: X_regime is empty.")

    # —— 读取落盘 CSV 构造带主键的标签表（避免 KeyError）——
    train_df = pd.read_csv(base_csv, parse_dates=["trade_date"])
    train_df["trade_date"] = pd.to_datetime(train_df["trade_date"], errors="coerce").dt.normalize()
    y_ret_df = train_df[["ticker","trade_date","y_ret_k"]].dropna()
    y_vol_df = train_df[["ticker","trade_date","y_vol_k"]].dropna()

    # —— LASSO 选因 ——
    selected = lasso_select(res["factor_pool"], y_ret_df,
                            out_path="./outputs/lasso_selected_features.txt")

    # —— Return Prediction：GBDT ——
    _gbdt_scores = train_gbdt_alpha(res["X_alpha"], y_ret_df,
                                    out_scores="./outputs/gbdt_scores.csv")

    # —— Signal Integration：Stacking ——
    _stack_scores = train_stack_alpha(res["X_alpha"], y_ret_df,
                                      selected_features=selected,
                                      out_scores="./outputs/model_scores.csv")

    # —— Risk Forecasting：短期波动回归 ——
    _risk = train_risk_model(res["X_risk"], y_vol_df,
                             out_path="./outputs/risk_forecast.csv")
