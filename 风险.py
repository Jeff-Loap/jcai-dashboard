import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2, t

def fit_garch11(r, dist="normal"):
    r = np.asarray(r, dtype=float)
    r = r - np.nanmean(r)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 200:
        raise ValueError("样本太少，建议至少 200+ 条收益率")

    # 用 sigmoid/exp 做约束：omega>0, alpha,beta in (0,1) 且 alpha+beta<1
    # 若 dist='t'：nu=2+exp(x) 保证 nu>2
    def unpack(theta):
        omega = np.exp(theta[0])
        alpha = 1 / (1 + np.exp(-theta[1]))
        beta  = 1 / (1 + np.exp(-theta[2]))
        s = alpha + beta
        if s >= 0.999:
            alpha = alpha / s * 0.999
            beta  = beta  / s * 0.999

        if dist == "t":
            nu = 2.0 + np.exp(theta[3])
            return omega, alpha, beta, nu
        return omega, alpha, beta, None

    def neg_loglik(theta):
        omega, alpha, beta, nu = unpack(theta)

        h = np.empty(n)
        h[0] = np.var(r) + 1e-12
        for i in range(1, n):
            h[i] = omega + alpha * (r[i-1] ** 2) + beta * h[i-1]
            if h[i] <= 0:
                return 1e18

        if dist == "t":
            # z = r/sqrt(h)，其方差=1；令 u 服从标准t(df=nu)，需要做尺度变换
            z = r / np.sqrt(h)
            scale = np.sqrt(nu / (nu - 2.0))   # 标准t的方差=nu/(nu-2)
            u = z * scale                      # u ~ t(df=nu)
            ll = np.sum(t.logpdf(u, df=nu) + np.log(scale) - 0.5 * np.log(h))
            return -float(ll)

        ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(h) + (r**2)/h)
        return -float(ll)

    # 初值：alpha=0.05, beta=0.9；t 分布 nu 初值=8
    if dist == "t":
        theta0 = np.array([
            np.log(0.1*np.var(r) + 1e-12),
            np.log(0.05/0.95),
            np.log(0.9/0.1),
            np.log(8.0 - 2.0)
        ])
    else:
        theta0 = np.array([
            np.log(0.1*np.var(r) + 1e-12),
            np.log(0.05/0.95),
            np.log(0.9/0.1)
        ])

    res = minimize(neg_loglik, theta0, method="L-BFGS-B")
    omega, alpha, beta, nu = unpack(res.x)

    h = np.empty(n)
    h[0] = np.var(r) + 1e-12
    for i in range(1, n):
        h[i] = omega + alpha * (r[i-1] ** 2) + beta * h[i-1]

    out = {"omega": omega, "alpha": alpha, "beta": beta, "h": h, "r": r, "opt": res}
    if dist == "t":
        out["nu"] = float(nu)
    return out

def forecast_1step(garch_fit):
    r = garch_fit["r"]
    h = garch_fit["h"]
    omega, alpha, beta = garch_fit["omega"], garch_fit["alpha"], garch_fit["beta"]
    h_next = omega + alpha * (r[-1] ** 2) + beta * h[-1]
    return float(np.sqrt(h_next))

def var_from_sigma(sigma, alpha=0.01, mu=0.0, dist="normal", nu=None):
    # 返回“收益率阈值”，若收益率 < VaR 则视为穿透（更差）
    if dist == "t":
        if nu is None:
            raise ValueError("dist='t' 时必须提供 nu")
        scale = np.sqrt(nu / (nu - 2.0))
        return float(mu + sigma * (t.ppf(alpha, df=nu) / scale))
    return float(mu + sigma * norm.ppf(alpha))

def safe_rate(num, den, eps=1e-12):
    if den <= 0:
        return 0.0
    v = num / den
    return float(min(max(v, eps), 1 - eps))

def backtest_garch_var(
    r,
    alpha_level=0.01,
    initial_train=400,
    refit_every=20,
    dist="t",
):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < initial_train + 50:
        raise ValueError(f"样本太少：n={n}，至少需要 initial_train+50")

    pred_sigma = np.full(n, np.nan, dtype=float)
    pred_var = np.full(n, np.nan, dtype=float)

    fit = None
    for i in range(initial_train, n):
        # 用 r[:i] 拟合，预测第 i 天（1-step ahead）
        if (i == initial_train) or ((i - initial_train) % refit_every == 0):
            fit = fit_garch11(r[:i], dist=dist)

        sigma_i = forecast_1step(fit)
        pred_sigma[i] = sigma_i

        if dist == "t":
            pred_var[i] = var_from_sigma(
                sigma_i, alpha=alpha_level, mu=0.0, dist="t", nu=fit["nu"]
            )
        else:
            pred_var[i] = var_from_sigma(
                sigma_i, alpha=alpha_level, mu=0.0, dist="normal"
            )

    mask = np.isfinite(pred_sigma)
    r_oos = r[mask]
    sigma_oos = pred_sigma[mask]
    h_oos = sigma_oos ** 2
    var_oos = pred_var[mask]

    # ---- 波动率预测误差（|r| proxy）----
    abs_proxy = np.abs(r_oos)
    sq_proxy = r_oos ** 2
    mae = float(np.mean(np.abs(abs_proxy - sigma_oos)))
    rmse = float(np.sqrt(np.mean((abs_proxy - sigma_oos) ** 2)))
    corr = float(np.corrcoef(abs_proxy, sigma_oos)[0, 1]) if len(r_oos) > 2 else float("nan")

    # QLIKE（用 r^2 proxy）
    eps = 1e-12
    qlike = float(np.mean(sq_proxy / (h_oos + eps) + np.log(h_oos + eps)))

    # ---- VaR 回测 ----
    breach = (r_oos < var_oos).astype(int)
    T = int(len(breach))
    x = int(breach.sum())
    breach_rate = x / T

    # Kupiec POF（无条件覆盖率）
    p = alpha_level
    phat = float(min(max(breach_rate, eps), 1 - eps))
    logL0 = (T - x) * np.log(1 - p) + x * np.log(p)
    logL1 = (T - x) * np.log(1 - phat) + x * np.log(phat)
    LR_uc = float(-2 * (logL0 - logL1))
    p_uc = float(1 - chi2.cdf(LR_uc, df=1))

    # Christoffersen 独立性 + 条件覆盖率
    if T >= 2:
        b0 = breach[:-1]
        b1 = breach[1:]
        n00 = int(np.sum((b0 == 0) & (b1 == 0)))
        n01 = int(np.sum((b0 == 0) & (b1 == 1)))
        n10 = int(np.sum((b0 == 1) & (b1 == 0)))
        n11 = int(np.sum((b0 == 1) & (b1 == 1)))

        pi01 = safe_rate(n01, n00 + n01, eps=eps)
        pi11 = safe_rate(n11, n10 + n11, eps=eps)
        pi1  = safe_rate(n01 + n11, n00 + n01 + n10 + n11, eps=eps)

        logL_ind0 = (n00 + n10) * np.log(1 - pi1) + (n01 + n11) * np.log(pi1)
        logL_ind1 = (
            n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
            n10 * np.log(1 - pi11) + n11 * np.log(pi11)
        )
        LR_ind = float(-2 * (logL_ind0 - logL_ind1))
        p_ind = float(1 - chi2.cdf(LR_ind, df=1))

        LR_cc = float(LR_uc + LR_ind)
        p_cc = float(1 - chi2.cdf(LR_cc, df=2))
    else:
        n00 = n01 = n10 = n11 = 0
        LR_ind = float("nan")
        p_ind = float("nan")
        LR_cc = float("nan")
        p_cc = float("nan")

    last_params = {k: fit[k] for k in ["omega", "alpha", "beta"]}
    if dist == "t":
        last_params["nu"] = fit["nu"]

    return {
        "last_fit_params": last_params,
        "oos_days": T,
        "vol_mae": mae,
        "vol_rmse": rmse,
        "vol_qlike": qlike,
        "vol_corr_absret_sigma": corr,
        "var_alpha": alpha_level,
        "var_breaches": x,
        "var_breach_rate": breach_rate,
        "kupiec_LR_uc": LR_uc,
        "kupiec_p": p_uc,
        "christoffersen_n00": n00,
        "christoffersen_n01": n01,
        "christoffersen_n10": n10,
        "christoffersen_n11": n11,
        "christoffersen_LR_ind": LR_ind,
        "christoffersen_p_ind": p_ind,
        "christoffersen_LR_cc": LR_cc,
        "christoffersen_p_cc": p_cc,
        "pred_sigma": pred_sigma,
        "pred_var": pred_var,
        "r_clean": r,
        "mask_oos": mask,
        "dist": dist,
    }

if __name__ == "__main__":
    # ===== 读数据 =====
    df = pd.read_csv(r"D:\PythonFile\JCAI\data_ak\internal_derived_dataset_clean.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    one = df[df["ticker"] == "000001.SZ"].sort_values("trade_date").copy()
    r = one["ret_1"].astype(float).values

    # ===== 回测参数（可改）=====
    alpha_level = 0.01
    initial_train = 400
    refit_every = 20
    dist = "t"  # 't' 或 'normal'

    # ===== 运行回测 =====
    out = backtest_garch_var(
        r,
        alpha_level=alpha_level,
        initial_train=initial_train,
        refit_every=refit_every,
        dist=dist,
    )

    print(f"OOS days={out['oos_days']}")
    print(
        f"VaR({int(alpha_level * 100)}%): breaches={out['var_breaches']}/{out['oos_days']}, "
        f"rate={out['var_breach_rate']:.4%}"
    )
    print(f"Kupiec: p={out['kupiec_p']:.4f}")
    print(f"Christoffersen: p_cc={out['christoffersen_p_cc']:.4f}")
