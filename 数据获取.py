# -*- coding: utf-8 -*-
"""
生成 internal_derived_dataset.csv：
- 找到“上个交易日”
- 用 akshare 实时行情挑“最火”的候选池（涨跌幅/成交额/换手率/量比综合）
- 逐个拉取历史（日线后复权），若“两年窗口交易日不足”则跳过并顺延下一个票
- 对最终选出的 3 只票，输出“最近两年（截至上个交易日）所有交易日”的特征行
"""

import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import akshare as ak
except Exception as e:
    raise RuntimeError("需要先安装 akshare：pip install akshare") from e


OUTPUT_PATH = "internal_derived_dataset.csv"

NEED_K = 10                # 需要最终凑够的票数
CANDIDATE_TOPN = 200        # 初始候选池大小（不够会自动扩大/顺延）
SLEEP_SEC = 0.25

# 两年窗口：日历 730 天；交易日至少要多少（约 252*2=504，留点余量）
TWO_YEAR_CAL_DAYS = 730
MIN_BARS_2Y = 480

# 拉历史：回溯多少自然日（给两年窗口留缓冲，处理停牌/节假日）
LOOKBACK_DAYS = 900

CURRENCY_CODE = "CNY"
CURRENCY_NAME = "人民币"


# =========================
# 工具：交易日
# =========================
def get_prev_trade_date(today=None):
    today = pd.Timestamp(today or datetime.now().date())
    # 优先：akshare 交易日历
    try:
        cal = ak.tool_trade_date_hist_sina()
        if cal is not None and not cal.empty:
            # 常见列名：trade_date / 交易日期 / date
            for c in ["trade_date", "交易日期", "date", "日期"]:
                if c in cal.columns:
                    ds = pd.to_datetime(cal[c], errors="coerce")
                    ds = ds.dropna().sort_values().reset_index(drop=True)
                    ds = ds[ds < today]
                    if len(ds) > 0:
                        return pd.Timestamp(ds.iloc[-1]).normalize()
    except Exception:
        pass

    # 兜底：用工作日回退（不严格，但总比没有好）
    d = today - pd.Timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d -= pd.Timedelta(days=1)
    return pd.Timestamp(d).normalize()


def guess_ticker(code6: str) -> str:
    c = str(code6).strip()
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return f"{c}.SZ"


# =========================
# 工具：候选（多方法）
# =========================
def fetch_hot_candidates(top_n=200, max_retry=3, sleep_sec=1.2):
    """
    用实时行情近似“上个交易日最火”（你收盘后跑，一般就是上一交易日的收盘数据）。
    综合排序：涨跌幅、成交额、换手率、量比
    """
    df = None
    src = None

    for i in range(1, max_retry + 1):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                src = "stock_zh_a_spot_em"
                break
        except Exception:
            time.sleep(sleep_sec * i)

    if df is None or df.empty:
        for i in range(1, max_retry + 1):
            try:
                df = ak.stock_zh_a_spot()
                if df is not None and not df.empty:
                    src = "stock_zh_a_spot"
                    break
            except Exception:
                time.sleep(sleep_sec * i)

    if df is None or df.empty:
        raise RuntimeError("获取行情失败：spot_em / spot 都不可用")

    # 标准化字段
    rename_map = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "pct_chg",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "量比": "volume_ratio",
    }
    for k, v in list(rename_map.items()):
        if k not in df.columns:
            rename_map.pop(k, None)

    df = df.rename(columns=rename_map)

    # 兜底列
    if "amount" not in df.columns:
        # 有的源叫成交额/成交额(元)/成交额（万）等，这里宁可缺失也别乱猜
        df["amount"] = np.nan
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = np.nan
    if "volume_ratio" not in df.columns:
        df["volume_ratio"] = np.nan
    if "pct_chg" not in df.columns:
        df["pct_chg"] = np.nan

    df["code"] = df["code"].astype(str)
    df["name"] = df["name"].astype(str)

    for c in ["pct_chg", "amount", "turnover_rate", "volume_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["code", "name"]).copy()

    # 综合排序（缺失按很小值处理，避免 NaN 排前面）
    df["pct_chg_f"] = df["pct_chg"].fillna(-1e9)
    df["amount_f"] = df["amount"].fillna(-1e9)
    df["turnover_rate_f"] = df["turnover_rate"].fillna(-1e9)
    df["volume_ratio_f"] = df["volume_ratio"].fillna(-1e9)

    df = df.sort_values(
        by=["pct_chg_f", "amount_f", "turnover_rate_f", "volume_ratio_f"],
        ascending=False
    ).reset_index(drop=True)

    out = df[["code", "name"]].head(int(top_n)).copy()
    out["ticker"] = out["code"].apply(guess_ticker)

    print(f"[INFO] 候选池来源：{src}，候选数={len(out)}")
    return out


# =========================
# 工具：拉历史（多方法）
# =========================
def fetch_history_daily_qfq(code6: str, days=900, max_retry=4, sleep_sec=0.8):
    """
    拉后复权日线：
    1) stock_zh_a_daily（sina）
    2) stock_zh_a_hist（eastmoney）
    返回列：date, open, high, low, close, volume, amount
    """
    raw = str(code6).strip()
    ticker = guess_ticker(raw)
    if ticker.endswith(".SH"):
        symbol_daily = "sh" + raw
    elif ticker.endswith(".BJ"):
        symbol_daily = "bj" + raw
    else:
        symbol_daily = "sz" + raw

    end = datetime.now()
    start = end - timedelta(days=int(days))

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    # --- 1) sina daily ---
    for i in range(1, max_retry + 1):
        try:
            df = ak.stock_zh_a_daily(symbol=symbol_daily, start_date=start_str, end_date=end_str, adjust="qfq")
            if df is not None and not df.empty:
                # 兼容中英文列
                if "date" not in df.columns and "日期" in df.columns:
                    df = df.rename(columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "成交额": "amount",
                    })
                # 有时 amount 不返回，补 NaN
                if "amount" not in df.columns:
                    df["amount"] = np.nan

                need = ["date", "open", "high", "low", "close", "volume", "amount"]
                if all(c in df.columns for c in need):
                    df = df[need].copy()
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
                    df = df.sort_values("date").reset_index(drop=True)
                    df = df[df["volume"].astype(float) > 0]
                    return df
        except Exception:
            time.sleep(sleep_sec * i)

    # --- 2) em hist fallback ---
    for i in range(1, max_retry + 1):
        try:
            df = ak.stock_zh_a_hist(symbol=raw, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            if df is None or df.empty:
                break

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            })

            if "amount" not in df.columns:
                df["amount"] = np.nan

            need = ["date", "open", "high", "low", "close", "volume", "amount"]
            if all(c in df.columns for c in need):
                df = df[need].copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
                df = df.sort_values("date").reset_index(drop=True)
                df = df[df["volume"].astype(float) > 0]
                return df
        except Exception:
            time.sleep(sleep_sec * i)

    return None


# =========================
# 指标计算（不依赖 TA-Lib）
# =========================
def _ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, n=9, k=3, d=3):
    low_n = low.rolling(n, min_periods=max(3, n // 2)).min()
    high_n = high.rolling(n, min_periods=max(3, n // 2)).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0.0, np.nan) * 100.0
    k_val = rsv.ewm(alpha=1.0 / k, adjust=False).mean()
    d_val = k_val.ewm(alpha=1.0 / d, adjust=False).mean()
    return k_val, d_val


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = _atr(high, low, close, period=period)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0.0, np.nan))

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return plus_di, minus_di, adx, atr


def _bbands(close: pd.Series, period=20, nb=2.0):
    mid = close.rolling(period, min_periods=max(5, period // 2)).mean()
    sd = close.rolling(period, min_periods=max(5, period // 2)).std(ddof=1)
    up = mid + nb * sd
    dn = mid - nb * sd
    bw = (up - dn) / mid.replace(0.0, np.nan)
    pb = (close - dn) / (up - dn).replace(0.0, np.nan)
    return mid, up, dn, bw, pb


def _obv(close: pd.Series, volume: pd.Series):
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _vpt(close: pd.Series, volume: pd.Series):
    ret = close.pct_change().fillna(0.0)
    return (volume * ret).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period=14):
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    delta_tp = tp.diff()

    pos_mf = rmf.where(delta_tp > 0, 0.0)
    neg_mf = rmf.where(delta_tp < 0, 0.0).abs()

    pos_sum = pos_mf.rolling(period, min_periods=max(5, period // 2)).sum()
    neg_sum = neg_mf.rolling(period, min_periods=max(5, period // 2)).sum()

    mfr = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + mfr)
    return mfi


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period=20):
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=max(5, period // 2)).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=max(5, period // 2)).mean()
    cci = (tp - sma) / (0.015 * mad.replace(0.0, np.nan))
    return cci


def compute_features(df_hist: pd.DataFrame):
    """
    输入：date/open/high/low/close/volume/amount
    输出：trade_date + 你截图里那套字段（缺失会补 NaN）
    """
    x = df_hist.copy()
    x = x.rename(columns={"date": "trade_date", "amount": "turnover"})
    x["trade_date"] = pd.to_datetime(x["trade_date"]).dt.normalize()

    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(subset=["trade_date", "open", "high", "low", "close", "volume"]).copy()
    x = x.sort_values("trade_date").reset_index(drop=True)

    # returns / vol
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_5"] = x["close"].pct_change(5)
    x["ret_20"] = x["close"].pct_change(20)
    x["vol_10"] = x["ret_1"].rolling(10, min_periods=5).std(ddof=1)
    x["vol_20"] = x["ret_1"].rolling(20, min_periods=10).std(ddof=1)

    # MA/EMA
    x["sma_5"] = x["close"].rolling(5, min_periods=3).mean()
    x["sma_10"] = x["close"].rolling(10, min_periods=5).mean()
    x["sma_20"] = x["close"].rolling(20, min_periods=10).mean()
    x["ema_12"] = _ema(x["close"], 12)
    x["ema_26"] = _ema(x["close"], 26)
    x["ma_gap_5_20"] = x["sma_5"] / x["sma_20"].replace(0.0, np.nan) - 1.0
    x["ema_gap_12_26"] = x["ema_12"] / x["ema_26"].replace(0.0, np.nan) - 1.0

    # MACD
    x["macd"] = x["ema_12"] - x["ema_26"]
    x["macd_signal"] = _ema(x["macd"], 9)
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    # RSI
    x["rsi"] = _rsi(x["close"], 14)

    # KDJ
    k, d = _kdj(x["high"], x["low"], x["close"], 9, 3, 3)
    x["kdj_k"] = k
    x["kdj_d"] = d

    # ATR + DI/ADX
    pdi, mdi, adx, atr = _adx(x["high"], x["low"], x["close"], 14)
    x["atr"] = atr
    x["pdi"] = pdi
    x["mdi"] = mdi
    x["adx"] = adx

    # BBANDS
    bb_mid, bb_up, bb_dn, bb_bw, bb_pb = _bbands(x["close"], 20, 2.0)
    x["bb_mid"] = bb_mid
    x["bb_up"] = bb_up
    x["bb_dn"] = bb_dn
    x["bb_bw"] = bb_bw
    x["bb_pb"] = bb_pb

    # OBV / VPT
    x["obv"] = _obv(x["close"], x["volume"])
    x["vpt"] = _vpt(x["close"], x["volume"])

    # MFI / CCI
    x["mfi"] = _mfi(x["high"], x["low"], x["close"], x["volume"], 14)
    x["cci"] = _cci(x["high"], x["low"], x["close"], 20)

    # VWAP window（5日）
    # 日 vwap 近似：turnover/volume（注意 turnover 单位可能不同源不一致，但形态特征仍可用）
    vwap_day = x["turnover"] / x["volume"].replace(0.0, np.nan)
    x["vwap_win"] = (
        x["turnover"].rolling(5, min_periods=3).sum()
        / x["volume"].rolling(5, min_periods=3).sum().replace(0.0, np.nan)
    )
    x["vwap_win"] = x["vwap_win"].fillna(vwap_day)

    # zscore
    m20 = x["close"].rolling(20, min_periods=10).mean()
    s20 = x["close"].rolling(20, min_periods=10).std(ddof=1).replace(0.0, np.nan)
    x["z_close_20"] = (x["close"] - m20) / s20

    mt20 = x["turnover"].rolling(20, min_periods=10).mean()
    st20 = x["turnover"].rolling(20, min_periods=10).std(ddof=1).replace(0.0, np.nan)
    x["z_turnover_20"] = (x["turnover"] - mt20) / st20

    return x


# =========================
# 主流程
# =========================
def main():
    prev_td = get_prev_trade_date()
    print(f"[INFO] 上个交易日：{prev_td.date()}")

    cand = fetch_hot_candidates(top_n=CANDIDATE_TOPN)
    if cand is None or cand.empty:
        raise RuntimeError("候选池为空")

    selected_segments = []
    picked = 0

    i = 0
    while picked < NEED_K and i < len(cand):
        code = str(cand.loc[i, "code"])
        name = str(cand.loc[i, "name"])
        ticker = str(cand.loc[i, "ticker"])
        i += 1

        print(f"[{picked+1}/{NEED_K}] 拉取历史并计算特征：{ticker} {name}")
        t0 = time.time()

        hist = fetch_history_daily_qfq(code6=code, days=LOOKBACK_DAYS)
        if hist is None or hist.empty:
            print(f"  [WARN] 拉历史失败，跳过：{ticker}")
            continue

        feat = compute_features(hist)
        if feat.empty:
            print(f"  [WARN] 特征计算为空，跳过：{ticker}")
            continue

        # 必须包含上个交易日（否则停牌/缺K线）
        if feat.loc[feat["trade_date"] == prev_td].empty:
            print(f"  [WARN] 缺少上个交易日K线，跳过：{ticker}")
            continue

        # 取最近两年（截至上个交易日）
        start_td = prev_td - pd.Timedelta(days=TWO_YEAR_CAL_DAYS)
        seg = feat.loc[(feat["trade_date"] >= start_td) & (feat["trade_date"] <= prev_td)].copy()
        seg = seg.sort_values("trade_date").reset_index(drop=True)

        # 两年窗口交易日不足 -> 跳过并顺延下一个票
        if len(seg) < MIN_BARS_2Y:
            print(f"  [WARN] 两年窗口交易日不足，跳过：{ticker} (bars_in_2y={len(seg)})")
            continue

        # 填充元信息列
        seg["trade_date"] = pd.to_datetime(seg["trade_date"]).dt.strftime("%Y/%m/%d")
        seg["ticker"] = ticker
        seg["stock_name"] = name
        seg["currency_code"] = CURRENCY_CODE
        seg["currency_name"] = CURRENCY_NAME
        seg["anon_id"] = f"A{picked+1:04d}"
        seg["alpha_signal"] = np.nan
        seg["risk_score"] = np.nan

        out_cols = [
            "trade_date", "ticker",
            "open", "high", "low", "close",
            "volume", "turnover",
            "ret_1", "ret_5", "ret_20",
            "vol_10", "vol_20",
            "sma_5", "sma_10", "sma_20",
            "ema_12", "ema_26",
            "ma_gap_5_20", "ema_gap_12_26",
            "macd", "macd_signal", "macd_hist",
            "rsi", "kdj_k", "kdj_d",
            "atr", "pdi", "mdi", "adx",
            "bb_mid", "bb_up", "bb_dn", "bb_bw", "bb_pb",
            "obv", "vpt", "mfi", "cci",
            "vwap_win", "z_close_20", "z_turnover_20",
            "stock_name", "currency_code", "currency_name",
            "anon_id", "alpha_signal", "risk_score"
        ]

        for c in out_cols:
            if c not in seg.columns:
                seg[c] = np.nan

        selected_segments.append(seg[out_cols])
        picked += 1

        print(f"  [OK] 用时 {time.time() - t0:.2f}s，两年窗口行数={len(seg)}，已收集 {picked}/{NEED_K}")
        time.sleep(SLEEP_SEC)

    if picked < NEED_K:
        print(f"[WARN] 候选池跑完仍未凑够 {NEED_K} 只票，仅得到 {picked} 只。")

    if not selected_segments:
        raise RuntimeError("最终没有任何股票满足“两年历史数据”要求")

    out_df = pd.concat(selected_segments, axis=0, ignore_index=True)
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] 已生成：{OUTPUT_PATH}，总行数={len(out_df)}，股票数={picked}")


if __name__ == "__main__":
    main()
