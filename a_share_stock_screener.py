from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "a_share_stock_screener"
CACHE_DIR = PROJECT_DIR / "work" / "a_share_stock_screener_cache"
STRONG_TIER = "10日短线强信号"
WATCH_TIER = "观察候选"


def request_json(url: str, retries: int = 3, timeout: int = 25) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"请求失败: {url}\n{last_error}") from last_error


def safe_num(value: object) -> float:
    try:
        if value in ("-", None, ""):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending).fillna(0.0)


def band_score(series: pd.Series, low: float, high: float, ideal: float) -> pd.Series:
    left = ((series - low) / max(ideal - low, 1e-9)).clip(0, 1)
    right = ((high - series) / max(high - ideal, 1e-9)).clip(0, 1)
    return np.minimum(left, right).fillna(0)


def market_url(page: int, page_size: int = 100) -> str:
    fields = "f12,f14,f2,f3,f6,f8,f9,f15,f16,f17,f18,f20,f21,f23,f100"
    params = {
        "pn": page,
        "pz": page_size,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f6",
        "fs": "m:0+t:6,m:1+t:2",
        "fields": fields,
    }
    return "https://push2.eastmoney.com/api/qt/clist/get?" + urllib.parse.urlencode(params)


def holder_latest_url() -> str:
    params = {
        "sortColumns": "END_DATE,SECURITY_CODE",
        "sortTypes": "-1,-1",
        "pageSize": 10000,
        "pageNumber": 1,
        "reportName": "RPT_HOLDERNUMLATEST",
        "columns": "ALL",
        "source": "WEB",
        "client": "WEB",
    }
    return "https://datacenter-web.eastmoney.com/api/data/v1/get?" + urllib.parse.urlencode(params)


def holder_detail_url(code: str) -> str:
    params = {
        "reportName": "RPT_HOLDERNUM_DET",
        "columns": "ALL",
        "filter": f'(SECURITY_CODE="{code}")',
        "pageSize": 20,
        "pageNumber": 1,
        "sortColumns": "END_DATE",
        "sortTypes": "-1",
        "source": "WEB",
        "client": "WEB",
    }
    return "https://datacenter-web.eastmoney.com/api/data/v1/get?" + urllib.parse.urlencode(params)


def kline_url(code: str, start: str, end: str) -> str:
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start,
        "end": end,
    }
    return "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)


def load_market(refresh: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "market_snapshot.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, dtype={"code": str})

    try:
        rows: list[dict] = []
        total = None
        page = 1
        while True:
            payload = request_json(market_url(page))
            data = payload.get("data", {}) or {}
            if total is None:
                total = int(data.get("total") or 0)
            diff = data.get("diff") or []
            if not diff:
                break
            rows.extend(diff)
            if len(rows) >= total:
                break
            page += 1
            time.sleep(0.03)
    except Exception:
        if cache.exists():
            print("行情刷新失败，使用本地缓存。")
            return pd.read_csv(cache, dtype={"code": str})
        raise

    records = []
    for item in rows:
        records.append(
            {
                "code": str(item.get("f12", "")).zfill(6),
                "name": item.get("f14", ""),
                "close": safe_num(item.get("f2")),
                "pct_chg": safe_num(item.get("f3")) / 100,
                "amount": safe_num(item.get("f6")),
                "turnover": safe_num(item.get("f8")) / 100,
                "pe": safe_num(item.get("f9")),
                "high": safe_num(item.get("f15")),
                "low": safe_num(item.get("f16")),
                "open": safe_num(item.get("f17")),
                "prev_close": safe_num(item.get("f18")),
                "total_mv": safe_num(item.get("f20")),
                "float_mv": safe_num(item.get("f21")),
                "pb": safe_num(item.get("f23")),
                "industry": item.get("f100") or "未知",
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def load_holder_latest(refresh: bool) -> pd.DataFrame:
    cache = CACHE_DIR / "holder_latest.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, dtype={"code": str}, parse_dates=["holder_date", "notice_date"])

    try:
        rows = request_json(holder_latest_url()).get("result", {}).get("data", [])
    except Exception:
        if cache.exists():
            print("股东户数刷新失败，使用本地缓存。")
            return pd.read_csv(cache, dtype={"code": str}, parse_dates=["holder_date", "notice_date"])
        raise
    records = []
    for item in rows:
        records.append(
            {
                "code": str(item.get("SECURITY_CODE", "")).zfill(6),
                "holder_num": safe_num(item.get("HOLDER_NUM")),
                "pre_holder_num": safe_num(item.get("PRE_HOLDER_NUM")),
                "holder_change_pct": safe_num(item.get("HOLDER_NUM_RATIO")) / 100,
                "holder_date": pd.to_datetime(item.get("END_DATE"), errors="coerce"),
                "notice_date": pd.to_datetime(item.get("HOLD_NOTICE_DATE"), errors="coerce"),
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def fetch_kline(code: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    cache = CACHE_DIR / f"kline_{code}_{start}_{end}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["date"])
    if not refresh:
        old_caches = sorted(CACHE_DIR.glob(f"kline_{code}_{start}_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if old_caches:
            return pd.read_csv(old_caches[0], parse_dates=["date"])
        return pd.DataFrame()

    old_caches = sorted(CACHE_DIR.glob(f"kline_{code}_{start}_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    try:
        payload = request_json(kline_url(code, start, end), retries=1, timeout=8)
    except Exception:
        if old_caches:
            return pd.read_csv(old_caches[0], parse_dates=["date"])
        return pd.DataFrame()
    rows = [line.split(",") for line in payload.get("data", {}).get("klines", [])]
    columns = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def holder_detail_features(code: str, refresh: bool) -> dict:
    cache = CACHE_DIR / f"holder_detail_{code}.csv"
    if cache.exists() and not refresh:
        detail = pd.read_csv(cache, parse_dates=["holder_date"])
    elif not refresh:
        old_caches = sorted(CACHE_DIR.glob(f"holder_detail_{code}.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if old_caches:
            detail = pd.read_csv(old_caches[0], parse_dates=["holder_date"])
        else:
            detail = pd.DataFrame()
    else:
        try:
            rows = request_json(holder_detail_url(code), retries=1, timeout=8).get("result", {}).get("data", [])
        except Exception:
            return {"code": code, "updates_180": 0, "updates_365": 0, "consecutive_downs": 0, "holder_3_change": np.nan}
        detail = pd.DataFrame(
            [
                {
                    "code": str(item.get("SECURITY_CODE", "")).zfill(6),
                    "holder_num": safe_num(item.get("HOLDER_NUM")),
                    "holder_change_pct": safe_num(item.get("HOLDER_NUM_RATIO")) / 100,
                    "holder_date": pd.to_datetime(item.get("END_DATE"), errors="coerce"),
                }
                for item in rows
            ]
        )
        detail.to_csv(cache, index=False, encoding="utf-8-sig")

    empty = {"code": code, "updates_180": 0, "updates_365": 0, "consecutive_downs": 0, "holder_3_change": np.nan}
    if detail.empty:
        return empty
    today = pd.Timestamp.today().normalize()
    detail = detail.dropna(subset=["holder_date"]).sort_values("holder_date", ascending=False)
    if detail.empty:
        return empty
    updates_180 = int((detail["holder_date"] >= today - pd.Timedelta(days=180)).sum())
    updates_365 = int((detail["holder_date"] >= today - pd.Timedelta(days=365)).sum())
    consecutive_downs = 0
    for value in detail["holder_change_pct"].head(6):
        if pd.notna(value) and value < 0:
            consecutive_downs += 1
        else:
            break
    holder_3_change = np.nan
    if len(detail) >= 3 and detail.iloc[2]["holder_num"] > 0:
        holder_3_change = detail.iloc[0]["holder_num"] / detail.iloc[2]["holder_num"] - 1
    return {
        "code": code,
        "updates_180": updates_180,
        "updates_365": updates_365,
        "consecutive_downs": consecutive_downs,
        "holder_3_change": holder_3_change,
    }


def days_since_max(series: pd.Series) -> int:
    clean = series.dropna()
    if clean.empty:
        return 999
    return len(clean) - 1 - int(np.argmax(clean.to_numpy()))


def kline_features(code: str, start: str, end: str, refresh: bool) -> dict:
    df = fetch_kline(code, start, end, refresh)
    if len(df) < 160:
        return {"code": code, "kline_ok": False}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    latest = df.iloc[-1]
    returns = close.pct_change()
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    high20 = high.tail(20).max()
    high60 = high.tail(60).max()
    high120 = high.tail(120).max()
    low20 = low.tail(20).min()
    low60 = low.tail(60).min()
    avg_amount_5 = df["amount"].tail(5).mean()
    avg_amount_20 = df["amount"].tail(20).mean()
    avg_amount_60 = df["amount"].tail(60).mean()
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean().iloc[-1] / latest["close"]
    close_position_20 = (latest["close"] - low20) / max(high20 - low20, 1e-9)
    close_position_60 = (latest["close"] - low60) / max(high60 - low60, 1e-9)

    return {
        "code": code,
        "kline_ok": True,
        "ret_3": close.iloc[-1] / close.iloc[-4] - 1,
        "ret_5": close.iloc[-1] / close.iloc[-6] - 1,
        "ret_10": close.iloc[-1] / close.iloc[-11] - 1,
        "ret_20": close.iloc[-1] / close.iloc[-21] - 1,
        "ret_60": close.iloc[-1] / close.iloc[-61] - 1,
        "ret_120": close.iloc[-1] / close.iloc[-121] - 1,
        "volatility_20": returns.tail(20).std() * math.sqrt(252),
        "volatility_60": returns.tail(60).std() * math.sqrt(252),
        "atr14": atr14,
        "amount_ratio_5_20": avg_amount_5 / avg_amount_20 if avg_amount_20 else np.nan,
        "amount_ratio_20_60": avg_amount_20 / avg_amount_60 if avg_amount_60 else np.nan,
        "turnover_20": df["turnover"].tail(20).mean() / 100,
        "above_ma5": bool(latest["close"] > ma5.iloc[-1]),
        "above_ma10": bool(latest["close"] > ma10.iloc[-1]),
        "above_ma20": bool(latest["close"] > ma20.iloc[-1]),
        "above_ma60": bool(latest["close"] > ma60.iloc[-1]),
        "above_ma120": bool(latest["close"] > ma120.iloc[-1]),
        "ma5_gt_ma10": bool(ma5.iloc[-1] > ma10.iloc[-1]),
        "ma10_gt_ma20": bool(ma10.iloc[-1] > ma20.iloc[-1]),
        "ma20_gt_ma60": bool(ma20.iloc[-1] > ma60.iloc[-1]),
        "ma60_gt_ma120": bool(ma60.iloc[-1] > ma120.iloc[-1]),
        "ma10_slope_5": ma10.iloc[-1] / ma10.iloc[-6] - 1,
        "ma20_slope_10": ma20.iloc[-1] / ma20.iloc[-11] - 1,
        "ma60_slope_20": ma60.iloc[-1] / ma60.iloc[-21] - 1,
        "dist_ma20": latest["close"] / ma20.iloc[-1] - 1,
        "dist_ma60": latest["close"] / ma60.iloc[-1] - 1,
        "high_gap_20": 1 - latest["close"] / high20,
        "high_gap_60": 1 - latest["close"] / high60,
        "high_gap_120": 1 - latest["close"] / high120,
        "days_since_20_high": days_since_max(high.tail(20)),
        "days_since_60_high": days_since_max(high.tail(60)),
        "days_since_120_high": days_since_max(high.tail(120)),
        "close_position_20": close_position_20,
        "close_position_60": close_position_60,
    }


def is_main_board(code: str) -> bool:
    return code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def build_base_universe(market: pd.DataFrame, holders: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = market.merge(holders, on="code", how="left")
    today = pd.Timestamp.today().normalize()
    df["holder_age_days"] = (today - df["holder_date"]).dt.days
    non_st = (
        ~df["name"].str.contains("ST", na=False)
        & ~df["name"].str.contains("退", na=False)
        & ~df["name"].str.startswith("N", na=False)
        & ~df["name"].str.startswith("C", na=False)
    )
    mask = (
        df["code"].map(is_main_board)
        & non_st
        & df["close"].notna()
        & df["float_mv"].between(args.min_float_mv * 1e8, args.max_float_mv * 1e8)
        & (df["amount"] >= args.min_amount * 1e8)
        & (df["amount"] <= args.max_amount * 1e8)
        & df["holder_change_pct"].notna()
        & (df["holder_change_pct"] <= args.max_latest_holder_change)
        & (df["holder_age_days"] <= args.max_holder_age)
        & ~((df["pct_chg"] >= 0.095) & (df["close"] >= df["high"]))
    )
    base = df.loc[mask].copy()
    industry_strength = base.groupby("industry")["pct_chg"].mean().rename("industry_pct_chg")
    base = base.join(industry_strength, on="industry")
    base["industry_heat_rank"] = base["industry_pct_chg"].rank(pct=True)
    base["pre_score"] = (
        0.28 * pct_rank(base["amount"])
        + 0.20 * pct_rank(base["turnover"])
        + 0.18 * pct_rank(-base["holder_change_pct"].clip(upper=args.max_latest_holder_change))
        + 0.18 * pct_rank(base["pct_chg"])
        + 0.16 * base["industry_heat_rank"].fillna(0)
    )
    return base.sort_values("pre_score", ascending=False).head(args.max_scan).reset_index(drop=True)


def enrich(base: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    krows = []
    for idx, code in enumerate(base["code"].astype(str), start=1):
        print(f"走势分析 {idx}/{len(base)}: {code}")
        try:
            krows.append(kline_features(code, args.kline_start, args.end, args.refresh))
        except Exception as exc:
            krows.append({"code": code, "kline_ok": False, "error": str(exc)})
        time.sleep(args.sleep)

    df = base.merge(pd.DataFrame(krows), on="code", how="left")
    df = df[df["kline_ok"] == True].copy()
    if df.empty:
        df["signal_tier"] = []
        return df

    df["right_trend_ok"] = (
        df["above_ma20"]
        & df["above_ma60"]
        & df["ma10_gt_ma20"]
        & df["ma20_gt_ma60"]
        & (df["ma20_slope_10"] > 0)
        & (df["ma60_slope_20"] > 0)
    )
    df["ten_day_momentum_ok"] = (
        df["ret_5"].between(args.min_ret5, args.max_ret5)
        & df["ret_10"].between(args.min_ret10, args.max_ret10)
        & df["ret_20"].between(args.min_ret20, args.max_ret20)
        & df["ret_60"].between(args.min_ret60, args.max_ret60)
        & df["dist_ma20"].between(args.min_dist_ma20, args.max_dist_ma20)
    )
    df["pressure_ok"] = (
        (df["days_since_60_high"] >= args.min_days_since_60_high)
        & (df["days_since_120_high"] >= args.min_days_since_120_high)
        & df["high_gap_60"].between(args.min_high_gap_60, args.max_high_gap_60)
        & df["high_gap_120"].between(args.min_high_gap_120, args.max_high_gap_120)
    )
    df["volume_ok"] = (
        df["amount_ratio_5_20"].between(args.min_amount_ratio_5_20, args.max_amount_ratio_5_20)
        & (df["amount_ratio_20_60"] >= args.min_amount_ratio_20_60)
        & df["turnover_20"].between(args.min_turnover_20, args.max_turnover_20)
    )
    df["risk_ok"] = (
        (df["volatility_60"] <= args.max_volatility_60)
        & (df["atr14"] <= args.max_atr14)
        & df["close_position_20"].between(args.min_close_position_20, args.max_close_position_20)
    )
    pre_ok = df["right_trend_ok"] & df["ten_day_momentum_ok"] & df["pressure_ok"] & df["volume_ok"] & df["risk_ok"]
    df = df[pre_ok].copy()
    if df.empty:
        df["signal_tier"] = []
        return df

    hrows = []
    for idx, code in enumerate(df["code"].astype(str), start=1):
        print(f"筹码分析 {idx}/{len(df)}: {code}")
        try:
            hrows.append(holder_detail_features(code, args.refresh))
        except Exception as exc:
            hrows.append(
                {
                    "code": code,
                    "updates_180": 0,
                    "updates_365": 0,
                    "consecutive_downs": 0,
                    "holder_3_change": np.nan,
                    "error": str(exc),
                }
            )
        time.sleep(args.sleep)

    df = df.merge(pd.DataFrame(hrows), on="code", how="left")
    df = df[df["updates_180"] >= args.min_holder_updates_180].copy()
    if df.empty:
        df["signal_tier"] = []
        return df

    df["chip_ok"] = (
        (df["holder_change_pct"] < 0)
        | (df["holder_3_change"] <= -0.02)
        | (df["consecutive_downs"] >= 2)
    )
    df["strict_ok"] = (
        df["chip_ok"]
        & df["right_trend_ok"]
        & df["ten_day_momentum_ok"]
        & df["pressure_ok"]
        & df["volume_ok"]
        & df["risk_ok"]
    )
    df["signal_tier"] = np.where(df["strict_ok"], STRONG_TIER, WATCH_TIER)
    return df.reset_index(drop=True)


def score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    line_stack = (
        0.15 * df["above_ma20"].astype(float)
        + 0.15 * df["above_ma60"].astype(float)
        + 0.15 * df["ma10_gt_ma20"].astype(float)
        + 0.20 * df["ma20_gt_ma60"].astype(float)
        + 0.10 * df["ma60_gt_ma120"].astype(float)
        + 0.15 * pct_rank(df["ma20_slope_10"])
        + 0.10 * pct_rank(df["ma60_slope_20"])
    )
    tactical_window = (
        0.25 * band_score(df["ret_10"], 0.00, 0.14, 0.06)
        + 0.20 * band_score(df["ret_20"], 0.02, 0.22, 0.10)
        + 0.20 * band_score(df["ret_60"], 0.08, 0.40, 0.20)
        + 0.20 * band_score(df["dist_ma20"], 0.00, 0.12, 0.05)
        + 0.15 * band_score(df["close_position_20"], 0.42, 0.88, 0.68)
    )
    df["trend_score"] = 0.55 * line_stack + 0.45 * tactical_window
    df["pressure_score"] = (
        0.35 * band_score(df["high_gap_60"], 0.05, 0.20, 0.11)
        + 0.35 * band_score(df["high_gap_120"], 0.06, 0.28, 0.15)
        + 0.15 * band_score(df["days_since_60_high"], 8, 45, 18)
        + 0.15 * band_score(df["days_since_120_high"], 12, 70, 28)
    )
    df["volume_score"] = (
        0.30 * band_score(df["amount_ratio_5_20"], 0.85, 2.20, 1.35)
        + 0.25 * band_score(df["amount_ratio_20_60"], 0.95, 2.20, 1.25)
        + 0.20 * band_score(df["turnover_20"], 0.015, 0.11, 0.045)
        + 0.15 * pct_rank(df["amount"])
        + 0.10 * df["industry_heat_rank"].fillna(0)
    )
    df["chip_score"] = (
        0.35 * pct_rank(-df["holder_change_pct"])
        + 0.25 * pct_rank(-df["holder_3_change"])
        + 0.25 * pct_rank(df["updates_180"])
        + 0.15 * (np.minimum(df["consecutive_downs"], 3) / 3)
    )
    df["risk_score"] = (
        0.35 * pct_rank(df["volatility_60"], ascending=False)
        + 0.25 * pct_rank(df["atr14"], ascending=False)
        + 0.20 * (df["ret_10"] <= 0.14).astype(float)
        + 0.20 * df["pressure_ok"].astype(float)
    )
    tier_bonus = np.where(df["signal_tier"] == STRONG_TIER, 0.08, 0.0)
    df["total_score"] = (
        0.42 * df["trend_score"]
        + 0.20 * df["volume_score"]
        + 0.16 * df["pressure_score"]
        + 0.12 * df["chip_score"]
        + 0.10 * df["risk_score"]
        + tier_bonus
    )
    df["rank"] = df["total_score"].rank(method="first", ascending=False).astype(int)
    return df.sort_values(["signal_tier", "total_score"], ascending=[True, False]).reset_index(drop=True)


def fmt_pct(x: object) -> str:
    return "-" if pd.isna(x) else f"{float(x):.2%}"


def fmt_yi(x: object) -> str:
    return "-" if pd.isna(x) else f"{float(x) / 1e8:.1f}亿"


def reason(row: pd.Series) -> str:
    parts = [
        f"10日涨幅{row['ret_10']:.1%}",
        f"20日涨幅{row['ret_20']:.1%}",
        f"60日涨幅{row['ret_60']:.1%}",
        f"离60日高点{row['high_gap_60']:.1%}",
        f"距60日高点{int(row['days_since_60_high'])}日",
    ]
    if row["right_trend_ok"]:
        parts.append("右侧均线结构通过")
    if row["volume_ok"]:
        parts.append(f"量能温和放大{row['amount_ratio_5_20']:.2f}倍")
    if row["holder_change_pct"] < 0:
        parts.append(f"最新户数下降{row['holder_change_pct']:.1%}")
    elif row["holder_3_change"] < 0:
        parts.append(f"近3期户数下降{row['holder_3_change']:.1%}")
    return "；".join(parts)


def risk(row: pd.Series) -> str:
    risks = []
    if not row["pressure_ok"]:
        risks.append("前高压力未完全通过")
    if row["high_gap_60"] < 0.05:
        risks.append("离60日高点太近")
    if row["high_gap_60"] > 0.20:
        risks.append("离60日高点太远")
    if row["ret_10"] > 0.14 or row["ret_20"] > 0.22:
        risks.append("短线涨幅偏高")
    if row["volatility_60"] > 0.55 or row["atr14"] > 0.075:
        risks.append("波动偏大")
    if not row["chip_ok"]:
        risks.append("散户减少趋势不够明确")
    return "；".join(risks) or "主要风险未触发"


def failed_rules(row: pd.Series) -> str:
    checks = [
        ("右侧趋势", row.get("right_trend_ok", False)),
        ("10日动能窗口", row.get("ten_day_momentum_ok", False)),
        ("前高压力", row.get("pressure_ok", False)),
        ("温和放量", row.get("volume_ok", False)),
        ("波动风险", row.get("risk_ok", False)),
        ("散户减少", row.get("chip_ok", False)),
    ]
    failed = [name for name, ok in checks if not bool(ok)]
    return "全部通过" if not failed else "未通过: " + "、".join(failed)


def write_report(scored: pd.DataFrame, base_count: int, args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strong = scored[scored["signal_tier"] == STRONG_TIER].head(args.strong_top).copy()
    watch = scored[scored["signal_tier"] != STRONG_TIER].head(args.strong_top).copy()
    strict_count = int((scored["signal_tier"] == STRONG_TIER).sum()) if not scored.empty else 0

    def build_rows(frame: pd.DataFrame) -> str:
        rows = []
        for row in frame.itertuples(index=False):
            s = pd.Series(row._asdict())
            rows.append(
                "<tr>"
                f"<td>{int(s['rank'])}</td><td>{s['code']}</td><td>{html.escape(str(s['name']))}</td>"
                f"<td><span class='tier'>{html.escape(str(s['signal_tier']))}</span></td>"
                f"<td>{html.escape(str(s['industry']))}</td><td>{fmt_yi(s['float_mv'])}</td>"
                f"<td>{fmt_pct(s['ret_10'])}</td><td>{fmt_pct(s['ret_20'])}</td><td>{fmt_pct(s['ret_60'])}</td>"
                f"<td>{fmt_pct(s['high_gap_60'])}</td><td>{int(s['days_since_60_high'])}</td>"
                f"<td>{fmt_pct(s['amount_ratio_5_20'] - 1)}</td><td>{fmt_pct(s['holder_change_pct'])}</td>"
                f"<td>{int(s['updates_180'])}</td><td>{s['total_score']:.4f}</td>"
                f"<td class='wide'>{html.escape(reason(s))}</td>"
                f"<td class='wide'>{html.escape(failed_rules(s))}</td>"
                f"<td class='wide'>{html.escape(risk(s))}</td>"
                "</tr>"
            )
        return "".join(rows)

    strong_body = build_rows(strong)
    watch_body = build_rows(watch)
    if strong.empty:
        strong_body = "<tr><td colspan='18'>今天没有严格强信号。宁可空仓观察，也不硬凑走势差的票。</td></tr>"
    if watch.empty:
        watch_body = "<tr><td colspan='18'>今天没有观察候选。</td></tr>"

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>A股主板10日短线选股报告</title>
  <style>
    body {{ margin:0; font-family:"Microsoft YaHei", Arial, sans-serif; background:#f6f8fb; color:#111827; }}
    main {{ max-width:1480px; margin:0 auto; padding:28px 24px 52px; }}
    h1 {{ margin:0 0 10px; font-size:31px; letter-spacing:0; }}
    h2 {{ margin:26px 0 12px; font-size:22px; }}
    .meta {{ color:#5b677a; line-height:1.8; margin-bottom:18px; }}
    .cards {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; margin:18px 0; }}
    .card {{ background:#fff; border:1px solid #dfe5ee; border-radius:8px; padding:14px; }}
    .label {{ color:#64748b; font-size:13px; }}
    .value {{ font-size:23px; font-weight:700; margin-top:6px; }}
    .panel {{ background:#fff; border:1px solid #dfe5ee; border-radius:8px; padding:18px; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; white-space:nowrap; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:9px 8px; text-align:left; vertical-align:top; }}
    th {{ background:#f8fafc; color:#475569; position:sticky; top:0; }}
    .wide {{ min-width:210px; max-width:330px; white-space:normal; line-height:1.5; }}
    .tier {{ display:inline-block; border:1px solid #b6c2d4; border-radius:6px; padding:2px 6px; background:#f8fafc; }}
    .note {{ margin-top:14px; color:#64748b; line-height:1.7; font-size:13px; }}
  </style>
</head>
<body>
<main>
  <h1>A股主板10日短线选股报告</h1>
  <div class="meta">
    生成时间：{generated}<br/>
    新策略：主板非ST，中小流通市值，右侧趋势优先；要求10日内有动能但不过热，最近一段时间不创新高，离前高保留缓冲；散户减少和更新频率作为加分与强信号门槛。<br/>
    使用提醒：这是候选筛选，不是买入指令。短线更适合配合盘中承接、止损和市场环境确认。
  </div>
  <div class="cards">
    <div class="card"><div class="label">强信号展示</div><div class="value">{len(strong)}</div></div>
    <div class="card"><div class="label">强信号总数</div><div class="value">{strict_count}</div></div>
    <div class="card"><div class="label">最终候选</div><div class="value">{len(scored)}</div></div>
    <div class="card"><div class="label">粗筛通过</div><div class="value">{base_count}</div></div>
  </div>
  <h2>10日短线强信号 {len(strong)} 只</h2>
  <div class="panel">
    <table>
      <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>等级</th><th>行业</th><th>流通市值</th><th>10日涨幅</th><th>20日涨幅</th><th>60日涨幅</th><th>离60日高点</th><th>距高点天数</th><th>5日量能</th><th>户数变化</th><th>半年更新</th><th>总分</th><th class="wide">入选原因</th><th class="wide">规则检查</th><th class="wide">风险提示</th></tr></thead>
      <tbody>{strong_body}</tbody>
    </table>
  </div>
  <h2>观察候选前 {len(watch)} 只</h2>
  <div class="panel">
    <table>
      <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>等级</th><th>行业</th><th>流通市值</th><th>10日涨幅</th><th>20日涨幅</th><th>60日涨幅</th><th>离60日高点</th><th>距高点天数</th><th>5日量能</th><th>户数变化</th><th>半年更新</th><th>总分</th><th class="wide">入选原因</th><th class="wide">规则检查</th><th class="wide">风险提示</th></tr></thead>
      <tbody>{watch_body}</tbody>
    </table>
  </div>
  <div class="note">
    强信号硬门槛：20日线在60日线上、20日线和60日线向上；10日涨幅0%到14%，20日涨幅2%到22%，60日涨幅8%到40%；距60日高点至少8个交易日，距120日高点至少12个交易日；离60日高点5%到20%，离120日高点6%到28%；5日量能相对20日为0.85到2.2倍；20日平均换手1.5%到11%；60日年化波动不超过60%，ATR不超过8%。
  </div>
</main>
</body>
</html>
"""
    (OUTPUT_DIR / "report.html").write_text(report, encoding="utf-8")


def write_outputs(scored: pd.DataFrame, base_count: int, args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_DIR / "stock_candidates.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    scored[scored["signal_tier"] == STRONG_TIER].head(args.strong_top).to_csv(
        OUTPUT_DIR / "strong_signals.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )
    scored[scored["signal_tier"] != STRONG_TIER].head(args.strong_top).to_csv(
        OUTPUT_DIR / "watchlist.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )
    write_report(scored, base_count, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-share main-board 10-day tactical stock screener.")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--kline-start", default="20240101")
    parser.add_argument("--min-float-mv", type=float, default=25)
    parser.add_argument("--max-float-mv", type=float, default=350)
    parser.add_argument("--min-amount", type=float, default=0.8)
    parser.add_argument("--max-amount", type=float, default=35)
    parser.add_argument("--max-holder-age", type=int, default=180)
    parser.add_argument("--max-latest-holder-change", type=float, default=0.04)
    parser.add_argument("--min-holder-updates-180", type=int, default=2)
    parser.add_argument("--min-ret5", type=float, default=-0.03)
    parser.add_argument("--max-ret5", type=float, default=0.10)
    parser.add_argument("--min-ret10", type=float, default=0.00)
    parser.add_argument("--max-ret10", type=float, default=0.14)
    parser.add_argument("--min-ret20", type=float, default=0.02)
    parser.add_argument("--max-ret20", type=float, default=0.22)
    parser.add_argument("--min-ret60", type=float, default=0.08)
    parser.add_argument("--max-ret60", type=float, default=0.40)
    parser.add_argument("--min-dist-ma20", type=float, default=0.00)
    parser.add_argument("--max-dist-ma20", type=float, default=0.12)
    parser.add_argument("--min-days-since-60-high", type=int, default=8)
    parser.add_argument("--min-days-since-120-high", type=int, default=12)
    parser.add_argument("--min-high-gap-60", type=float, default=0.05)
    parser.add_argument("--max-high-gap-60", type=float, default=0.20)
    parser.add_argument("--min-high-gap-120", type=float, default=0.06)
    parser.add_argument("--max-high-gap-120", type=float, default=0.28)
    parser.add_argument("--min-amount-ratio-5-20", type=float, default=0.85)
    parser.add_argument("--max-amount-ratio-5-20", type=float, default=2.20)
    parser.add_argument("--min-amount-ratio-20-60", type=float, default=0.95)
    parser.add_argument("--min-turnover-20", type=float, default=0.015)
    parser.add_argument("--max-turnover-20", type=float, default=0.11)
    parser.add_argument("--max-volatility-60", type=float, default=0.60)
    parser.add_argument("--max-atr14", type=float, default=0.08)
    parser.add_argument("--min-close-position-20", type=float, default=0.42)
    parser.add_argument("--max-close-position-20", type=float, default=0.88)
    parser.add_argument("--max-scan", type=int, default=420)
    parser.add_argument("--strong-top", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("获取全市场行情...")
    market = load_market(args.refresh)
    print("获取股东户数...")
    holders = load_holder_latest(args.refresh)
    base = build_base_universe(market, holders, args)
    print(f"粗筛通过 {len(base)} 只，开始趋势、前高压力、量能和筹码分析...")
    enriched = enrich(base, args)
    scored = score(enriched)
    strong_count = int((scored["signal_tier"] == STRONG_TIER).sum()) if not scored.empty else 0
    print(f"最终候选 {len(scored)} 只，其中10日短线强信号 {strong_count} 只。")
    write_outputs(scored, len(base), args)
    print(f"报告: {OUTPUT_DIR / 'report.html'}")
    print(f"强信号: {OUTPUT_DIR / 'strong_signals.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
