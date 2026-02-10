import os
import numpy as np
import pandas as pd
import ta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import DATA_DIR, SYMBOLS, INTERVAL_NAMES, PRIMARY_INTERVAL


def load_symbol_data(symbol: str) -> dict:
    symbol_dir = os.path.join(DATA_DIR, symbol)
    data = {}

    for interval_key, interval_name in INTERVAL_NAMES.items():
        fpath = os.path.join(symbol_dir, f"klines_{interval_name}.parquet")
        if os.path.exists(fpath):
            data[interval_name] = pd.read_parquet(fpath)

    fr_path = os.path.join(symbol_dir, "funding_rate.parquet")
    if os.path.exists(fr_path):
        data["funding_rate"] = pd.read_parquet(fr_path)

    oi_path = os.path.join(symbol_dir, "open_interest.parquet")
    if os.path.exists(oi_path):
        data["open_interest"] = pd.read_parquet(oi_path)

    return data


def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]
    o = df["open"]

    for p in [7, 14, 21, 50]:
        df[f"sma_{p}"] = ta.trend.sma_indicator(c, window=p)
        df[f"ema_{p}"] = ta.trend.ema_indicator(c, window=p)

    df["rsi_7"] = ta.momentum.rsi(c, window=7)
    df["rsi_14"] = ta.momentum.rsi(c, window=14)
    df["rsi_21"] = ta.momentum.rsi(c, window=21)

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb_20 = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb_20.bollinger_hband()
    df["bb_lower"] = bb_20.bollinger_lband()
    df["bb_mid"] = bb_20.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = bb_20.bollinger_pband()

    for p in [7, 14, 21]:
        df[f"atr_{p}"] = ta.volatility.average_true_range(h, l, c, window=p)

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["adx"] = ta.trend.adx(h, l, c, window=14)
    df["adx_pos"] = ta.trend.adx_pos(h, l, c, window=14)
    df["adx_neg"] = ta.trend.adx_neg(h, l, c, window=14)

    df["cci_14"] = ta.trend.cci(h, l, c, window=14)
    df["cci_20"] = ta.trend.cci(h, l, c, window=20)

    df["williams_r"] = ta.momentum.williams_r(h, l, c, window=14)

    df["mfi_14"] = ta.volume.money_flow_index(h, l, c, v, window=14)

    df["obv"] = ta.volume.on_balance_volume(c, v)
    df["vwap_ratio"] = (v * (h + l + c) / 3).cumsum() / v.cumsum()

    df["cmf"] = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    ic = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df["ichimoku_a"] = ic.ichimoku_a()
    df["ichimoku_b"] = ic.ichimoku_b()
    df["ichimoku_base"] = ic.ichimoku_base_line()
    df["ichimoku_conv"] = ic.ichimoku_conversion_line()

    df["psar"] = ta.trend.PSARIndicator(h, l, c).psar()

    for p in [1, 2, 3, 5, 10, 20]:
        df[f"return_{p}"] = c.pct_change(p)

    for p in [5, 10, 20, 50]:
        df[f"volatility_{p}"] = c.pct_change().rolling(p).std()

    for p in [5, 10, 20]:
        df[f"volume_sma_{p}"] = v.rolling(p).mean()
        df[f"volume_ratio_{p}"] = v / v.rolling(p).mean()

    df["body_size"] = abs(c - o) / (h - l + 1e-10)
    df["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (h - l + 1e-10)
    df["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (h - l + 1e-10)
    df["is_green"] = (c > o).astype(int)

    for p in [3, 5, 10]:
        df[f"green_ratio_{p}"] = df["is_green"].rolling(p).mean()

    for p in [5, 10, 20]:
        df[f"high_max_{p}"] = h.rolling(p).max()
        df[f"low_min_{p}"] = l.rolling(p).min()
        df[f"range_pct_{p}"] = (df[f"high_max_{p}"] - df[f"low_min_{p}"]) / c

    df["dist_from_high_20"] = (c - h.rolling(20).max()) / c
    df["dist_from_low_20"] = (c - l.rolling(20).min()) / c

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def merge_multi_timeframe(data: dict, primary_tf: str = "1h") -> pd.DataFrame:
    df = data.get(primary_tf)
    if df is None:
        raise ValueError(f"Primary timeframe {primary_tf} not found")

    df = df.copy()
    df = add_ta_indicators(df)

    higher_tfs = {"4h": "4h"}
    for tf_name, tf_key in higher_tfs.items():
        if tf_key in data:
            htf = data[tf_key].copy()
            htf = add_ta_indicators(htf)

            htf_cols = ["timestamp"]
            for col in ["rsi_14", "macd", "macd_diff", "bb_pct", "bb_width",
                         "adx", "atr_14", "volume_ratio_10", "return_1", "return_5"]:
                if col in htf.columns:
                    new_name = f"{tf_name}_{col}"
                    htf[new_name] = htf[col]
                    htf_cols.append(new_name)

            htf_merge = htf[htf_cols].copy()
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                htf_merge.sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )

    if "funding_rate" in data and not data["funding_rate"].empty:
        fr = data["funding_rate"][["timestamp", "funding_rate"]].copy()
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            fr.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
        df["funding_rate"] = df["funding_rate"].ffill()

    if "open_interest" in data and not data["open_interest"].empty:
        oi = data["open_interest"][["timestamp", "open_interest"]].copy()
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            oi.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
        df["open_interest"] = df["open_interest"].ffill()
        df["oi_change"] = df["open_interest"].pct_change()
        for p in [5, 10, 20]:
            df[f"oi_sma_{p}"] = df["open_interest"].rolling(p).mean()
            df[f"oi_ratio_{p}"] = df["open_interest"] / df[f"oi_sma_{p}"]

    return df


def create_labels(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

    df["label"] = 0
    threshold = df["future_return"].std() * 0.3
    df.loc[df["future_return"] > threshold, "label"] = 1
    df.loc[df["future_return"] < -threshold, "label"] = -1

    df["future_high"] = df["high"].rolling(horizon).max().shift(-horizon)
    df["future_low"] = df["low"].rolling(horizon).min().shift(-horizon)
    df["future_max_up"] = (df["future_high"] - df["close"]) / df["close"]
    df["future_max_down"] = (df["close"] - df["future_low"]) / df["close"]

    return df


def compute_time_weights(df: pd.DataFrame, decay_factor: float = 0.0005) -> np.ndarray:
    n = len(df)
    indices = np.arange(n)
    weights = np.exp(decay_factor * (indices - n + 1))
    weights = weights / weights.mean()
    return weights


def build_features_for_symbol(symbol: str) -> pd.DataFrame:
    print(f"Building features for {symbol}...")
    data = load_symbol_data(symbol)

    primary_tf = INTERVAL_NAMES[PRIMARY_INTERVAL]
    df = merge_multi_timeframe(data, primary_tf=primary_tf)

    df = create_labels(df)

    exclude_cols = {
        "timestamp", "datetime", "open", "high", "low", "close", "volume", "turnover",
        "future_return", "future_high", "future_low", "future_max_up", "future_max_down",
        "label", "hour", "day_of_week"
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df["symbol"] = symbol

    print(f"  {symbol}: {len(df)} rows, {len(feature_cols)} features")
    return df


def build_all_features() -> dict:
    all_data = {}
    for symbol in SYMBOLS:
        df = build_features_for_symbol(symbol)
        out_path = os.path.join(DATA_DIR, symbol, "features.parquet")
        df.to_parquet(out_path, index=False)
        all_data[symbol] = df
        print(f"  Saved features to {out_path}")
    return all_data


if __name__ == "__main__":
    build_all_features()
