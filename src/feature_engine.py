import os
import numpy as np
import pandas as pd
import ta
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import (
    DATA_DIR, SYMBOLS, PRIMARY_TF, ALL_TIMEFRAMES,
    HIGHER_TF_FEATURES, LOWER_TF_FEATURES, LABEL_HORIZON,
    TIME_WEIGHT_DECAY,
)


def load_symbol_data(symbol: str) -> dict:
    symbol_dir = os.path.join(DATA_DIR, symbol)
    data = {}
    for tf in ALL_TIMEFRAMES:
        fpath = os.path.join(symbol_dir, f"klines_{tf}.parquet")
        if os.path.exists(fpath):
            data[tf] = pd.read_parquet(fpath)
    fr_path = os.path.join(symbol_dir, "funding_rate.parquet")
    if os.path.exists(fr_path):
        data["funding_rate"] = pd.read_parquet(fr_path)
    return data


def add_ta_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    h, l, c, v, o = df["high"], df["low"], df["close"], df["volume"], df["open"]
    p = prefix

    for w in [7, 14, 21, 50]:
        df[f"{p}sma_{w}"] = ta.trend.sma_indicator(c, window=w)
        df[f"{p}ema_{w}"] = ta.trend.ema_indicator(c, window=w)

    df[f"{p}sma_100"] = ta.trend.sma_indicator(c, window=100)
    df[f"{p}sma_200"] = ta.trend.sma_indicator(c, window=200)
    df[f"{p}ema_100"] = ta.trend.ema_indicator(c, window=100)
    df[f"{p}ema_200"] = ta.trend.ema_indicator(c, window=200)

    for w in [7, 14, 21]:
        df[f"{p}rsi_{w}"] = ta.momentum.rsi(c, window=w)

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df[f"{p}macd"] = macd.macd()
    df[f"{p}macd_signal"] = macd.macd_signal()
    df[f"{p}macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df[f"{p}bb_upper"] = bb.bollinger_hband()
    df[f"{p}bb_lower"] = bb.bollinger_lband()
    df[f"{p}bb_mid"] = bb.bollinger_mavg()
    df[f"{p}bb_width"] = (df[f"{p}bb_upper"] - df[f"{p}bb_lower"]) / (df[f"{p}bb_mid"] + 1e-10)
    df[f"{p}bb_pct"] = bb.bollinger_pband()

    for w in [7, 14, 21]:
        df[f"{p}atr_{w}"] = ta.volatility.average_true_range(h, l, c, window=w)

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df[f"{p}stoch_k"] = stoch.stoch()
    df[f"{p}stoch_d"] = stoch.stoch_signal()

    df[f"{p}adx"] = ta.trend.adx(h, l, c, window=14)
    df[f"{p}adx_pos"] = ta.trend.adx_pos(h, l, c, window=14)
    df[f"{p}adx_neg"] = ta.trend.adx_neg(h, l, c, window=14)

    df[f"{p}cci_14"] = ta.trend.cci(h, l, c, window=14)
    df[f"{p}cci_20"] = ta.trend.cci(h, l, c, window=20)

    df[f"{p}williams_r"] = ta.momentum.williams_r(h, l, c, lbp=14)
    df[f"{p}mfi_14"] = ta.volume.money_flow_index(h, l, c, v, window=14)
    df[f"{p}obv"] = ta.volume.on_balance_volume(c, v)
    df[f"{p}cmf"] = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    vwap_cum = (v * (h + l + c) / 3).cumsum()
    v_cum = v.cumsum()
    df[f"{p}vwap_ratio"] = vwap_cum / (v_cum + 1e-10)

    ic = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df[f"{p}ichimoku_a"] = ic.ichimoku_a()
    df[f"{p}ichimoku_b"] = ic.ichimoku_b()
    df[f"{p}ichimoku_base"] = ic.ichimoku_base_line()
    df[f"{p}ichimoku_conv"] = ic.ichimoku_conversion_line()

    df[f"{p}psar"] = ta.trend.PSARIndicator(h, l, c).psar()

    for w in [1, 2, 3, 5, 10, 20]:
        df[f"{p}return_{w}"] = c.pct_change(w)

    for w in [5, 10, 20, 50]:
        df[f"{p}volatility_{w}"] = c.pct_change().rolling(w).std()

    for w in [5, 10, 20]:
        df[f"{p}volume_sma_{w}"] = v.rolling(w).mean()
        df[f"{p}volume_ratio_{w}"] = v / (v.rolling(w).mean() + 1e-10)

    df[f"{p}body_size"] = abs(c - o) / (h - l + 1e-10)
    df[f"{p}upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (h - l + 1e-10)
    df[f"{p}lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (h - l + 1e-10)
    df[f"{p}is_green"] = (c > o).astype(int)

    for w in [3, 5, 10]:
        df[f"{p}green_ratio_{w}"] = df[f"{p}is_green"].rolling(w).mean()

    for w in [5, 10, 20]:
        df[f"{p}high_max_{w}"] = h.rolling(w).max()
        df[f"{p}low_min_{w}"] = l.rolling(w).min()
        df[f"{p}range_pct_{w}"] = (df[f"{p}high_max_{w}"] - df[f"{p}low_min_{w}"]) / (c + 1e-10)

    df[f"{p}dist_from_high_20"] = (c - h.rolling(20).max()) / (c + 1e-10)
    df[f"{p}dist_from_low_20"] = (c - l.rolling(20).min()) / (c + 1e-10)

    return df


def add_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    kc = ta.volatility.KeltnerChannel(h, l, c, window=20, window_atr=10)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_width"] = (df["kc_upper"] - df["kc_lower"]) / (c + 1e-10)

    dc = ta.volatility.DonchianChannel(h, l, c, window=20)
    df["dc_upper"] = dc.donchian_channel_hband()
    df["dc_lower"] = dc.donchian_channel_lband()
    df["dc_width"] = (df["dc_upper"] - df["dc_lower"]) / (c + 1e-10)
    df["dc_pct"] = (c - df["dc_lower"]) / (df["dc_upper"] - df["dc_lower"] + 1e-10)

    for w in [5, 10, 20]:
        df[f"roc_{w}"] = ta.momentum.roc(c, window=w)

    df["trix"] = ta.trend.trix(c, window=15)
    df["mass_index"] = ta.trend.mass_index(h, l, window_fast=9, window_slow=25)

    aroon = ta.trend.AroonIndicator(h, l, window=25)
    df["aroon_up"] = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()
    df["aroon_diff"] = df["aroon_up"] - df["aroon_down"]

    df["dpo"] = ta.trend.dpo(c, window=20)
    df["ulcer_index"] = ta.volatility.ulcer_index(c, window=14)

    for w in [5, 10, 20]:
        log_ret = np.log(c / c.shift(1))
        df[f"realized_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(w)

    for w in [10, 20]:
        hl_ratio = np.log(h / l)
        df[f"parkinson_vol_{w}"] = hl_ratio.rolling(w).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2))), raw=True
        )

    for w in [7, 14, 21]:
        df[f"price_position_{w}"] = (c - l.rolling(w).min()) / (h.rolling(w).max() - l.rolling(w).min() + 1e-10)

    df["volume_trend"] = v.rolling(10).mean() / (v.rolling(50).mean() + 1e-10)
    df["volume_spike"] = v / (v.rolling(20).mean() + 1e-10)

    for w in [5, 10]:
        df[f"close_std_{w}"] = c.rolling(w).std() / (c.rolling(w).mean() + 1e-10)

    df["gap"] = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-10)

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    adx = df.get("adx")
    if adx is None:
        return df

    df["regime_trend"] = (adx > 25).astype(int)
    df["regime_strong_trend"] = (adx > 40).astype(int)

    vol_20 = df.get("volatility_20")
    if vol_20 is not None:
        vol_median = vol_20.rolling(100).median()
        df["regime_high_vol"] = (vol_20 > vol_median).astype(int)
        df["vol_regime_ratio"] = vol_20 / (vol_median + 1e-10)

    bb_width = df.get("bb_width")
    if bb_width is not None:
        bw_median = bb_width.rolling(100).median()
        df["regime_squeeze"] = (bb_width < bw_median * 0.7).astype(int)
        df["regime_expansion"] = (bb_width > bw_median * 1.5).astype(int)

    ret_20 = c.pct_change(20)
    df["regime_bullish"] = (ret_20 > 0.02).astype(int)
    df["regime_bearish"] = (ret_20 < -0.02).astype(int)

    sma_50 = df.get("sma_50")
    sma_200 = df.get("sma_200")
    if sma_50 is not None and sma_200 is not None:
        df["regime_golden_cross"] = (sma_50 > sma_200).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    hour = dt.dt.hour
    dow = dt.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def compute_indicators_for_tf(df: pd.DataFrame, tf_name: str, is_primary: bool) -> pd.DataFrame:
    df = df.copy()
    if is_primary:
        df = add_ta_indicators(df, prefix="")
        df = add_extended_indicators(df)
        df = add_regime_features(df)
        df = add_time_features(df)
    else:
        df = add_ta_indicators(df, prefix=f"{tf_name}_")
    return df


def merge_timeframes(data: dict) -> pd.DataFrame:
    primary = data.get(PRIMARY_TF)
    if primary is None:
        raise ValueError(f"Primary timeframe {PRIMARY_TF} not found")

    print(f"  Computing indicators on {PRIMARY_TF} ({len(primary)} rows)...")
    df = compute_indicators_for_tf(primary, PRIMARY_TF, is_primary=True)

    higher_tfs = [tf for tf in ["1h", "4h"] if tf in data and tf != PRIMARY_TF]
    for tf in higher_tfs:
        print(f"  Merging {tf} features...")
        htf = compute_indicators_for_tf(data[tf], tf, is_primary=False)
        htf_cols = ["timestamp"]
        prefix = f"{tf}_"
        for col in HIGHER_TF_FEATURES:
            full_col = f"{prefix}{col}"
            if full_col in htf.columns:
                htf_cols.append(full_col)
        htf_merge = htf[htf_cols].copy()
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            htf_merge.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    lower_tfs = [tf for tf in ["1m", "5m"] if tf in data and tf != PRIMARY_TF]
    for tf in lower_tfs:
        print(f"  Merging {tf} features...")
        ltf = compute_indicators_for_tf(data[tf], tf, is_primary=False)
        ltf_cols = ["timestamp"]
        prefix = f"{tf}_"
        for col in LOWER_TF_FEATURES:
            full_col = f"{prefix}{col}"
            if full_col in ltf.columns:
                ltf_cols.append(full_col)
        ltf_merge = ltf[ltf_cols].copy()
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            ltf_merge.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    if "funding_rate" in data and not data["funding_rate"].empty:
        fr = data["funding_rate"][["timestamp", "funding_rate"]].copy()
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            fr.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["funding_rate"] = df["funding_rate"].ffill()
        df["funding_rate_abs"] = df["funding_rate"].abs()
        df["funding_rate_ma"] = df["funding_rate"].rolling(8, min_periods=1).mean()

    return df


def add_cross_tf_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if "rsi_14" in df.columns and "1h_rsi_14" in df.columns:
        df["rsi_15m_vs_1h"] = df["rsi_14"] - df["1h_rsi_14"]
    if "rsi_14" in df.columns and "4h_rsi_14" in df.columns:
        df["rsi_15m_vs_4h"] = df["rsi_14"] - df["4h_rsi_14"]
    if "macd_diff" in df.columns and "1h_macd_diff" in df.columns:
        df["macd_15m_vs_1h"] = np.sign(df["macd_diff"]) - np.sign(df["1h_macd_diff"])
    if "atr_14" in df.columns and "1h_atr_14" in df.columns:
        df["atr_ratio_15m_1h"] = df["atr_14"] / (df["1h_atr_14"] + 1e-10)
    if "volume_ratio_10" in df.columns and "1h_volume_ratio_10" in df.columns:
        df["vol_ratio_15m_vs_1h"] = df["volume_ratio_10"] / (df["1h_volume_ratio_10"] + 1e-10)
    if "adx" in df.columns and "1h_adx" in df.columns:
        df["adx_15m_vs_1h"] = df["adx"] - df["1h_adx"]
    if "bb_width" in df.columns and "1h_bb_width" in df.columns:
        df["bbw_ratio_15m_1h"] = df["bb_width"] / (df["1h_bb_width"] + 1e-10)
    if "stoch_k" in df.columns and "1h_stoch_k" in df.columns:
        df["stoch_15m_vs_1h"] = df["stoch_k"] - df["1h_stoch_k"]

    return df


def add_cross_coin_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    btc_close = btc_df[["timestamp", "close"]].copy()
    btc_close.rename(columns={"close": "btc_close"}, inplace=True)

    df = pd.merge_asof(
        df.sort_values("timestamp"),
        btc_close.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    df["btc_return_1"] = df["btc_close"].pct_change(1)
    df["btc_return_5"] = df["btc_close"].pct_change(5)
    df["btc_return_20"] = df["btc_close"].pct_change(20)

    coin_ret = df["close"].pct_change(1)
    btc_ret = df["btc_return_1"]
    df["coin_btc_spread"] = coin_ret - btc_ret
    df["coin_btc_spread_ma"] = df["coin_btc_spread"].rolling(20, min_periods=1).mean()

    df["coin_btc_corr_20"] = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
    df["coin_btc_corr_50"] = coin_ret.rolling(50, min_periods=20).corr(btc_ret)

    df["coin_btc_beta"] = coin_ret.rolling(50, min_periods=20).cov(btc_ret) / (btc_ret.rolling(50, min_periods=20).var() + 1e-10)

    df.drop(columns=["btc_close"], inplace=True)

    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    horizon = LABEL_HORIZON
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


def compute_time_weights(df: pd.DataFrame, decay_factor: float = TIME_WEIGHT_DECAY) -> np.ndarray:
    n = len(df)
    indices = np.arange(n)
    weights = np.exp(decay_factor * (indices - n + 1))
    weights = weights / weights.mean()
    return weights


EXCLUDE_COLS = {
    "timestamp", "datetime", "open", "high", "low", "close", "volume", "turnover",
    "future_return", "future_high", "future_low", "future_max_up", "future_max_down",
    "label", "hour", "day_of_week", "symbol",
}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def build_features_for_symbol(symbol: str, btc_primary: pd.DataFrame = None) -> pd.DataFrame:
    print(f"\nBuilding features for {symbol}...")
    data = load_symbol_data(symbol)

    df = merge_timeframes(data)
    df = add_cross_tf_ratios(df)

    if btc_primary is not None and symbol != "BTCUSDT":
        df = add_cross_coin_features(df, btc_primary)

    df = create_labels(df)
    df["symbol"] = symbol

    feature_cols = get_feature_cols(df)
    print(f"  {symbol}: {len(df)} rows, {len(feature_cols)} features")
    return df


def build_all_features() -> dict:
    all_data = {}

    btc_data = load_symbol_data("BTCUSDT")
    btc_primary = btc_data.get(PRIMARY_TF)

    for symbol in SYMBOLS:
        df = build_features_for_symbol(symbol, btc_primary=btc_primary)
        out_path = os.path.join(DATA_DIR, symbol, "features_v2.parquet")
        df.to_parquet(out_path, index=False)
        all_data[symbol] = df
        print(f"  Saved features to {out_path}")

    return all_data


if __name__ == "__main__":
    build_all_features()
