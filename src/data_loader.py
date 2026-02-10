import os
import time
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import (
    SYMBOLS, INTERVALS, INTERVAL_NAMES, START_DATE, END_DATE, DATA_DIR,
)

BINANCE_VISION_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_LIMIT = 1000

OKX_BASE_URL = "https://www.okx.com"
OKX_LIMIT = 100

BINANCE_INTERVALS = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "240": "4h",
}

OKX_SYMBOLS = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
    "SOLUSDT": "SOL-USDT-SWAP",
    "XRPUSDT": "XRP-USDT-SWAP",
    "BNBUSDT": "BNB-USDT-SWAP",
    "DOGEUSDT": "DOGE-USDT-SWAP",
}

RATE_DELAY = 0.05


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    bi = BINANCE_INTERVALS[interval]
    all_rows = []
    cursor = start_ms
    interval_ms = int(interval) * 60 * 1000
    total_expected = (end_ms - start_ms) // interval_ms + 1

    pbar = tqdm(total=total_expected, desc=f"  {symbol} {INTERVAL_NAMES[interval]}", unit="candles", leave=False)

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": bi,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
        }
        rows = []
        for attempt in range(5):
            try:
                resp = requests.get(BINANCE_VISION_URL, params=params, timeout=20)
                if resp.status_code == 200:
                    rows = resp.json()
                    break
                elif resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    break
            except Exception:
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))

        if not rows:
            break

        all_rows.extend(rows)
        pbar.update(len(rows))

        last_ts = int(rows[-1][0])
        cursor = last_ts + interval_ms

        if len(rows) < BINANCE_LIMIT:
            break

        time.sleep(RATE_DELAY)

    pbar.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.rename(columns={"quote_volume": "turnover"}, inplace=True)

    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]

    return df


def fetch_funding_rate(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    okx_inst = OKX_SYMBOLS.get(symbol)
    if not okx_inst:
        return pd.DataFrame()

    url = f"{OKX_BASE_URL}/api/v5/public/funding-rate-history"
    all_rows = []
    cursor = ""

    pbar = tqdm(desc=f"  {symbol} funding", unit="records", leave=False)

    while True:
        params = {
            "instId": okx_inst,
            "limit": "100",
        }
        if cursor:
            params["after"] = cursor

        data = {"code": "-1"}
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=15)
                data = resp.json()
                break
            except Exception:
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))

        if data.get("code") != "0":
            break

        rows = data.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        pbar.update(len(rows))

        oldest_ts = rows[-1].get("fundingTime", "")
        if oldest_ts and int(oldest_ts) <= start_ms:
            break

        cursor = oldest_ts
        if len(rows) < 100:
            break

        time.sleep(RATE_DELAY)

    pbar.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_numeric(df["fundingTime"])
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["timestamp", "funding_rate"]].drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]

    return df


def fetch_open_interest(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    okx_inst = OKX_SYMBOLS.get(symbol)
    if not okx_inst:
        return pd.DataFrame()

    url = f"{OKX_BASE_URL}/api/v5/rubik/stat/contracts/open-interest-history"
    all_rows = []

    pbar = tqdm(desc=f"  {symbol} OI", unit="records", leave=False)

    params = {
        "instId": okx_inst,
        "period": "1D",
    }
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            break
        except Exception:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
            else:
                data = {"code": "-1"}

    if data.get("code") == "0" and data.get("data"):
        all_rows.extend(data["data"])
        pbar.update(len(data["data"]))

    pbar.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "ts" in df.columns:
        df["timestamp"] = pd.to_numeric(df["ts"])
        oi_col = next((c for c in ["oi", "oiCcy"] if c in df.columns), None)
        if oi_col:
            df["open_interest"] = pd.to_numeric(df[oi_col], errors="coerce")
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    df = df[["timestamp", "open_interest"]].drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def download_all_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    start_ms = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Downloading data for {symbol}")
        print(f"  Klines: Binance Vision | Funding/OI: OKX")
        print(f"{'='*60}")

        symbol_dir = os.path.join(DATA_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        for interval in INTERVALS:
            fname = f"klines_{INTERVAL_NAMES[interval]}.parquet"
            fpath = os.path.join(symbol_dir, fname)
            if os.path.exists(fpath):
                existing = pd.read_parquet(fpath)
                print(f"  {INTERVAL_NAMES[interval]}: already have {len(existing)} candles, skipping")
                continue

            df = fetch_klines(symbol, interval, start_ms, end_ms)
            if not df.empty:
                df.to_parquet(fpath, index=False)
                print(f"  {INTERVAL_NAMES[interval]}: saved {len(df)} candles")
            else:
                print(f"  {INTERVAL_NAMES[interval]}: no data received")

        fr_path = os.path.join(symbol_dir, "funding_rate.parquet")
        if not os.path.exists(fr_path):
            df_fr = fetch_funding_rate(symbol, start_ms, end_ms)
            if not df_fr.empty:
                df_fr.to_parquet(fr_path, index=False)
                print(f"  Funding rate: saved {len(df_fr)} records")
            else:
                print(f"  Funding rate: no data")
        else:
            print(f"  Funding rate: already exists, skipping")

        oi_path = os.path.join(symbol_dir, "open_interest.parquet")
        if not os.path.exists(oi_path):
            df_oi = fetch_open_interest(symbol, start_ms, end_ms)
            if not df_oi.empty:
                df_oi.to_parquet(oi_path, index=False)
                print(f"  Open interest: saved {len(df_oi)} records")
            else:
                print(f"  Open interest: no data")
        else:
            print(f"  Open interest: already exists, skipping")

    print(f"\nAll data downloaded to {DATA_DIR}")


if __name__ == "__main__":
    download_all_data()
