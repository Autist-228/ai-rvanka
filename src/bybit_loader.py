import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import SYMBOLS, INTERVALS, INTERVAL_NAMES, DATA_DIR

BYBIT_BASE = "https://api.bybit.com"
BYBIT_LIMIT = 200

PROXY_HOST = os.environ.get("PROXY_HOST", "")
PROXY_PORT = os.environ.get("PROXY_PORT", "")
PROXY_USER = os.environ.get("PROXY_USER", "")
PROXY_PASS = os.environ.get("PROXY_PASS", "")

PROXIES = {}
if PROXY_HOST and PROXY_PORT:
    _proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}" if PROXY_USER else f"http://{PROXY_HOST}:{PROXY_PORT}"
    PROXIES = {"http": _proxy_url, "https": _proxy_url}

BYBIT_INTERVALS = {
    "1": "1",
    "5": "5",
    "15": "15",
    "60": "60",
    "240": "240",
}

BYBIT_DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "data_bybit")


def fetch_bybit_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    bi = BYBIT_INTERVALS[interval]
    all_rows = []
    cursor = end_ms
    interval_ms = int(interval) * 60 * 1000
    total_expected = (end_ms - start_ms) // interval_ms + 1

    pbar = tqdm(total=total_expected, desc=f"  {symbol} {INTERVAL_NAMES[interval]}", unit="candles", leave=False)

    while cursor > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bi,
            "end": cursor,
            "limit": BYBIT_LIMIT,
        }

        data = None
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{BYBIT_BASE}/v5/market/kline",
                    params=params,
                    proxies=PROXIES,
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("retCode") == 0:
                        break
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                if attempt < 4:
                    time.sleep(3 * (attempt + 1))

        if not data or data.get("retCode") != 0:
            print(f"    Error: {data}")
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        all_rows.extend(rows)
        pbar.update(len(rows))

        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor = oldest_ts - 1

        if len(rows) < BYBIT_LIMIT:
            break

        time.sleep(0.1)

    pbar.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]

    return df


def fetch_bybit_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    cursor = ""

    pbar = tqdm(desc=f"  {symbol} funding", unit="records", leave=False)

    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        data = None
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{BYBIT_BASE}/v5/market/funding/history",
                    params=params,
                    proxies=PROXIES,
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("retCode") == 0:
                        break
                time.sleep(2 * (attempt + 1))
            except Exception:
                if attempt < 4:
                    time.sleep(3 * (attempt + 1))

        if not data or data.get("retCode") != 0:
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        all_rows.extend(rows)
        pbar.update(len(rows))

        next_cursor = data.get("result", {}).get("nextPageCursor", "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

        time.sleep(0.1)

    pbar.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_numeric(df["fundingRateTimestamp"])
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["timestamp", "funding_rate"]].drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def fetch_bybit_oi(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    cursor = ""

    pbar = tqdm(desc=f"  {symbol} OI", unit="records", leave=False)

    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        data = None
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{BYBIT_BASE}/v5/market/open-interest",
                    params=params,
                    proxies=PROXIES,
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("retCode") == 0:
                        break
                time.sleep(2 * (attempt + 1))
            except Exception:
                if attempt < 4:
                    time.sleep(3 * (attempt + 1))

        if not data or data.get("retCode") != 0:
            break

        rows = data.get("result", {}).get("list", [])
        if not rows:
            break

        all_rows.extend(rows)
        pbar.update(len(rows))

        next_cursor = data.get("result", {}).get("nextPageCursor", "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

        time.sleep(0.1)

    pbar.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["open_interest"] = pd.to_numeric(df["openInterest"], errors="coerce")
    df = df[["timestamp", "open_interest"]].drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def download_bybit_3months():
    os.makedirs(BYBIT_DATA_DIR, exist_ok=True)

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=92)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Source: Bybit API (via proxy)")
    print(f"Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} (~3 months)")
    print(f"Saving to: {BYBIT_DATA_DIR}")

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Downloading {symbol} from Bybit")
        print(f"{'='*60}")

        symbol_dir = os.path.join(BYBIT_DATA_DIR, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        for interval in INTERVALS:
            fname = f"klines_{INTERVAL_NAMES[interval]}.parquet"
            fpath = os.path.join(symbol_dir, fname)
            if os.path.exists(fpath):
                existing = pd.read_parquet(fpath)
                print(f"  {INTERVAL_NAMES[interval]}: already have {len(existing)} candles, skipping")
                continue

            df = fetch_bybit_klines(symbol, interval, start_ms, end_ms)
            if not df.empty:
                df.to_parquet(fpath, index=False)
                print(f"  {INTERVAL_NAMES[interval]}: saved {len(df)} candles")
            else:
                print(f"  {INTERVAL_NAMES[interval]}: no data received")

        fr_path = os.path.join(symbol_dir, "funding_rate.parquet")
        if not os.path.exists(fr_path):
            df_fr = fetch_bybit_funding(symbol, start_ms, end_ms)
            if not df_fr.empty:
                df_fr.to_parquet(fr_path, index=False)
                print(f"  Funding rate: saved {len(df_fr)} records")
            else:
                print(f"  Funding rate: no data")
        else:
            print(f"  Funding rate: already exists, skipping")

        oi_path = os.path.join(symbol_dir, "open_interest.parquet")
        if not os.path.exists(oi_path):
            df_oi = fetch_bybit_oi(symbol, "1d", start_ms, end_ms)
            if not df_oi.empty:
                df_oi.to_parquet(oi_path, index=False)
                print(f"  Open interest: saved {len(df_oi)} records")
            else:
                print(f"  Open interest: no data")
        else:
            print(f"  Open interest: already exists, skipping")

    print(f"\nAll Bybit data downloaded to {BYBIT_DATA_DIR}")
    return BYBIT_DATA_DIR


if __name__ == "__main__":
    download_bybit_3months()
