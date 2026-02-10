import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
]

INTERVALS = ["1", "5", "15", "60", "240"]
INTERVAL_NAMES = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "240": "4h",
}

END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=730)

BACKTEST_MONTHS = 3

BYBIT_BASE_URL = "https://api.bybit.com"
BYBIT_KLINE_LIMIT = 200
BYBIT_RATE_LIMIT_DELAY = 0.05

PRIMARY_INTERVAL = "60"

RISK_PROFILES = {
    "conservative": {
        "max_leverage": 3,
        "max_position_pct": 0.05,
        "max_daily_loss_pct": 0.03,
        "max_concurrent_positions": 2,
        "base_tp_pct": 0.02,
        "base_sl_pct": 0.01,
    },
    "moderate": {
        "max_leverage": 10,
        "max_position_pct": 0.15,
        "max_daily_loss_pct": 0.07,
        "max_concurrent_positions": 4,
        "base_tp_pct": 0.05,
        "base_sl_pct": 0.02,
    },
    "aggressive": {
        "max_leverage": 25,
        "max_position_pct": 0.30,
        "max_daily_loss_pct": 0.15,
        "max_concurrent_positions": 6,
        "base_tp_pct": 0.12,
        "base_sl_pct": 0.05,
    },
}

INITIAL_BALANCE = 500.0

CONFIDENCE_THRESHOLD = 0.52
