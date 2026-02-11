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

ALL_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
PRIMARY_TF = "15m"

END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=730)

BACKTEST_MONTHS = 3

INITIAL_BALANCE = 500.0
CONFIDENCE_THRESHOLD = 0.55

DYNAMIC_RISK = {
    "max_leverage": 20,
    "min_leverage": 1,
    "max_position_pct": 0.25,
    "min_position_pct": 0.03,
    "max_daily_loss_pct": 0.10,
    "max_concurrent_positions": 3,
    "base_tp_pct": 0.04,
    "base_sl_pct": 0.02,
}

OPTUNA_TRIALS = 25
WALKFORWARD_FOLDS = 5
LABEL_HORIZON = 4

HIGHER_TF_FEATURES = [
    "rsi_14", "macd_diff", "bb_pct", "bb_width", "adx",
    "atr_14", "volume_ratio_10", "return_1", "return_5",
    "stoch_k", "cci_14", "cmf",
]

LOWER_TF_FEATURES = [
    "rsi_14", "macd_diff", "atr_14", "volume_ratio_10",
    "return_1", "return_5", "volatility_5", "stoch_k",
]

TIME_WEIGHT_DECAY = 0.001
