import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import DATA_DIR, MODELS_DIR, SYMBOLS, BACKTEST_MONTHS
from src.feature_engine import compute_time_weights


EXCLUDE_COLS = {
    "timestamp", "datetime", "open", "high", "low", "close", "volume", "turnover",
    "future_return", "future_high", "future_low", "future_max_up", "future_max_down",
    "label", "hour", "day_of_week", "symbol"
}


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def split_train_test(df: pd.DataFrame, backtest_months: int = BACKTEST_MONTHS):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["label"])

    max_ts = df["datetime"].max()
    cutoff = max_ts - pd.DateOffset(months=backtest_months)

    train = df[df["datetime"] < cutoff].copy()
    test = df[df["datetime"] >= cutoff].copy()

    return train, test


def train_lightgbm(train_df: pd.DataFrame, feature_cols: list) -> lgb.LGBMClassifier:
    X = train_df[feature_cols].values
    y = train_df["label"].values

    y_mapped = y.copy()
    y_mapped[y_mapped == -1] = 2

    weights = compute_time_weights(train_df)

    model = lgb.LGBMClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )

    model.fit(X, y_mapped, sample_weight=weights)
    return model


def train_xgboost(train_df: pd.DataFrame, feature_cols: list) -> xgb.XGBClassifier:
    X = train_df[feature_cols].values
    y = train_df["label"].values

    y_mapped = y.copy()
    y_mapped[y_mapped == -1] = 2

    weights = compute_time_weights(train_df)

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    model.fit(X, y_mapped, sample_weight=weights)
    return model


def evaluate_model(model, test_df: pd.DataFrame, feature_cols: list, model_name: str, symbol: str):
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    y_test_mapped = y_test.copy()
    y_test_mapped[y_test_mapped == -1] = 2

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test_mapped, y_pred)
    print(f"\n  {model_name} on {symbol}:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Class distribution (test): {pd.Series(y_test_mapped).value_counts().to_dict()}")
    print(f"    Predictions distribution: {pd.Series(y_pred).value_counts().to_dict()}")

    return {
        "model_name": model_name,
        "symbol": symbol,
        "accuracy": acc,
        "predictions": y_pred,
        "probabilities": y_proba,
        "true_labels": y_test_mapped,
    }


def train_all_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Training models for {symbol}")
        print(f"{'='*60}")

        feat_path = os.path.join(DATA_DIR, symbol, "features.parquet")
        if not os.path.exists(feat_path):
            print(f"  No features found for {symbol}, skipping")
            continue

        df = pd.read_parquet(feat_path)
        feature_cols = get_feature_cols(df)

        valid_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)
        df = df.dropna(subset=["label"])
        print(f"  Rows: {valid_before} -> {len(df)} after cleaning")
        print(f"  Features: {len(feature_cols)}")

        train_df, test_df = split_train_test(df)
        print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        print(f"  Train period: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
        print(f"  Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")

        print(f"\n  Training LightGBM for {symbol}...")
        lgb_model = train_lightgbm(train_df, feature_cols)
        lgb_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm.pkl")
        joblib.dump(lgb_model, lgb_path)
        lgb_eval = evaluate_model(lgb_model, test_df, feature_cols, "LightGBM", symbol)

        print(f"\n  Training XGBoost for {symbol}...")
        xgb_model = train_xgboost(train_df, feature_cols)
        xgb_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost.pkl")
        joblib.dump(xgb_model, xgb_path)
        xgb_eval = evaluate_model(xgb_model, test_df, feature_cols, "XGBoost", symbol)

        test_path = os.path.join(DATA_DIR, symbol, "test_data.parquet")
        test_df.to_parquet(test_path, index=False)

        feature_cols_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols.pkl")
        joblib.dump(feature_cols, feature_cols_path)

        results[symbol] = {
            "lightgbm": lgb_eval,
            "xgboost": xgb_eval,
        }

    return results


if __name__ == "__main__":
    train_all_models()
