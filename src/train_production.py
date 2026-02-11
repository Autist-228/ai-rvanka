import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import (
    DATA_DIR, MODELS_DIR, SYMBOLS,
    OPTUNA_TRIALS, WALKFORWARD_FOLDS,
)
from src.feature_engine import compute_time_weights, get_feature_cols

PROD_SUFFIX = "_prod"


def walkforward_split(df, n_folds=WALKFORWARD_FOLDS):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    min_train = int(n * 0.4)
    fold_size = (n - min_train) // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * fold_size
        val_end = min(train_end + fold_size, n)
        if val_end <= train_end:
            break
        splits.append((
            df.iloc[:train_end].copy(),
            df.iloc[train_end:val_end].copy(),
        ))
    return splits


def optuna_lgb_objective(trial, X_train, y_train, w_train, X_val, y_val):
    from sklearn.metrics import accuracy_score
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 95),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "class_weight": "balanced",
    }
    model = lgb.LGBMClassifier(**params)
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=eval_set,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)


def optuna_xgb_objective(trial, X_train, y_train, w_train, X_val, y_val):
    from sklearn.metrics import accuracy_score
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "early_stopping_rounds": 20,
    }
    model = xgb.XGBClassifier(**params)
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=eval_set,
        verbose=False,
    )
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)


def tune_lgb(all_df, feature_cols, symbol):
    print(f"    Optuna LightGBM tuning ({OPTUNA_TRIALS} trials, {WALKFORWARD_FOLDS} folds)...")
    splits = walkforward_split(all_df, WALKFORWARD_FOLDS)
    split_data = []
    for train_fold, val_fold in splits:
        X_tr = train_fold[feature_cols].values
        y_tr = train_fold["label"].values.copy()
        y_tr[y_tr == -1] = 2
        w_tr = compute_time_weights(train_fold)
        X_val = val_fold[feature_cols].values
        y_val = val_fold["label"].values.copy()
        y_val[y_val == -1] = 2
        split_data.append((X_tr, y_tr, w_tr, X_val, y_val))

    trial_count = [0]
    def objective(trial):
        scores = []
        for X_tr, y_tr, w_tr, X_val, y_val in split_data:
            score = optuna_lgb_objective(trial, X_tr, y_tr, w_tr, X_val, y_val)
            scores.append(score)
        trial_count[0] += 1
        avg = np.mean(scores)
        if trial_count[0] % 5 == 0 or trial_count[0] == 1:
            print(f"      Trial {trial_count[0]}/{OPTUNA_TRIALS}: {avg:.4f}", flush=True)
        return avg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1
    best_params["class_weight"] = "balanced"
    print(f"    Best LGB accuracy (CV): {study.best_value:.4f}")
    return best_params, study.best_value


def tune_xgb(all_df, feature_cols, symbol):
    print(f"    Optuna XGBoost tuning ({OPTUNA_TRIALS} trials, {WALKFORWARD_FOLDS} folds)...")
    splits = walkforward_split(all_df, WALKFORWARD_FOLDS)
    split_data = []
    for train_fold, val_fold in splits:
        X_tr = train_fold[feature_cols].values
        y_tr = train_fold["label"].values.copy()
        y_tr[y_tr == -1] = 2
        w_tr = compute_time_weights(train_fold)
        X_val = val_fold[feature_cols].values
        y_val = val_fold["label"].values.copy()
        y_val[y_val == -1] = 2
        split_data.append((X_tr, y_tr, w_tr, X_val, y_val))

    trial_count = [0]
    def objective(trial):
        scores = []
        for X_tr, y_tr, w_tr, X_val, y_val in split_data:
            score = optuna_xgb_objective(trial, X_tr, y_tr, w_tr, X_val, y_val)
            scores.append(score)
        trial_count[0] += 1
        avg = np.mean(scores)
        if trial_count[0] % 5 == 0 or trial_count[0] == 1:
            print(f"      Trial {trial_count[0]}/{OPTUNA_TRIALS}: {avg:.4f}", flush=True)
        return avg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "mlogloss"
    print(f"    Best XGB accuracy (CV): {study.best_value:.4f}")
    return best_params, study.best_value


def train_production_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    start_time = time.time()
    results = {}

    print("=" * 70)
    print("PRODUCTION TRAINING - ALL 2 YEARS DATA (NO HOLDOUT)")
    print("=" * 70)
    print(f"Symbols: {SYMBOLS}")
    print(f"Optuna trials: {OPTUNA_TRIALS}")
    print(f"Walk-Forward folds: {WALKFORWARD_FOLDS}")
    print(f"Models will be saved as *{PROD_SUFFIX}.pkl")
    print()

    for symbol in SYMBOLS:
        sym_start = time.time()
        print(f"\n{'='*60}")
        print(f"Training PRODUCTION models for {symbol}")
        print(f"{'='*60}")

        lgb_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm{PROD_SUFFIX}.pkl")
        xgb_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost{PROD_SUFFIX}.pkl")
        fc_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols{PROD_SUFFIX}.pkl")

        feat_path = os.path.join(DATA_DIR, symbol, "features_v2.parquet")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(DATA_DIR, symbol, "features.parquet")
        if not os.path.exists(feat_path):
            print(f"  No features for {symbol}, skipping")
            continue

        df = pd.read_parquet(feat_path)
        feature_cols = get_feature_cols(df)

        valid_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)
        df = df.dropna(subset=["label"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"  Rows: {valid_before} -> {len(df)} (ALL used for training)")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        lgb_cv_acc = 0
        xgb_cv_acc = 0

        if os.path.exists(lgb_path):
            print(f"\n  LightGBM PROD already trained, loading...")
            lgb_model = joblib.load(lgb_path)
        else:
            print(f"\n  Training LightGBM on ALL {len(df)} rows...")
            lgb_params, lgb_cv_acc = tune_lgb(df, feature_cols, symbol)

            X_all = df[feature_cols].values
            y_all = df["label"].values.copy()
            y_all[y_all == -1] = 2
            w_all = compute_time_weights(df)

            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(X_all, y_all, sample_weight=w_all)
            joblib.dump(lgb_model, lgb_path)
            print(f"    Saved: {lgb_path}")

        if os.path.exists(xgb_path):
            print(f"\n  XGBoost PROD already trained, loading...")
            xgb_model = joblib.load(xgb_path)
        else:
            print(f"\n  Training XGBoost on ALL {len(df)} rows...")
            xgb_params, xgb_cv_acc = tune_xgb(df, feature_cols, symbol)

            X_all = df[feature_cols].values
            y_all = df["label"].values.copy()
            y_all[y_all == -1] = 2
            w_all = compute_time_weights(df)

            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_all, y_all, sample_weight=w_all)
            joblib.dump(xgb_model, xgb_path)
            print(f"    Saved: {xgb_path}")

        joblib.dump(feature_cols, fc_path)

        params_path = os.path.join(MODELS_DIR, f"{symbol}_best_params{PROD_SUFFIX}.pkl")
        joblib.dump({
            "lgb_cv_acc": lgb_cv_acc,
            "xgb_cv_acc": xgb_cv_acc,
        }, params_path)

        X_sample = df[feature_cols].values[-min(1000, len(df)):]
        try:
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)
            importance = pd.DataFrame({
                "feature": feature_cols,
                "shap_importance": mean_abs,
            }).sort_values("shap_importance", ascending=False)
            shap_path = os.path.join(MODELS_DIR, f"{symbol}_shap_lgb{PROD_SUFFIX}.csv")
            importance.to_csv(shap_path, index=False)
        except Exception as e:
            print(f"    SHAP failed: {e}")

        sym_elapsed = time.time() - sym_start
        results[symbol] = {
            "lgb_cv_acc": lgb_cv_acc,
            "xgb_cv_acc": xgb_cv_acc,
            "rows": len(df),
            "features": len(feature_cols),
            "time_sec": round(sym_elapsed, 1),
        }
        print(f"\n  {symbol} DONE: LGB_CV={lgb_cv_acc:.4f}, XGB_CV={xgb_cv_acc:.4f} ({sym_elapsed:.0f}s)")

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = int(total_time % 60)

    print(f"\n{'='*70}")
    print(f"PRODUCTION TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {hours}h {mins}m {secs}s")
    print(f"\nResults:")
    print(f"{'Symbol':<12} {'Rows':>8} {'Features':>10} {'LGB_CV':>10} {'XGB_CV':>10} {'Time':>8}")
    print("-" * 60)
    for symbol, r in results.items():
        print(f"{symbol:<12} {r['rows']:>8} {r['features']:>10} {r['lgb_cv_acc']:>10.4f} {r['xgb_cv_acc']:>10.4f} {r['time_sec']:>7.0f}s")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PRODUCTION MODELS - TRAINED ON ALL 2 YEARS")
    report_lines.append("=" * 70)
    report_lines.append(f"Date: {pd.Timestamp.now()}")
    report_lines.append(f"Training time: {hours}h {mins}m {secs}s")
    report_lines.append(f"")
    report_lines.append(f"{'Symbol':<12} {'Rows':>8} {'Features':>10} {'LGB_CV':>10} {'XGB_CV':>10}")
    report_lines.append("-" * 60)
    for symbol, r in results.items():
        report_lines.append(f"{symbol:<12} {r['rows']:>8} {r['features']:>10} {r['lgb_cv_acc']:>10.4f} {r['xgb_cv_acc']:>10.4f}")

    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "production_training_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")

    return results


if __name__ == "__main__":
    train_production_models()
