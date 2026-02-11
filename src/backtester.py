import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, SYMBOLS,
    DYNAMIC_RISK, INITIAL_BALANCE, CONFIDENCE_THRESHOLD,
)


class Position:
    def __init__(self, direction, entry_price, size_usd, leverage, tp_price, sl_price, entry_time):
        self.direction = direction
        self.entry_price = entry_price
        self.size_usd = size_usd
        self.leverage = leverage
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.exit_reason = None

    def check_exit(self, high, low, close, current_time):
        if self.direction == 1:
            if high >= self.tp_price:
                self.exit_price = self.tp_price
                self.exit_reason = "TP"
            elif low <= self.sl_price:
                self.exit_price = self.sl_price
                self.exit_reason = "SL"
        else:
            if low <= self.tp_price:
                self.exit_price = self.tp_price
                self.exit_reason = "TP"
            elif high >= self.sl_price:
                self.exit_price = self.sl_price
                self.exit_reason = "SL"

        if self.exit_price is not None:
            self.exit_time = current_time
            if self.direction == 1:
                price_change = (self.exit_price - self.entry_price) / self.entry_price
            else:
                price_change = (self.entry_price - self.exit_price) / self.entry_price

            self.pnl = self.size_usd * self.leverage * price_change
            fee = self.size_usd * self.leverage * 0.0012
            self.pnl -= fee
            self.pnl_pct = self.pnl / self.size_usd
            return True
        return False


def decode_prediction(pred_class, probabilities):
    label_map = {0: 0, 1: 1, 2: -1}
    direction = label_map.get(pred_class, 0)
    confidence = probabilities[pred_class]
    return direction, confidence


def compute_dynamic_leverage(confidence, volatility, risk=DYNAMIC_RISK):
    conf_ratio = (confidence - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD)
    conf_ratio = min(max(conf_ratio, 0.0), 1.0)

    if volatility is not None and volatility > 0:
        vol_factor = min(1.0, 0.02 / (volatility + 1e-10))
    else:
        vol_factor = 0.5

    combined = conf_ratio * 0.7 + vol_factor * 0.3

    leverage = risk["min_leverage"] + combined * (risk["max_leverage"] - risk["min_leverage"])
    leverage = max(risk["min_leverage"], min(risk["max_leverage"], int(leverage)))

    position_pct = risk["min_position_pct"] + combined * (risk["max_position_pct"] - risk["min_position_pct"])
    position_pct = max(risk["min_position_pct"], min(risk["max_position_pct"], position_pct))

    return leverage, position_pct


def run_backtest_ensemble(symbol: str):
    lgb_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm_v2.pkl")
    xgb_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost_v2.pkl")
    features_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols_v2.pkl")
    test_path = os.path.join(DATA_DIR, symbol, "test_data_v2.parquet")

    if not all(os.path.exists(p) for p in [lgb_path, xgb_path, features_path, test_path]):
        return None

    lgb_model = joblib.load(lgb_path)
    xgb_model = joblib.load(xgb_path)
    feature_cols = joblib.load(features_path)
    test_df = pd.read_parquet(test_path)

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    risk = DYNAMIC_RISK
    balance = INITIAL_BALANCE
    positions = []
    closed_trades = []
    equity_curve = []
    daily_pnl = {}
    signals_log = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_time = row["datetime"]
        current_date = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]

        for pos in positions[:]:
            if pos.check_exit(row["high"], row["low"], row["close"], current_time):
                balance += pos.pnl
                closed_trades.append(pos)
                positions.remove(pos)
                if current_date not in daily_pnl:
                    daily_pnl[current_date] = 0
                daily_pnl[current_date] += pos.pnl

        equity_curve.append({"timestamp": row["timestamp"], "balance": balance})

        if balance <= 0:
            break

        today_loss = daily_pnl.get(current_date, 0)
        if today_loss < -INITIAL_BALANCE * risk["max_daily_loss_pct"]:
            continue

        if len(positions) >= risk["max_concurrent_positions"]:
            continue

        X = row[feature_cols].values.reshape(1, -1)

        lgb_pred = lgb_model.predict(X)[0]
        lgb_proba = lgb_model.predict_proba(X)[0]
        lgb_dir, lgb_conf = decode_prediction(lgb_pred, lgb_proba)

        xgb_pred = xgb_model.predict(X)[0]
        xgb_proba = xgb_model.predict_proba(X)[0]
        xgb_dir, xgb_conf = decode_prediction(xgb_pred, xgb_proba)

        if lgb_dir != xgb_dir:
            continue
        if lgb_dir == 0:
            continue

        direction = lgb_dir
        confidence = (lgb_conf + xgb_conf) / 2.0

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        volatility = row.get("volatility_20", None)
        leverage, position_pct = compute_dynamic_leverage(confidence, volatility)

        size_usd = balance * position_pct
        if size_usd < 1:
            continue

        entry_price = row["close"]
        atr = row.get("atr_14", 0)
        if atr and atr > 0:
            atr_pct = atr / entry_price
            tp_mult = max(risk["base_tp_pct"], atr_pct * 2.5)
            sl_mult = max(risk["base_sl_pct"], atr_pct * 1.2)
        else:
            tp_mult = risk["base_tp_pct"]
            sl_mult = risk["base_sl_pct"]

        if direction == 1:
            tp_price = entry_price * (1 + tp_mult)
            sl_price = entry_price * (1 - sl_mult)
        else:
            tp_price = entry_price * (1 - tp_mult)
            sl_price = entry_price * (1 + sl_mult)

        pos = Position(direction, entry_price, size_usd, leverage, tp_price, sl_price, current_time)
        positions.append(pos)

        signals_log.append({
            "time": current_time,
            "direction": direction,
            "lgb_conf": lgb_conf,
            "xgb_conf": xgb_conf,
            "avg_conf": confidence,
            "leverage": leverage,
            "size_usd": size_usd,
        })

    for pos in positions:
        last_row = test_df.iloc[-1]
        pos.exit_price = last_row["close"]
        pos.exit_time = last_row["datetime"]
        if pos.direction == 1:
            price_change = (pos.exit_price - pos.entry_price) / pos.entry_price
        else:
            price_change = (pos.entry_price - pos.exit_price) / pos.entry_price
        pos.pnl = pos.size_usd * pos.leverage * price_change
        fee = pos.size_usd * pos.leverage * 0.0012
        pos.pnl -= fee
        pos.pnl_pct = pos.pnl / pos.size_usd
        pos.exit_reason = "CLOSE"
        balance += pos.pnl
        closed_trades.append(pos)

    stats = compute_stats(closed_trades, equity_curve, symbol)
    stats["signals_count"] = len(signals_log)
    return stats


def compute_stats(trades, equity_curve, symbol):
    if not trades:
        return {
            "symbol": symbol,
            "strategy": "Ensemble",
            "total_trades": 0,
            "final_balance": INITIAL_BALANCE,
            "total_profit_usd": 0,
            "total_profit_pct": 0,
            "win_rate": 0,
            "avg_win_usd": 0,
            "avg_loss_usd": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "best_trade_usd": 0,
            "worst_trade_usd": 0,
            "avg_leverage": 0,
            "avg_holding_hours": 0,
            "tp_count": 0,
            "sl_count": 0,
            "profit_factor": 0,
            "signals_count": 0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    final_balance = INITIAL_BALANCE + sum(pnls)
    total_profit = sum(pnls)

    eq = pd.DataFrame(equity_curve)
    if not eq.empty:
        eq["peak"] = eq["balance"].cummax()
        eq["drawdown"] = (eq["balance"] - eq["peak"]) / eq["peak"]
        max_dd = eq["drawdown"].min()
    else:
        max_dd = 0

    returns = pd.Series(pnls) / INITIAL_BALANCE
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)
    else:
        sharpe = 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss

    avg_leverage = np.mean([t.leverage for t in trades])

    holding_hours = []
    for t in trades:
        try:
            diff = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600
            holding_hours.append(diff)
        except Exception:
            pass

    return {
        "symbol": symbol,
        "strategy": "Ensemble",
        "total_trades": len(trades),
        "final_balance": round(final_balance, 2),
        "total_profit_usd": round(total_profit, 2),
        "total_profit_pct": round((total_profit / INITIAL_BALANCE) * 100, 2),
        "win_rate": round((len(wins) / len(trades)) * 100, 2) if trades else 0,
        "avg_win_usd": round(np.mean(wins), 2) if wins else 0,
        "avg_loss_usd": round(np.mean(losses), 2) if losses else 0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "best_trade_usd": round(max(pnls), 2) if pnls else 0,
        "worst_trade_usd": round(min(pnls), 2) if pnls else 0,
        "avg_leverage": round(avg_leverage, 1),
        "avg_holding_hours": round(np.mean(holding_hours), 1) if holding_hours else 0,
        "tp_count": sum(1 for t in trades if t.exit_reason == "TP"),
        "sl_count": sum(1 for t in trades if t.exit_reason == "SL"),
        "profit_factor": round(profit_factor, 2),
    }


def run_backtest_single(symbol: str, model_type: str):
    if model_type == "lightgbm":
        model_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm_v2.pkl")
    else:
        model_path = os.path.join(MODELS_DIR, f"{symbol}_xgboost_v2.pkl")
    features_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols_v2.pkl")
    test_path = os.path.join(DATA_DIR, symbol, "test_data_v2.parquet")

    if not all(os.path.exists(p) for p in [model_path, features_path, test_path]):
        return None

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    test_df = pd.read_parquet(test_path)

    test_df = test_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    risk = DYNAMIC_RISK
    balance = INITIAL_BALANCE
    positions = []
    closed_trades = []
    equity_curve = []
    daily_pnl = {}
    signals_log = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_time = row["datetime"]
        current_date = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]

        for pos in positions[:]:
            if pos.check_exit(row["high"], row["low"], row["close"], current_time):
                balance += pos.pnl
                closed_trades.append(pos)
                positions.remove(pos)
                if current_date not in daily_pnl:
                    daily_pnl[current_date] = 0
                daily_pnl[current_date] += pos.pnl

        equity_curve.append({"timestamp": row["timestamp"], "balance": balance})

        if balance <= 0:
            break

        today_loss = daily_pnl.get(current_date, 0)
        if today_loss < -INITIAL_BALANCE * risk["max_daily_loss_pct"]:
            continue

        if len(positions) >= risk["max_concurrent_positions"]:
            continue

        X = row[feature_cols].values.reshape(1, -1)

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        direction, confidence = decode_prediction(pred, proba)

        if direction == 0:
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        volatility = row.get("volatility_20", None)
        leverage, position_pct = compute_dynamic_leverage(confidence, volatility)

        size_usd = balance * position_pct
        if size_usd < 1:
            continue

        entry_price = row["close"]
        atr = row.get("atr_14", 0)
        if atr and atr > 0:
            atr_pct = atr / entry_price
            tp_mult = max(risk["base_tp_pct"], atr_pct * 2.5)
            sl_mult = max(risk["base_sl_pct"], atr_pct * 1.2)
        else:
            tp_mult = risk["base_tp_pct"]
            sl_mult = risk["base_sl_pct"]

        if direction == 1:
            tp_price = entry_price * (1 + tp_mult)
            sl_price = entry_price * (1 - sl_mult)
        else:
            tp_price = entry_price * (1 - tp_mult)
            sl_price = entry_price * (1 + sl_mult)

        pos = Position(direction, entry_price, size_usd, leverage, tp_price, sl_price, current_time)
        positions.append(pos)

        signals_log.append({
            "time": current_time,
            "direction": direction,
            "confidence": confidence,
            "leverage": leverage,
            "size_usd": size_usd,
        })

    for pos in positions:
        last_row = test_df.iloc[-1]
        pos.exit_price = last_row["close"]
        pos.exit_time = last_row["datetime"]
        if pos.direction == 1:
            price_change = (pos.exit_price - pos.entry_price) / pos.entry_price
        else:
            price_change = (pos.entry_price - pos.exit_price) / pos.entry_price
        pos.pnl = pos.size_usd * pos.leverage * price_change
        fee = pos.size_usd * pos.leverage * 0.0012
        pos.pnl -= fee
        pos.pnl_pct = pos.pnl / pos.size_usd
        pos.exit_reason = "CLOSE"
        balance += pos.pnl
        closed_trades.append(pos)

    stats = compute_stats(closed_trades, equity_curve, symbol)
    stats["strategy"] = model_type.upper()
    stats["signals_count"] = len(signals_log)
    return stats


def run_all_backtests():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for symbol in SYMBOLS:
        for model_type in ["lightgbm", "xgboost"]:
            label = "LGB" if model_type == "lightgbm" else "XGB"
            print(f"Backtesting {symbol} | {label} solo | Dynamic Risk...")
            stats = run_backtest_single(symbol, model_type)
            if stats:
                all_results.append(stats)
                print(f"  -> Trades: {stats['total_trades']}, "
                      f"Profit: ${stats['total_profit_usd']} ({stats['total_profit_pct']}%), "
                      f"Win Rate: {stats['win_rate']}%")

        print(f"Backtesting {symbol} | Ensemble (LGB+XGB voting) | Dynamic Risk...")
        stats = run_backtest_ensemble(symbol)
        if stats:
            all_results.append(stats)
            print(f"  -> Trades: {stats['total_trades']}, "
                  f"Profit: ${stats['total_profit_usd']} ({stats['total_profit_pct']}%), "
                  f"Win Rate: {stats['win_rate']}%")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("total_profit_pct", ascending=False)

    csv_path = os.path.join(RESULTS_DIR, "backtest_results_v2.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return results_df


if __name__ == "__main__":
    results = run_all_backtests()
    print("\n" + "=" * 80)
    print("ALL 18 BACKTEST RESULTS (sorted by profit %)")
    print("=" * 80)
    print(results.to_string(index=False))
