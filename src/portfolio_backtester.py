import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import DATA_DIR, MODELS_DIR, RESULTS_DIR, SYMBOLS, CONFIDENCE_THRESHOLD
from src.portfolio_manager import MetaAllocator, RiskController, PORTFOLIO_CONFIG


class PortfolioPosition:
    def __init__(self, symbol, direction, entry_price, size_usd, leverage,
                 tp_price, sl_price, entry_time, trailing_activation, trailing_step):
        self.symbol = symbol
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
        self.exit_reason = None
        self.trailing_activated = False
        self.trailing_activation = trailing_activation
        self.trailing_step = trailing_step
        self.highest_price = entry_price if direction == 1 else entry_price
        self.lowest_price = entry_price if direction == -1 else entry_price

    def update_trailing(self, high, low):
        if self.direction == 1:
            if high > self.highest_price:
                self.highest_price = high
            unrealized_pct = (self.highest_price - self.entry_price) / self.entry_price
            if unrealized_pct >= self.trailing_activation:
                self.trailing_activated = True
                new_sl = self.highest_price * (1 - self.trailing_step)
                if new_sl > self.sl_price:
                    self.sl_price = new_sl
        else:
            if low < self.lowest_price:
                self.lowest_price = low
            unrealized_pct = (self.entry_price - self.lowest_price) / self.entry_price
            if unrealized_pct >= self.trailing_activation:
                self.trailing_activated = True
                new_sl = self.lowest_price * (1 + self.trailing_step)
                if new_sl < self.sl_price:
                    self.sl_price = new_sl

    def check_exit(self, high, low, close, current_time):
        self.update_trailing(high, low)

        if self.direction == 1:
            if high >= self.tp_price:
                self.exit_price = self.tp_price
                self.exit_reason = "TP"
            elif low <= self.sl_price:
                self.exit_price = self.sl_price
                self.exit_reason = "TRAILING_SL" if self.trailing_activated else "SL"
        else:
            if low <= self.tp_price:
                self.exit_price = self.tp_price
                self.exit_reason = "TP"
            elif high >= self.sl_price:
                self.exit_price = self.sl_price
                self.exit_reason = "TRAILING_SL" if self.trailing_activated else "SL"

        if self.exit_price is not None:
            self.exit_time = current_time
            if self.direction == 1:
                price_change = (self.exit_price - self.entry_price) / self.entry_price
            else:
                price_change = (self.entry_price - self.exit_price) / self.entry_price

            fee_pct = PORTFOLIO_CONFIG["trading"]["fee_pct"]
            self.pnl = self.size_usd * self.leverage * price_change
            self.pnl -= self.size_usd * self.leverage * fee_pct
            return True
        return False


def decode_prediction(pred_class, probabilities):
    label_map = {0: 0, 1: 1, 2: -1}
    direction = label_map.get(pred_class, 0)
    confidence = probabilities[pred_class]
    return direction, confidence


def compute_dynamic_leverage(confidence, volatility, leverage_mult=1.0):
    trading = PORTFOLIO_CONFIG["trading"]
    conf_ratio = (confidence - trading["confidence_threshold"]) / (1.0 - trading["confidence_threshold"])
    conf_ratio = min(max(conf_ratio, 0.0), 1.0)

    if volatility is not None and volatility > 0:
        vol_factor = min(1.0, 0.02 / (volatility + 1e-10))
    else:
        vol_factor = 0.5

    combined = conf_ratio * 0.7 + vol_factor * 0.3
    max_lev = trading["max_leverage"]
    min_lev = trading["min_leverage"]

    leverage = min_lev + combined * (max_lev - min_lev)
    leverage = leverage * leverage_mult
    leverage = max(min_lev, min(max_lev, int(leverage)))

    return leverage


def load_all_data():
    models = {}
    test_data = {}
    feature_cols = {}

    for symbol in SYMBOLS:
        lgb_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm_v2.pkl")
        fc_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols_v2.pkl")
        test_path = os.path.join(DATA_DIR, symbol, "test_data_v2.parquet")

        if not all(os.path.exists(p) for p in [lgb_path, fc_path, test_path]):
            print(f"  Skipping {symbol} — missing files")
            continue

        models[symbol] = joblib.load(lgb_path)
        feature_cols[symbol] = joblib.load(fc_path)
        df = pd.read_parquet(test_path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        df[feature_cols[symbol]] = df[feature_cols[symbol]].fillna(0)
        test_data[symbol] = df

    return models, test_data, feature_cols


def run_portfolio_backtest():
    print("=" * 80)
    print("PORTFOLIO BACKTEST — LGB + Meta-Allocator + Risk Controller")
    print("=" * 80)

    models, test_data, feature_cols = load_all_data()

    config = PORTFOLIO_CONFIG
    initial_balance = config["initial_balance"]
    trading = config["trading"]

    risk_ctrl = RiskController(config["risk_controller"])
    risk_ctrl.peak_balance = initial_balance
    allocator = MetaAllocator(config["allocator"])

    balance = initial_balance
    positions = []
    closed_trades = []
    equity_curve = []
    allocation_log = []
    daily_pnl = {}
    coin_stats = {s: {"trades": 0, "wins": 0, "pnl": 0, "allocated_usd": 0} for s in SYMBOLS}

    min_len = min(len(df) for df in test_data.values())
    print(f"\nTest period: {min_len} candles (~{min_len * 15 / 60 / 24:.0f} days)")
    print(f"Initial balance: ${initial_balance}")
    print(f"Coins: {list(test_data.keys())}")
    print()

    for i in range(min_len):
        if i % 1000 == 0 and i > 0:
            print(f"  Candle {i}/{min_len} | Balance: ${balance:.2f} | "
                  f"Positions: {len(positions)} | Trades: {len(closed_trades)}")

        sample_symbol = list(test_data.keys())[0]
        current_time = test_data[sample_symbol].iloc[i]["datetime"]
        current_date = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]

        for pos in positions[:]:
            row = test_data[pos.symbol].iloc[i]
            if pos.check_exit(row["high"], row["low"], row["close"], current_time):
                balance += pos.pnl
                closed_trades.append(pos)
                positions.remove(pos)

                risk_ctrl.record_trade(pos.symbol, pos.pnl, i)
                risk_ctrl.update_balance(balance, initial_balance)

                coin_stats[pos.symbol]["trades"] += 1
                coin_stats[pos.symbol]["pnl"] += pos.pnl
                if pos.pnl > 0:
                    coin_stats[pos.symbol]["wins"] += 1

                if current_date not in daily_pnl:
                    daily_pnl[current_date] = 0
                daily_pnl[current_date] += pos.pnl

        equity_curve.append({"timestamp": test_data[sample_symbol].iloc[i]["timestamp"], "balance": balance})

        if balance <= 0:
            print(f"  LIQUIDATED at candle {i}")
            break

        today_loss = daily_pnl.get(current_date, 0)
        if today_loss < -initial_balance * 0.10:
            continue

        active_coin_positions = {}
        for pos in positions:
            active_coin_positions[pos.symbol] = active_coin_positions.get(pos.symbol, 0) + 1

        if len(positions) >= trading["max_concurrent_total"]:
            continue

        signals = {}
        for symbol in test_data:
            if active_coin_positions.get(symbol, 0) >= trading["max_concurrent_per_coin"]:
                continue

            row = test_data[symbol].iloc[i]
            X = row[feature_cols[symbol]].values.reshape(1, -1)

            pred = models[symbol].predict(X)[0]
            proba = models[symbol].predict_proba(X)[0]
            direction, confidence = decode_prediction(pred, proba)

            regime = "normal"
            adx_val = row.get("adx", 25)
            vol_val = row.get("volatility_20", 0.02)
            if isinstance(adx_val, (int, float)) and isinstance(vol_val, (int, float)):
                if adx_val < 15 and vol_val > 0.04:
                    regime = "crash"
                elif adx_val > 30:
                    regime = "trend"

            signals[symbol] = {
                "direction": direction,
                "confidence": confidence,
                "threshold": trading["confidence_threshold"],
                "candle_idx": i,
                "regime": regime,
                "volatility": row.get("volatility_20", None),
                "atr": row.get("atr_14", 0),
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
            }

        allocations = allocator.allocate(signals, risk_ctrl)

        if allocations and i % 500 == 0:
            allocation_log.append({
                "candle": i,
                "time": current_time,
                "balance": balance,
                "allocations": dict(allocations),
            })

        leverage_mult = risk_ctrl.get_leverage_multiplier(balance, initial_balance)

        for symbol, alloc_pct in allocations.items():
            sig = signals[symbol]
            if sig["direction"] == 0:
                continue

            max_pos_usd = PORTFOLIO_CONFIG["allocator"].get("max_position_usd", 200.0)
            size_usd = min(balance * alloc_pct, max_pos_usd)
            if size_usd < 1:
                continue

            coin_stats[symbol]["allocated_usd"] += size_usd

            leverage = compute_dynamic_leverage(sig["confidence"], sig["volatility"], leverage_mult)

            entry_price = sig["close"]
            atr = sig["atr"]

            coin_tpsl = PORTFOLIO_CONFIG.get("coin_tp_sl", {}).get(symbol, {})
            base_tp = coin_tpsl.get("tp_pct", trading["base_tp_pct"])
            base_sl = coin_tpsl.get("sl_pct", trading["base_sl_pct"])
            atr_tp_m = coin_tpsl.get("atr_tp_mult", 2.5)
            atr_sl_m = coin_tpsl.get("atr_sl_mult", 1.2)

            if atr and atr > 0:
                atr_pct = atr / entry_price
                tp_mult = max(base_tp, atr_pct * atr_tp_m)
                sl_mult = max(base_sl, atr_pct * atr_sl_m)
            else:
                tp_mult = base_tp
                sl_mult = base_sl

            if sig["direction"] == 1:
                tp_price = entry_price * (1 + tp_mult)
                sl_price = entry_price * (1 - sl_mult)
            else:
                tp_price = entry_price * (1 - tp_mult)
                sl_price = entry_price * (1 + sl_mult)

            trail_config = PORTFOLIO_CONFIG["risk_controller"]
            pos = PortfolioPosition(
                symbol=symbol,
                direction=sig["direction"],
                entry_price=entry_price,
                size_usd=size_usd,
                leverage=leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_time=current_time,
                trailing_activation=trail_config["trailing_activation_pct"],
                trailing_step=trail_config["trailing_step_pct"],
            )
            positions.append(pos)

    for pos in positions:
        last_row = test_data[pos.symbol].iloc[-1]
        pos.exit_price = last_row["close"]
        pos.exit_time = last_row["datetime"]
        if pos.direction == 1:
            price_change = (pos.exit_price - pos.entry_price) / pos.entry_price
        else:
            price_change = (pos.entry_price - pos.exit_price) / pos.entry_price
        pos.pnl = pos.size_usd * pos.leverage * price_change
        pos.pnl -= pos.size_usd * pos.leverage * trading["fee_pct"]
        pos.exit_reason = "CLOSE"
        balance += pos.pnl
        closed_trades.append(pos)
        coin_stats[pos.symbol]["trades"] += 1
        coin_stats[pos.symbol]["pnl"] += pos.pnl
        if pos.pnl > 0:
            coin_stats[pos.symbol]["wins"] += 1

    stats = compute_portfolio_stats(closed_trades, equity_curve, initial_balance, coin_stats)
    stats["allocation_log"] = allocation_log

    return stats


def compute_portfolio_stats(trades, equity_curve, initial_balance, coin_stats):
    if not trades:
        return {"total_trades": 0, "final_balance": initial_balance}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    final_balance = initial_balance + sum(pnls)
    total_profit = sum(pnls)

    eq = pd.DataFrame(equity_curve)
    if not eq.empty:
        eq["peak"] = eq["balance"].cummax()
        eq["drawdown"] = (eq["balance"] - eq["peak"]) / eq["peak"]
        max_dd = eq["drawdown"].min()
    else:
        max_dd = 0

    returns = pd.Series(pnls) / initial_balance
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)
    else:
        sharpe = 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss

    avg_leverage = np.mean([t.leverage for t in trades])

    tp_count = sum(1 for t in trades if t.exit_reason == "TP")
    sl_count = sum(1 for t in trades if t.exit_reason == "SL")
    trailing_sl_count = sum(1 for t in trades if t.exit_reason == "TRAILING_SL")

    coin_breakdown = {}
    for symbol, cs in coin_stats.items():
        if cs["trades"] > 0:
            coin_breakdown[symbol] = {
                "trades": cs["trades"],
                "wins": cs["wins"],
                "win_rate": round(cs["wins"] / cs["trades"] * 100, 2),
                "pnl": round(cs["pnl"], 2),
                "allocated_usd": round(cs["allocated_usd"], 2),
            }

    return {
        "initial_balance": initial_balance,
        "final_balance": round(final_balance, 2),
        "total_profit_usd": round(total_profit, 2),
        "total_profit_pct": round((total_profit / initial_balance) * 100, 2),
        "total_trades": len(trades),
        "win_rate": round((len(wins) / len(trades)) * 100, 2),
        "avg_win_usd": round(np.mean(wins), 2) if wins else 0,
        "avg_loss_usd": round(np.mean(losses), 2) if losses else 0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_leverage": round(avg_leverage, 1),
        "tp_count": tp_count,
        "sl_count": sl_count,
        "trailing_sl_count": trailing_sl_count,
        "best_trade_usd": round(max(pnls), 2),
        "worst_trade_usd": round(min(pnls), 2),
        "coin_breakdown": coin_breakdown,
    }


def print_portfolio_report(stats):
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTEST RESULTS — LGB + Meta-Allocator + Risk Controller")
    print("=" * 80)

    print(f"\n  Initial Balance:  ${stats['initial_balance']:.2f}")
    print(f"  Final Balance:    ${stats['final_balance']:.2f}")
    print(f"  Total Profit:     ${stats['total_profit_usd']:.2f} ({stats['total_profit_pct']:.2f}%)")
    print(f"  Total Trades:     {stats['total_trades']}")
    print(f"  Win Rate:         {stats['win_rate']:.2f}%")
    print(f"  Max Drawdown:     {stats['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:     {stats['sharpe_ratio']:.2f}")
    print(f"  Profit Factor:    {stats['profit_factor']:.2f}")
    print(f"  Avg Leverage:     {stats['avg_leverage']:.1f}x")
    print(f"  Best Trade:       ${stats['best_trade_usd']:.2f}")
    print(f"  Worst Trade:      ${stats['worst_trade_usd']:.2f}")
    print(f"  TP: {stats['tp_count']} | SL: {stats['sl_count']} | Trailing SL: {stats['trailing_sl_count']}")

    print(f"\n  {'='*70}")
    print(f"  COIN BREAKDOWN:")
    print(f"  {'='*70}")
    print(f"  {'Coin':<12} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>12} {'Allocated':>12}")
    print(f"  {'-'*70}")

    for symbol, cs in sorted(stats.get("coin_breakdown", {}).items(), key=lambda x: x[1]["pnl"], reverse=True):
        print(f"  {symbol:<12} {cs['trades']:>7} {cs['wins']:>6} {cs['win_rate']:>6.1f}% ${cs['pnl']:>10.2f} ${cs['allocated_usd']:>10.2f}")

    print(f"\n  {'='*70}")
    print(f"  COMPARISON vs INDIVIDUAL BACKTESTS:")
    print(f"  {'='*70}")
    old_csv = os.path.join(RESULTS_DIR, "backtest_results_v2.csv")
    if os.path.exists(old_csv):
        old = pd.read_csv(old_csv)
        lgb_only = old[old["strategy"] == "LIGHTGBM"]
        ens_only = old[old["strategy"] == "Ensemble"]
        lgb_total = lgb_only["total_profit_usd"].sum()
        ens_total = ens_only["total_profit_usd"].sum()
        lgb_avg_wr = lgb_only["win_rate"].mean()
        ens_avg_wr = ens_only["win_rate"].mean()
        lgb_avg_dd = lgb_only["max_drawdown_pct"].mean()
        ens_avg_dd = ens_only["max_drawdown_pct"].mean()

        print(f"  {'Strategy':<30} {'Profit':>12} {'Avg WR%':>8} {'Avg DD%':>8}")
        print(f"  {'-'*60}")
        print(f"  {'6x LGB solo ($500 each)':30} ${lgb_total:>10.2f} {lgb_avg_wr:>7.1f}% {lgb_avg_dd:>7.1f}%")
        print(f"  {'6x Ensemble ($500 each)':30} ${ens_total:>10.2f} {ens_avg_wr:>7.1f}% {ens_avg_wr:>7.1f}%")
        print(f"  {'PORTFOLIO ($500 total)':30} ${stats['total_profit_usd']:>10.2f} {stats['win_rate']:>7.1f}% {stats['max_drawdown_pct']:>7.1f}%")


def save_portfolio_report(stats):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report_path = os.path.join(RESULTS_DIR, "portfolio_backtest_report.txt")
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_portfolio_report(stats)
    sys.stdout = old_stdout
    report_text = buffer.getvalue()

    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to {report_path}")
    return report_path


if __name__ == "__main__":
    stats = run_portfolio_backtest()
    print_portfolio_report(stats)
    save_portfolio_report(stats)
