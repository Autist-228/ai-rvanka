import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import MODELS_DIR, RESULTS_DIR, SYMBOLS, PRIMARY_TF, ALL_TIMEFRAMES, HIGHER_TF_FEATURES, LOWER_TF_FEATURES
from src.portfolio_manager import MetaAllocator, RiskController, PORTFOLIO_CONFIG
from src.portfolio_backtester import (
    PortfolioPosition, decode_prediction, compute_dynamic_leverage,
    compute_portfolio_stats, print_portfolio_report,
)
from src.feature_engine import (
    add_ta_indicators, add_extended_indicators, add_regime_features,
    add_time_features, compute_indicators_for_tf, add_cross_tf_ratios,
    add_cross_coin_features, get_feature_cols, EXCLUDE_COLS,
)

BYBIT_DATA_DIR = os.path.join(os.path.dirname(MODELS_DIR), "data_bybit")

REALISTIC_CONFIG = {
    "slippage_min_pct": 0.0001,
    "slippage_max_pct": 0.0005,
    "entry_delay_candles": 1,
    "funding_rate_enabled": True,
    "funding_interval_candles": 32,
    "taker_fee_pct": 0.00055,
    "maker_fee_pct": 0.0002,
    "use_taker_fees": True,
}


def load_bybit_symbol_data(symbol):
    symbol_dir = os.path.join(BYBIT_DATA_DIR, symbol)
    data = {}
    for tf in ALL_TIMEFRAMES:
        fpath = os.path.join(symbol_dir, f"klines_{tf}.parquet")
        if os.path.exists(fpath):
            data[tf] = pd.read_parquet(fpath)
    fr_path = os.path.join(symbol_dir, "funding_rate.parquet")
    if os.path.exists(fr_path):
        data["funding_rate"] = pd.read_parquet(fr_path)
    return data


def merge_bybit_timeframes(data):
    primary = data.get(PRIMARY_TF)
    if primary is None:
        raise ValueError(f"Primary timeframe {PRIMARY_TF} not found")

    df = compute_indicators_for_tf(primary, PRIMARY_TF, is_primary=True)

    higher_tfs = [tf for tf in ["1h", "4h"] if tf in data and tf != PRIMARY_TF]
    for tf in higher_tfs:
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


def build_bybit_features():
    print("Building features from Bybit data...")

    btc_data = load_bybit_symbol_data("BTCUSDT")
    btc_primary = btc_data.get(PRIMARY_TF)

    all_features = {}
    for symbol in SYMBOLS:
        print(f"\n  Processing {symbol}...")
        data = load_bybit_symbol_data(symbol)
        if PRIMARY_TF not in data:
            print(f"    Skipping {symbol} - no {PRIMARY_TF} data")
            continue

        df = merge_bybit_timeframes(data)
        df = add_cross_tf_ratios(df)

        if btc_primary is not None and symbol != "BTCUSDT":
            df = add_cross_coin_features(df, btc_primary)

        df["symbol"] = symbol
        all_features[symbol] = df
        print(f"    {symbol}: {len(df)} rows, {len(get_feature_cols(df))} features")

    return all_features


def load_funding_rates():
    funding = {}
    for symbol in SYMBOLS:
        fr_path = os.path.join(BYBIT_DATA_DIR, symbol, "funding_rate.parquet")
        if os.path.exists(fr_path):
            fr_df = pd.read_parquet(fr_path)
            fr_df = fr_df.sort_values("timestamp").reset_index(drop=True)
            funding[symbol] = fr_df
    return funding


def apply_slippage(price, direction, slippage_pct):
    if direction == 1:
        return price * (1 + slippage_pct)
    else:
        return price * (1 - slippage_pct)


def apply_exit_slippage(price, direction, slippage_pct):
    if direction == 1:
        return price * (1 - slippage_pct)
    else:
        return price * (1 + slippage_pct)


def get_funding_cost(funding_df, timestamp, position_value):
    if funding_df is None or funding_df.empty:
        return 0.0
    mask = funding_df["timestamp"] <= timestamp
    if not mask.any():
        return 0.0
    rate = funding_df.loc[mask].iloc[-1]["funding_rate"]
    return position_value * rate


def run_bybit_backtest_v4():
    print("=" * 80)
    print("BYBIT BACKTEST v4 - REALISTIC SIMULATION")
    print("  + Random slippage (0.01-0.05%)")
    print("  + Funding rate costs (every 8h)")
    print("  + Entry delay (1 candle = 15 min)")
    print("  + Taker fees (0.055%)")
    print("=" * 80)

    rng = np.random.RandomState(42)
    rc = REALISTIC_CONFIG

    bybit_data = build_bybit_features()
    funding_rates = load_funding_rates()

    models = {}
    feature_cols = {}
    test_data = {}

    for symbol in SYMBOLS:
        lgb_path = os.path.join(MODELS_DIR, f"{symbol}_lightgbm_v2.pkl")
        fc_path = os.path.join(MODELS_DIR, f"{symbol}_feature_cols_v2.pkl")

        if not os.path.exists(lgb_path) or not os.path.exists(fc_path):
            print(f"  Skipping {symbol} - no model")
            continue
        if symbol not in bybit_data:
            print(f"  Skipping {symbol} - no Bybit data")
            continue

        models[symbol] = joblib.load(lgb_path)
        trained_feature_cols = joblib.load(fc_path)

        df = bybit_data[symbol].copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        missing_cols = [c for c in trained_feature_cols if c not in df.columns]
        for col in missing_cols:
            df[col] = 0
        df[trained_feature_cols] = df[trained_feature_cols].fillna(0)

        df = df.sort_values("timestamp").reset_index(drop=True)

        warmup = 250
        df = df.iloc[warmup:].reset_index(drop=True)

        feature_cols[symbol] = trained_feature_cols
        test_data[symbol] = df
        print(f"  Loaded {symbol}: model + {len(df)} Bybit candles")

    if not models:
        print("No models loaded!")
        return None

    config = PORTFOLIO_CONFIG
    initial_balance = config["initial_balance"]
    trading = config["trading"]

    fee_pct = rc["taker_fee_pct"] if rc["use_taker_fees"] else rc["maker_fee_pct"]

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

    pending_signals = []

    total_slippage_cost = 0.0
    total_funding_cost = 0.0
    total_fee_cost = 0.0
    funding_events = 0

    min_len = min(len(df) for df in test_data.values())
    print(f"\nBybit test period: {min_len} candles (~{min_len * 15 / 60 / 24:.0f} days)")
    print(f"Initial balance: ${initial_balance}")
    print(f"Max position: ${config['allocator']['max_position_usd']}")
    print(f"Slippage: {rc['slippage_min_pct']*100:.2f}%-{rc['slippage_max_pct']*100:.2f}%")
    print(f"Entry delay: {rc['entry_delay_candles']} candle(s)")
    print(f"Fee: {fee_pct*100:.3f}% (taker)")
    print(f"Funding rate: {'ON' if rc['funding_rate_enabled'] else 'OFF'}")
    print(f"Coins: {list(test_data.keys())}")
    print()

    for i in range(min_len):
        if i % 1000 == 0 and i > 0:
            print(f"  Candle {i}/{min_len} | Balance: ${balance:.2f} | "
                  f"Positions: {len(positions)} | Trades: {len(closed_trades)} | "
                  f"Slippage: ${total_slippage_cost:.2f} | Funding: ${total_funding_cost:.2f}")

        sample_symbol = list(test_data.keys())[0]
        current_time = test_data[sample_symbol].iloc[i]["timestamp"]
        current_dt = test_data[sample_symbol].iloc[i]["datetime"]
        current_date = str(current_dt.date()) if hasattr(current_dt, 'date') else str(current_dt)[:10]

        if rc["funding_rate_enabled"] and i > 0 and i % rc["funding_interval_candles"] == 0:
            for pos in positions:
                pos_value = pos.size_usd * pos.leverage
                fr_df = funding_rates.get(pos.symbol)
                if fr_df is not None:
                    f_cost = get_funding_cost(fr_df, current_time, pos_value)
                    balance -= abs(f_cost)
                    total_funding_cost += abs(f_cost)
                    funding_events += 1

        for pos in positions[:]:
            row = test_data[pos.symbol].iloc[i]
            high = row["high"]
            low = row["low"]
            close = row["close"]

            pos.update_trailing(high, low)

            exit_price = None
            exit_reason = None

            if pos.direction == 1:
                if high >= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "TP"
                elif low <= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = "TRAILING_SL" if pos.trailing_activated else "SL"
            else:
                if low <= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "TP"
                elif high >= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = "TRAILING_SL" if pos.trailing_activated else "SL"

            if exit_price is not None:
                slip = rng.uniform(rc["slippage_min_pct"], rc["slippage_max_pct"])
                exit_price = apply_exit_slippage(exit_price, pos.direction, slip)

                pos.exit_price = exit_price
                pos.exit_time = current_dt
                pos.exit_reason = exit_reason

                if pos.direction == 1:
                    price_change = (exit_price - pos.entry_price) / pos.entry_price
                else:
                    price_change = (pos.entry_price - exit_price) / pos.entry_price

                trade_value = pos.size_usd * pos.leverage
                pos.pnl = trade_value * price_change
                pos.pnl -= trade_value * fee_pct

                slip_cost = trade_value * slip
                total_slippage_cost += slip_cost
                total_fee_cost += trade_value * fee_pct

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

        equity_curve.append({"timestamp": current_time, "balance": balance})

        if balance <= 0:
            print(f"  LIQUIDATED at candle {i}")
            break

        new_pending = []
        for ps in pending_signals:
            if i >= ps["execute_at"]:
                symbol = ps["symbol"]
                sig = ps["signal"]

                if len(positions) >= trading["max_concurrent_total"]:
                    continue
                active_count = sum(1 for p in positions if p.symbol == symbol)
                if active_count >= trading["max_concurrent_per_coin"]:
                    continue

                row = test_data[symbol].iloc[i]
                entry_price = row["open"]

                slip = rng.uniform(rc["slippage_min_pct"], rc["slippage_max_pct"])
                entry_price = apply_slippage(entry_price, sig["direction"], slip)

                entry_slip_cost = sig["size_usd"] * sig["leverage"] * slip
                total_slippage_cost += entry_slip_cost

                entry_fee = sig["size_usd"] * sig["leverage"] * fee_pct
                total_fee_cost += entry_fee

                atr = sig["atr"]
                coin_tpsl = config.get("coin_tp_sl", {}).get(symbol, {})
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

                trail_config = config["risk_controller"]
                pos = PortfolioPosition(
                    symbol=symbol,
                    direction=sig["direction"],
                    entry_price=entry_price,
                    size_usd=sig["size_usd"],
                    leverage=sig["leverage"],
                    tp_price=tp_price,
                    sl_price=sl_price,
                    entry_time=current_dt,
                    trailing_activation=trail_config["trailing_activation_pct"],
                    trailing_step=trail_config["trailing_step_pct"],
                )
                positions.append(pos)
                coin_stats[symbol]["allocated_usd"] += sig["size_usd"]
            else:
                new_pending.append(ps)
        pending_signals = new_pending

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
                "time": current_dt,
                "balance": balance,
                "allocations": dict(allocations),
            })

        leverage_mult = risk_ctrl.get_leverage_multiplier(balance, initial_balance)

        for symbol, alloc_pct in allocations.items():
            sig = signals[symbol]
            if sig["direction"] == 0:
                continue

            max_pos_usd = config["allocator"].get("max_position_usd", 300.0)
            size_usd = min(balance * alloc_pct, max_pos_usd)
            if size_usd < 1:
                continue

            leverage = compute_dynamic_leverage(sig["confidence"], sig["volatility"], leverage_mult)

            pending_signals.append({
                "symbol": symbol,
                "execute_at": i + rc["entry_delay_candles"],
                "signal": {
                    "direction": sig["direction"],
                    "confidence": sig["confidence"],
                    "size_usd": size_usd,
                    "leverage": leverage,
                    "atr": sig["atr"],
                },
            })

    for pos in positions:
        last_row = test_data[pos.symbol].iloc[-1]
        slip = rng.uniform(rc["slippage_min_pct"], rc["slippage_max_pct"])
        exit_price = apply_exit_slippage(last_row["close"], pos.direction, slip)

        pos.exit_price = exit_price
        pos.exit_time = last_row["datetime"]
        if pos.direction == 1:
            price_change = (exit_price - pos.entry_price) / pos.entry_price
        else:
            price_change = (pos.entry_price - exit_price) / pos.entry_price
        trade_value = pos.size_usd * pos.leverage
        pos.pnl = trade_value * price_change
        pos.pnl -= trade_value * fee_pct
        total_slippage_cost += trade_value * slip
        total_fee_cost += trade_value * fee_pct
        pos.exit_reason = "CLOSE"
        balance += pos.pnl
        closed_trades.append(pos)
        coin_stats[pos.symbol]["trades"] += 1
        coin_stats[pos.symbol]["pnl"] += pos.pnl
        if pos.pnl > 0:
            coin_stats[pos.symbol]["wins"] += 1

    stats = compute_portfolio_stats(closed_trades, equity_curve, initial_balance, coin_stats)
    stats["allocation_log"] = allocation_log
    stats["data_source"] = "Bybit (REALISTIC v4)"
    stats["total_slippage_cost"] = round(total_slippage_cost, 2)
    stats["total_funding_cost"] = round(total_funding_cost, 2)
    stats["total_fee_cost"] = round(total_fee_cost, 2)
    stats["funding_events"] = funding_events

    return stats


def save_bybit_report_v4(stats):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "bybit_backtest_v4_realistic.txt")

    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    print("\n" + "=" * 80)
    print("BYBIT BACKTEST v4 - REALISTIC SIMULATION")
    print("=" * 80)
    print(f"\n  Data Source:      Bybit (via proxy) - REALISTIC MODE")
    print(f"  Initial Balance:  ${stats['initial_balance']:.2f}")
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
    print(f"  REALISTIC COSTS:")
    print(f"  {'='*70}")
    print(f"  Total Slippage Cost:  ${stats['total_slippage_cost']:.2f}")
    print(f"  Total Funding Cost:   ${stats['total_funding_cost']:.2f}")
    print(f"  Total Fee Cost:       ${stats['total_fee_cost']:.2f}")
    print(f"  Total Hidden Costs:   ${stats['total_slippage_cost'] + stats['total_funding_cost'] + stats['total_fee_cost']:.2f}")
    print(f"  Funding Events:       {stats['funding_events']}")

    print(f"\n  {'='*70}")
    print(f"  COIN BREAKDOWN:")
    print(f"  {'='*70}")
    print(f"  {'Coin':<12} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>12} {'Allocated':>12}")
    print(f"  {'-'*70}")

    for symbol, cs in sorted(stats.get("coin_breakdown", {}).items(), key=lambda x: x[1]["pnl"], reverse=True):
        print(f"  {symbol:<12} {cs['trades']:>7} {cs['wins']:>6} {cs['win_rate']:>6.1f}% ${cs['pnl']:>10.2f} ${cs['allocated_usd']:>10.2f}")

    print(f"\n  {'='*70}")
    print(f"  COMPARISON: v3 (ideal) vs v4 (realistic)")
    print(f"  {'='*70}")
    print(f"  v3: $500 -> $3993 (+698%) - ideal execution")
    print(f"  v4: $500 -> ${stats['final_balance']:.2f} ({stats['total_profit_pct']:+.2f}%) - realistic")
    diff_pct = (stats['final_balance'] - 3993) / 3993 * 100
    print(f"  Difference: {diff_pct:+.1f}% (realistic penalty)")

    sys.stdout = old_stdout
    report_text = buffer.getvalue()

    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {report_path}")
    return report_path


if __name__ == "__main__":
    stats = run_bybit_backtest_v4()
    if stats and stats.get("total_trades", 0) > 0:
        save_bybit_report_v4(stats)
    else:
        print("No trades executed - check data and models")
