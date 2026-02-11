import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import RESULTS_DIR, MODELS_DIR, SYMBOLS


def generate_report(results_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("ML TRADING BOT v2 — ENSEMBLE BACKTEST REPORT")
    lines.append("15m Primary TF | All TFs as Features | Optuna Tuning | Walk-Forward CV | SHAP")
    lines.append("=" * 100)
    lines.append("")

    lines.append("-" * 100)
    lines.append("ENSEMBLE RESULTS BY COIN (LGB + XGB must agree to enter)")
    lines.append("-" * 100)

    header = (
        f"  {'Symbol':10s} | {'Balance':>10s} | {'Profit':>10s} | "
        f"{'Profit%':>8s} | {'WR%':>6s} | {'Trades':>6s} | {'MaxDD%':>7s} | "
        f"{'Sharpe':>6s} | {'AvgLev':>6s} | {'PF':>5s} | {'TP':>4s} | {'SL':>4s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for _, row in results_df.iterrows():
        lines.append(
            f"  {row['symbol']:10s} | "
            f"${row['final_balance']:>9.2f} | ${row['total_profit_usd']:>9.2f} | "
            f"{row['total_profit_pct']:>7.2f}% | {row['win_rate']:>5.1f}% | "
            f"{row['total_trades']:>6d} | {row['max_drawdown_pct']:>6.2f}% | "
            f"{row['sharpe_ratio']:>5.2f} | {row['avg_leverage']:>5.1f}x | "
            f"{row['profit_factor']:>4.2f} | {row.get('tp_count', 0):>4d} | "
            f"{row.get('sl_count', 0):>4d}"
        )

    lines.append("")

    profitable = results_df[results_df["total_profit_pct"] > 0]
    unprofitable = results_df[results_df["total_profit_pct"] <= 0]

    lines.append(f"  Profitable coins: {len(profitable)} / {len(results_df)}")
    if not results_df.empty:
        lines.append(f"  Average profit: {results_df['total_profit_pct'].mean():.2f}%")
        lines.append(f"  Average win rate: {results_df['win_rate'].mean():.1f}%")
        lines.append(f"  Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("SHAP FEATURE IMPORTANCE (Top 15 per coin)")
    lines.append("=" * 100)

    for symbol in SYMBOLS:
        lgb_shap_path = os.path.join(MODELS_DIR, f"{symbol}_shap_lgb.csv")
        if os.path.exists(lgb_shap_path):
            shap_df = pd.read_csv(lgb_shap_path)
            top15 = shap_df.head(15)
            lines.append(f"\n  {symbol} (LightGBM top 15):")
            for idx, row in top15.iterrows():
                bar_len = int(row["shap_importance"] / (top15["shap_importance"].max() + 1e-10) * 30)
                bar = "#" * bar_len
                lines.append(f"    {row['feature']:40s} {row['shap_importance']:.6f} {bar}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 100)

    if not results_df.empty:
        best = results_df.iloc[0]
        lines.append(f"\n  BEST COIN: {best['symbol']}")
        lines.append(f"    Final Balance: ${best['final_balance']:.2f} (started $500)")
        lines.append(f"    Total Profit: ${best['total_profit_usd']:.2f} ({best['total_profit_pct']:.2f}%)")
        lines.append(f"    Win Rate: {best['win_rate']:.1f}%")
        lines.append(f"    Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        lines.append(f"    Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        lines.append(f"    Profit Factor: {best['profit_factor']:.2f}")
        lines.append(f"    Avg Leverage: {best['avg_leverage']:.1f}x")

    lines.append("")

    total_invested = len(results_df) * 500
    total_final = results_df["final_balance"].sum() if not results_df.empty else 0
    total_profit = total_final - total_invested

    lines.append(f"  PORTFOLIO (all {len(results_df)} coins, $500 each = ${total_invested} total):")
    lines.append(f"    Total Final Balance: ${total_final:.2f}")
    lines.append(f"    Total Profit: ${total_profit:.2f} ({(total_profit/total_invested)*100:.2f}%)")

    report = "\n".join(lines)
    return report


def save_report():
    csv_path = os.path.join(RESULTS_DIR, "backtest_results_v2.csv")
    if not os.path.exists(csv_path):
        print("No backtest results found. Run backtester first.")
        return

    results_df = pd.read_csv(csv_path)
    report = generate_report(results_df)

    report_path = os.path.join(RESULTS_DIR, "report_v2.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    save_report()
