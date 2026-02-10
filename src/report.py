import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import RESULTS_DIR, SYMBOLS


def generate_report(results_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("ML TRADING BOT — BACKTEST REPORT")
    lines.append("=" * 100)
    lines.append("")

    lines.append("-" * 100)
    lines.append("TOP 10 BEST COMBINATIONS (by profit %)")
    lines.append("-" * 100)
    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        lines.append(
            f"  {row['symbol']:10s} | {row['model']:10s} | {row['risk_profile']:14s} | "
            f"Profit: ${row['total_profit_usd']:>8.2f} ({row['total_profit_pct']:>7.2f}%) | "
            f"WR: {row['win_rate']:>5.1f}% | Trades: {row['total_trades']:>4d} | "
            f"MaxDD: {row['max_drawdown_pct']:>6.2f}% | Sharpe: {row['sharpe_ratio']:>5.2f}"
        )

    lines.append("")
    lines.append("-" * 100)
    lines.append("BOTTOM 10 WORST COMBINATIONS (by profit %)")
    lines.append("-" * 100)
    bottom10 = results_df.tail(10)
    for _, row in bottom10.iterrows():
        lines.append(
            f"  {row['symbol']:10s} | {row['model']:10s} | {row['risk_profile']:14s} | "
            f"Profit: ${row['total_profit_usd']:>8.2f} ({row['total_profit_pct']:>7.2f}%) | "
            f"WR: {row['win_rate']:>5.1f}% | Trades: {row['total_trades']:>4d} | "
            f"MaxDD: {row['max_drawdown_pct']:>6.2f}% | Sharpe: {row['sharpe_ratio']:>5.2f}"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("DETAILED RESULTS BY COIN")
    lines.append("=" * 100)

    for symbol in SYMBOLS:
        sym_data = results_df[results_df["symbol"] == symbol]
        if sym_data.empty:
            continue

        lines.append(f"\n{'─' * 80}")
        lines.append(f"  {symbol}")
        lines.append(f"{'─' * 80}")

        header = (
            f"  {'Model':10s} | {'Risk':14s} | {'Balance':>10s} | {'Profit':>10s} | "
            f"{'Profit%':>8s} | {'WR%':>6s} | {'Trades':>6s} | {'MaxDD%':>7s} | "
            f"{'Sharpe':>6s} | {'AvgLev':>6s} | {'PF':>5s} | {'BestTr':>8s} | {'WorstTr':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for _, row in sym_data.iterrows():
            lines.append(
                f"  {row['model']:10s} | {row['risk_profile']:14s} | "
                f"${row['final_balance']:>9.2f} | ${row['total_profit_usd']:>9.2f} | "
                f"{row['total_profit_pct']:>7.2f}% | {row['win_rate']:>5.1f}% | "
                f"{row['total_trades']:>6d} | {row['max_drawdown_pct']:>6.2f}% | "
                f"{row['sharpe_ratio']:>5.2f} | {row['avg_leverage']:>5.1f}x | "
                f"{row['profit_factor']:>4.2f} | ${row['best_trade_usd']:>7.2f} | "
                f"${row['worst_trade_usd']:>7.2f}"
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("MODEL COMPARISON: LightGBM vs XGBoost")
    lines.append("=" * 100)

    for symbol in SYMBOLS:
        sym_data = results_df[results_df["symbol"] == symbol]
        lgb_data = sym_data[sym_data["model"] == "lightgbm"]
        xgb_data = sym_data[sym_data["model"] == "xgboost"]

        if lgb_data.empty or xgb_data.empty:
            continue

        lgb_avg_profit = lgb_data["total_profit_pct"].mean()
        xgb_avg_profit = xgb_data["total_profit_pct"].mean()
        lgb_avg_wr = lgb_data["win_rate"].mean()
        xgb_avg_wr = xgb_data["win_rate"].mean()

        winner = "LightGBM" if lgb_avg_profit > xgb_avg_profit else "XGBoost"

        lines.append(f"\n  {symbol}:")
        lines.append(f"    LightGBM — Avg Profit: {lgb_avg_profit:>7.2f}%, Avg WR: {lgb_avg_wr:>5.1f}%")
        lines.append(f"    XGBoost  — Avg Profit: {xgb_avg_profit:>7.2f}%, Avg WR: {xgb_avg_wr:>5.1f}%")
        lines.append(f"    >>> WINNER: {winner}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("RISK PROFILE COMPARISON")
    lines.append("=" * 100)

    for risk_name in ["conservative", "moderate", "aggressive"]:
        risk_data = results_df[results_df["risk_profile"] == risk_name]
        if risk_data.empty:
            continue
        lines.append(f"\n  {risk_name.upper()}:")
        lines.append(f"    Avg Profit: {risk_data['total_profit_pct'].mean():>7.2f}%")
        lines.append(f"    Avg Win Rate: {risk_data['win_rate'].mean():>5.1f}%")
        lines.append(f"    Avg MaxDD: {risk_data['max_drawdown_pct'].mean():>6.2f}%")
        lines.append(f"    Avg Sharpe: {risk_data['sharpe_ratio'].mean():>5.2f}")
        lines.append(f"    Best combo: {risk_data.iloc[0]['symbol']} {risk_data.iloc[0]['model']} "
                      f"({risk_data.iloc[0]['total_profit_pct']}%)")

    lines.append("")
    lines.append("=" * 100)
    lines.append("OVERALL RECOMMENDATION")
    lines.append("=" * 100)

    if not results_df.empty:
        best = results_df.iloc[0]
        lines.append(f"\n  BEST OVERALL: {best['symbol']} | {best['model']} | {best['risk_profile']}")
        lines.append(f"    Final Balance: ${best['final_balance']:.2f} (started with $500)")
        lines.append(f"    Total Profit: ${best['total_profit_usd']:.2f} ({best['total_profit_pct']:.2f}%)")
        lines.append(f"    Win Rate: {best['win_rate']:.1f}%")
        lines.append(f"    Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        lines.append(f"    Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        lines.append(f"    Profit Factor: {best['profit_factor']:.2f}")

    report = "\n".join(lines)
    return report


def save_report():
    csv_path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    if not os.path.exists(csv_path):
        print("No backtest results found. Run backtester first.")
        return

    results_df = pd.read_csv(csv_path)
    report = generate_report(results_df)

    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    save_report()
