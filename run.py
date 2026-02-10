import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import download_all_data
from src.feature_engine import build_all_features
from src.trainer import train_all_models
from src.backtester import run_all_backtests
from src.report import generate_report


def main():
    start = time.time()

    print("\n" + "=" * 60)
    print("STEP 1/5: DOWNLOADING DATA FROM BYBIT")
    print("=" * 60)
    download_all_data()

    print("\n" + "=" * 60)
    print("STEP 2/5: BUILDING FEATURES")
    print("=" * 60)
    build_all_features()

    print("\n" + "=" * 60)
    print("STEP 3/5: TRAINING 12 MODELS")
    print("=" * 60)
    train_all_models()

    print("\n" + "=" * 60)
    print("STEP 4/5: RUNNING 36 BACKTESTS")
    print("=" * 60)
    results_df = run_all_backtests()

    print("\n" + "=" * 60)
    print("STEP 5/5: GENERATING REPORT")
    print("=" * 60)
    report = generate_report(results_df)
    print(report)

    from configs.settings import RESULTS_DIR
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    print(f"\nTotal time: {hours}h {mins}m {secs}s")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
