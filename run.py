import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature_engine import build_all_features
from src.trainer import train_all_models
from src.backtester import run_all_backtests
from src.report import generate_report


def main():
    start = time.time()

    print("\n" + "=" * 60)
    print("STEP 1/4: BUILDING FEATURES v2 (200+ features, all TFs)")
    print("=" * 60)

    from configs.settings import DATA_DIR, SYMBOLS
    all_exist = all(
        os.path.exists(os.path.join(DATA_DIR, s, "features_v2.parquet"))
        for s in SYMBOLS
    )
    if all_exist:
        print("Features already computed, skipping...")
    else:
        build_all_features()

    print("\n" + "=" * 60)
    print("STEP 2/4: TRAINING 12 MODELS (Optuna + Walk-Forward CV)")
    print("=" * 60)
    train_all_models()

    print("\n" + "=" * 60)
    print("STEP 3/4: RUNNING ENSEMBLE BACKTESTS (6 coins)")
    print("=" * 60)
    results_df = run_all_backtests()

    print("\n" + "=" * 60)
    print("STEP 4/4: GENERATING REPORT")
    print("=" * 60)
    report = generate_report(results_df)
    print(report)

    from configs.settings import RESULTS_DIR
    report_path = os.path.join(RESULTS_DIR, "report_v2.txt")
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
