import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

RAW_PATH = ROOT / "data" / "raw" / "HousingData.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "boston_clean.csv"
TARGET = "MEDV"
OUTLIER_VALUE = 50.0


def make_dataset():
    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)
    df = df[df[TARGET] != OUTLIER_VALUE]
    print(f"Bo {n_raw - len(df)} dong outlier MEDV={OUTLIER_VALUE}, con {len(df)} dong")
    print(f"NaN con lai: {int(df.isna().sum().sum())} diem (se dien median o buoc build_features)")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Da luu -> {PROCESSED_PATH}")
    return df


if __name__ == "__main__":
    make_dataset()
