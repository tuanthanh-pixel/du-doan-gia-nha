"""Du doan gia nha tu 13 dac trung dau vao."""
import sys
from pathlib import Path

import argparse
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "linear_regression.joblib"

FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]

DEFAULT_SAMPLE = [0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]


def predict(values):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame(np.array(values, dtype=float).reshape(1, -1), columns=FEATURES)
    return float(model.predict(X)[0])


def main():
    parser = argparse.ArgumentParser(description="Du doan gia nha - Boston Housing")
    parser.add_argument(
        "values",
        nargs="*",
        type=float,
        help="13 gia tri theo thu tu: " + ", ".join(FEATURES),
    )
    args = parser.parse_args()
    values = args.values if args.values else DEFAULT_SAMPLE
    if len(values) != len(FEATURES):
        raise SystemExit(
            f"Can dung {len(FEATURES)} gia tri, nhan duoc {len(values)}. Thu tu: " + ", ".join(FEATURES)
        )

    pred = predict(values)
    print("Dau vao:")
    for name, value in zip(FEATURES, values):
        print(f"  {name} = {value}")
    print(f"Gia du doan (MEDV): {pred:.2f} (x1000 USD) = {pred * 1000:,.0f} USD")


if __name__ == "__main__":
    main()
