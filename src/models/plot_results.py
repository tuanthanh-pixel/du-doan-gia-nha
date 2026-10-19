import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.train_model import load_data

MODEL_PATH = ROOT / "models" / "linear_regression.joblib"
FIG_DIR = ROOT / "reports" / "figures"


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_test)
    residual = y_test - pred

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, residual, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Gia du doan")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Linear Regression - Residual plot (test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", label="y = x")
    ax.set_xlabel("Gia thuc (MEDV)")
    ax.set_ylabel("Gia du doan")
    ax.set_title("Linear Regression - Du doan vs thuc te (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predictions_vs_true.png", dpi=150)

    print(f"Da luu bieu do -> {FIG_DIR / 'residuals.png'}")
    print(f"Da luu bieu do -> {FIG_DIR / 'predictions_vs_true.png'}")


if __name__ == "__main__":
    main()
