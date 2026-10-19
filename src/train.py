"""Huan luyen mo hinh du doan gia nha (Boston Housing)."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "HousingData.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {
        "RMSE": mse ** 0.5,
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Kich thuoc train: {X_train.shape}, test: {X_test.shape}")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    best_name, best_model, best_r2 = None, None, -float("inf")
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        print(f"{name}: {metrics}")
        if metrics["R2"] > best_r2:
            best_name, best_model, best_r2 = name, model, metrics["R2"]

    # Luu mo hinh tot nhat
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Da luu mo hinh tot nhat: {best_name} -> {model_path}")

    # Bieu do trung thuc vs du doan
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", label="y = x")
    ax.set_xlabel("Gia thuc (MEDV)")
    ax.set_ylabel("Gia du doan")
    ax.set_title(f"{best_name} - du doan vs thuc te")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predictions_vs_true.png", dpi=150)
    print(f"Da luu bieu do -> {FIG_DIR / 'predictions_vs_true.png'}")

    # Ket qua
    report_path = ROOT / "reports" / "results.csv"
    pd.DataFrame(results).T.to_csv(report_path)
    print(f"Da luu ket qua -> {report_path}")


if __name__ == "__main__":
    main()
