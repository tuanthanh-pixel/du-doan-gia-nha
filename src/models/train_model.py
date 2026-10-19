import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from features.build_features import build_preprocessor

DATA_PATH = ROOT / "data" / "processed" / "boston_clean.csv"
MODEL_PATH = ROOT / "models" / "linear_regression.joblib"
COEF_PATH = ROOT / "reports" / "coefficients.csv"
RESULTS_PATH = ROOT / "reports" / "results.csv"
TARGET = "MEDV"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    print(
        f"Train: {X_train.shape[0]} dong | Val: {X_val.shape[0]} dong | Test: {X_test.shape[0]} dong"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "MSE": mean_squared_error(y, pred),
        "RMSE": root_mean_squared_error(y, pred),
        "MAE": mean_absolute_error(y, pred),
        "MAPE": mean_absolute_percentage_error(y, pred),
        "R2": r2_score(y, pred),
    }


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = Pipeline(steps=[("preprocessor", build_preprocessor()), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)

    results = {}
    results["val"] = evaluate(model, X_val, y_val)
    results["test"] = evaluate(model, X_test, y_test)
    for name, metrics in results.items():
        print(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    feature_names = list(X_train.columns)
    coef = model.named_steps["regressor"].coef_
    intercept = model.named_steps["regressor"].intercept_
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient_scaled": coef})
    coef_df = coef_df.sort_values("coefficient_scaled", key=abs, ascending=False)
    coef_df.loc[len(coef_df)] = ["INTERCEPT", intercept]
    COEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(COEF_PATH, index=False)

    MODELS_RESULTS_DIR = ROOT / "models"
    MODELS_RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    pd.DataFrame(results).T.to_csv(RESULTS_PATH)
    print(f"Da luu mo hinh -> {MODEL_PATH}")
    print(f"Da luu he so -> {COEF_PATH}")
    print(f"Da luu ket qua -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
