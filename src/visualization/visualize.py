import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.train_model import load_data

DATA_PATH = ROOT / "data" / "processed" / "boston_clean.csv"
COEF_PATH = ROOT / "reports" / "coefficients.csv"
MODEL_PATH = ROOT / "models" / "linear_regression.joblib"
FIG_DIR = ROOT / "reports" / "figures"
TARGET = "MEDV"

sns.set_theme(style="whitegrid")


def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Phan phoi {TARGET}")
    axes[0].set_xlabel("Gia nha (x1000 USD)")
    sns.boxplot(x=df[TARGET], ax=axes[1], color="steelblue")
    axes[1].set_title(f"Boxplot {TARGET}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Da luu -> {FIG_DIR / 'target_distribution.png'}")


def plot_correlation_heatmap(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Ma tran tuong quan cac dac trung")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Da luu -> {FIG_DIR / 'correlation_heatmap.png'}")


def plot_feature_vs_target(df):
    corr = df.corr()[TARGET].abs().drop(TARGET).sort_values(ascending=False)
    top_features = corr.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feature in zip(axes.flat, top_features):
        sns.regplot(data=df, x=feature, y=TARGET, ax=ax, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
        ax.set_title(f"{feature} vs {TARGET} (corr={corr[feature]:.2f})")
    fig.suptitle("4 dac trung tuong quan manh nhat voi gia nha")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_vs_target.png", dpi=150)
    plt.close(fig)
    print(f"Da luu -> {FIG_DIR / 'feature_vs_target.png'}")


def plot_predictions(y_test, pred):
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
    plt.close(fig)
    print(f"Da luu -> {FIG_DIR / 'predictions_vs_true.png'}")


def plot_residuals(y_test, pred):
    residual = y_test - pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(pred, residual, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Gia du doan")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Residual plot (test)")

    ax = axes[1]
    sns.histplot(residual, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_title("Phan phoi residual")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", dpi=150)
    plt.close(fig)
    print(f"Da luu -> {FIG_DIR / 'residuals.png'}")


def plot_coefficients():
    coef_df = pd.read_csv(COEF_PATH)
    coef_df = coef_df[coef_df["feature"] != "INTERCEPT"].copy()
    coef_df = coef_df.sort_values("coefficient_scaled", key=abs)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(coef_df["coefficient_scaled"] >= 0, "tab:blue", "tab:red")
    ax.barh(coef_df["feature"], coef_df["coefficient_scaled"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("He so hoi quy (du lieu chuan hoa)")
    ax.set_title("Tac dong cua tung dac trung")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coefficients.png", dpi=150)
    plt.close(fig)
    print(f"Da luu -> {FIG_DIR / 'coefficients.png'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_test)
    plot_predictions(y_test, pred)
    plot_residuals(y_test, pred)

    plot_coefficients()


if __name__ == "__main__":
    main()
