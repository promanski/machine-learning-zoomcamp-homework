"""
Predykcja wieku samochodu (age = 2017 - year) na podstawie data.csv.

Model: Hist Gradient Boosting.
Cel: age (wiek auta), metryka: RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from car_price_prediction import load_data, split_data, rmse


# cechy numeryczne używane do predykcji wieku
NUMERIC_FEATURES = [
    "engine_hp", "engine_cylinders", "number_of_doors",
    "highway_mpg", "city_mpg", "popularity", "msrp",
]

CATEGORICAL_COLUMNS = [
    "make", "model", "engine_fuel_type", "transmission_type",
    "driven_wheels", "market_category", "vehicle_size", "vehicle_style",
]


def build_categorical_map(df_train: pd.DataFrame) -> dict:
    categorical = {}
    for c in CATEGORICAL_COLUMNS:
        categorical[c] = list(df_train[c].value_counts().head().index)
    return categorical


def prepare_X(df: pd.DataFrame, categorical: dict) -> np.ndarray:
    df = df.copy()

    features = list(NUMERIC_FEATURES)

    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)

    df_num = df[features].fillna(0)
    return df_num.values


def main():
    df = load_data("data.csv")
    df_train, df_val, df_test = split_data(df)

    # cel: wiek samochodu
    for d in (df_train, df_val, df_test):
        d["age"] = 2017 - d["year"]

    y_train = df_train["age"].values
    y_val = df_val["age"].values
    y_test = df_test["age"].values

    categorical = build_categorical_map(df_train)

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print("Predykcja wieku samochodu (age):")
    print("-" * 60)
    print(f"RMSE(val)  = {rmse(y_val, y_pred_val):.4f}")
    print(f"RMSE(test) = {rmse(y_test, y_pred_test):.4f}")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Predykcja wieku samochodu (age)",
                 fontsize=16, fontweight="bold")

    # 1. predykcja vs rzeczywistość
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywisty wiek (lata)")
    ax.set_ylabel("przewidywany wiek (lata)")
    ax.set_title("Wiek: predykcja vs rzeczywistość")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. nakładka rozkładów
    ax = axes[0, 1]
    ax.hist(y_test, bins=40, alpha=0.55, label="prawdziwe",
            color="#4C72B0", density=True)
    ax.hist(y_pred_test, bins=40, alpha=0.55, label="przewidywane",
            color="#DD8452", density=True)
    ax.set_xlabel("wiek (lata)")
    ax.set_ylabel("gęstość")
    ax.set_title("Nakładka rozkładów wieku")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. rozkład reszt
    ax = axes[1, 0]
    residuals = y_test - y_pred_test
    ax.hist(residuals, bins=50, color="#55A868", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("reszta (rzeczywisty - przewidywany)")
    ax.set_ylabel("liczba próbek")
    ax.set_title(f"Rozkład reszt\nśrednia={residuals.mean():.2f}, "
                 f"std={residuals.std():.2f}")
    ax.grid(axis="y", alpha=0.3)

    # 4. ważność cech
    ax = axes[1, 1]
    feature_names = (
        list(NUMERIC_FEATURES)
        + [f"{name}_{value}"
           for name, values in categorical.items()
           for value in values]
    )
    # HistGradientBoosting nie ma feature_importances_ -> użyj permutation
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    order = np.argsort(importances)[::-1][:15]
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="#C44E52",
    )
    ax.set_xlabel("ważność (permutation)")
    ax.set_title("Top 15 najważniejszych cech")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "age_prediction.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # przykładowa predykcja
    print("\nPrzykład (wiersz 20):")
    print(f"  wiek: przewidywany={y_pred_test[20]:.1f}  "
          f"rzeczywisty={y_test[20]:.1f}")


if __name__ == "__main__":
    main()
