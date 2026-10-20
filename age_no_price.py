"""
Predykcja wieku samochodu BEZ użycia ceny (msrp) + porównanie modeli.

Cel: age (wiek auta), metryka: RMSE.
Cechy: tylko techniczne (bez msrp) - sprawdzamy, jak dobrze da się
przewidzieć wiek bez "podpowiedzi" w postaci ceny.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from car_price_prediction import load_data, split_data, rmse


# cechy numeryczne BEZ msrp
NUMERIC_FEATURES = [
    "engine_hp", "engine_cylinders", "number_of_doors",
    "highway_mpg", "city_mpg", "popularity",
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

    print("Predykcja wieku BEZ ceny (msrp) - porównanie modeli:")
    print("-" * 70)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=42),
    }

    names, val_scores, test_scores = [], [], []
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        p_val = model.predict(X_val)
        p_test = model.predict(X_test)
        names.append(name)
        val_scores.append(rmse(y_val, p_val))
        test_scores.append(rmse(y_test, p_test))
        preds[name] = p_test
        print(f"{name:<24} RMSE(val)={val_scores[-1]:.4f}  "
              f"RMSE(test)={test_scores[-1]:.4f}")

    best_name = names[int(np.argmin(val_scores))]
    best_pred = preds[best_name]
    print(f"\nNajlepszy model: {best_name}")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Predykcja wieku samochodu BEZ ceny (msrp)",
                 fontsize=16, fontweight="bold")

    # 1. RMSE słupkowy
    ax = axes[0, 0]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, val_scores, w, label="walidacja", color="#4C72B0")
    ax.bar(x + w / 2, test_scores, w, label="test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (lata)")
    ax.set_title("RMSE modeli (niżej = lepiej)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (v, t) in enumerate(zip(val_scores, test_scores)):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=8)

    # 2. predykcja vs rzeczywistość (najlepszy)
    ax = axes[0, 1]
    ax.scatter(y_test, best_pred, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(y_test.min(), best_pred.min()),
            max(y_test.max(), best_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywisty wiek (lata)")
    ax.set_ylabel("przewidywany wiek (lata)")
    ax.set_title(f"Predykcja vs rzeczywistość ({best_name})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. nakładka rozkładów
    ax = axes[1, 0]
    ax.hist(y_test, bins=40, alpha=0.55, label="prawdziwe",
            color="#4C72B0", density=True)
    ax.hist(best_pred, bins=40, alpha=0.55, label="przewidywane",
            color="#DD8452", density=True)
    ax.set_xlabel("wiek (lata)")
    ax.set_ylabel("gęstość")
    ax.set_title(f"Nakładka rozkładów ({best_name})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. ważność cech (Random Forest)
    ax = axes[1, 1]
    rf = models["Random Forest"]
    feature_names = (
        list(NUMERIC_FEATURES)
        + [f"{name}_{value}"
           for name, values in categorical.items()
           for value in values]
    )
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="#C44E52",
    )
    ax.set_xlabel("ważność")
    ax.set_title("Top 15 najważniejszych cech (Random Forest)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "age_no_price.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # przykładowa predykcja
    print("\nPrzykład (wiersz 20):")
    print(f"  wiek: przewidywany={best_pred[20]:.1f}  "
          f"rzeczywisty={y_test[20]:.1f}")


if __name__ == "__main__":
    main()
