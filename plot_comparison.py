"""
Wykresy porównujące algorytmy predykcji ceny samochodu (data.csv).

Generuje plik PNG z zestawem wykresów:
  1. RMSE (val/test) - wykres słupkowy
  2. Predykcja vs rzeczywistość (najlepszy model)
  3. Rozkład reszt (najlepszy model)
  4. Ważność cech (Random Forest)
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

from car_price_prediction import (
    load_data,
    split_data,
    build_categorical_map,
    prepare_X,
    rmse,
)


def main():
    df = load_data("data.csv")
    df_train, df_val, df_test = split_data(df)

    y_train = np.log1p(df_train["msrp"].values)
    y_val = np.log1p(df_val["msrp"].values)
    y_test = np.log1p(df_test["msrp"].values)

    for d in (df_train, df_val, df_test):
        del d["msrp"]

    categorical = build_categorical_map(df_train)

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    # nazwy cech (do wykresu ważności)
    feature_names = (
        ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg",
         "popularity", "age", "num_doors_2", "num_doors_3", "num_doors_4"]
        + [f"{name}_{value}"
           for name, values in categorical.items()
           for value in values]
    )

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
    # Rysowanie
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Porównanie algorytmów predykcji ceny samochodu",
                 fontsize=16, fontweight="bold")

    # 1. RMSE słupkowy
    ax = axes[0, 0]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, val_scores, w, label="walidacja", color="#4C72B0")
    ax.bar(x + w / 2, test_scores, w, label="test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (log-cena)")
    ax.set_title("RMSE modeli (niżej = lepiej)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (v, t) in enumerate(zip(val_scores, test_scores)):
        ax.text(i - w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)

    # 2. Predykcja vs rzeczywistość (najlepszy model)
    ax = axes[0, 1]
    ax.scatter(y_test, best_pred, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(y_test.min(), best_pred.min()),
            max(y_test.max(), best_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywista log-cena")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title(f"Predykcja vs rzeczywistość ({best_name})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Rozkład reszt (najlepszy model)
    ax = axes[1, 0]
    residuals = y_test - best_pred
    ax.hist(residuals, bins=60, color="#55A868", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("reszta (rzeczywista - przewidywana)")
    ax.set_ylabel("liczba próbek")
    ax.set_title(f"Rozkład reszt ({best_name})\n"
                 f"średnia={residuals.mean():.3f}, "
                 f"std={residuals.std():.3f}")
    ax.grid(axis="y", alpha=0.3)

    # 4. Ważność cech (Random Forest)
    ax = axes[1, 1]
    rf = models["Random Forest"]
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
    out = "model_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()
