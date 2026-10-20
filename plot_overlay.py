"""
Nakładka danych predykcyjnych i prawdziwych (data.csv).

Generuje plik PNG z wykresami:
  1. Posortowane ceny: prawdziwe vs przewidywane (nakładka liniowa)
  2. Nakładające się rozkłady (histogram) prawdziwych i przewidywanych cen
  3. Ceny w oryginalnej skali (expm1) - pierwsze 100 samochodów
  4. Błąd względny w funkcji ceny rzeczywistej
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

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

    # najlepszy model
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"RMSE(test) = {rmse(y_test, y_pred):.4f}")

    # ceny w oryginalnej skali
    y_test_price = np.expm1(y_test)
    y_pred_price = np.expm1(y_pred)

    # ------------------------------------------------------------------
    # Rysowanie
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Nakładka danych predykcyjnych i prawdziwych",
                 fontsize=16, fontweight="bold")

    # 1. Posortowane ceny: prawdziwe vs przewidywane
    ax = axes[0, 0]
    order = np.argsort(y_test)
    ax.plot(y_test[order], label="prawdziwe", color="#4C72B0", linewidth=2)
    ax.plot(y_pred[order], label="przewidywane", color="#DD8452",
            linewidth=2, alpha=0.85)
    ax.set_xlabel("samochody (posortowane wg ceny rzeczywistej)")
    ax.set_ylabel("log-cena")
    ax.set_title("Posortowane ceny: prawdziwe vs przewidywane")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Nakładające się rozkłady (histogram)
    ax = axes[0, 1]
    ax.hist(y_test, bins=60, alpha=0.55, label="prawdziwe",
            color="#4C72B0", density=True)
    ax.hist(y_pred, bins=60, alpha=0.55, label="przewidywane",
            color="#DD8452", density=True)
    ax.set_xlabel("log-cena")
    ax.set_ylabel("gęstość")
    ax.set_title("Nakładka rozkładów cen")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Ceny w oryginalnej skali - pierwsze 100 samochodów
    ax = axes[1, 0]
    n = 100
    idx = np.arange(n)
    ax.plot(idx, y_test_price[:n], "o-", label="prawdziwe",
            color="#4C72B0", markersize=4, linewidth=1)
    ax.plot(idx, y_pred_price[:n], "s--", label="przewidywane",
            color="#DD8452", markersize=4, linewidth=1)
    ax.set_xlabel("samochód")
    ax.set_ylabel("cena (MSRP)")
    ax.set_title("Ceny w oryginalnej skali (pierwsze 100)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Błąd względny w funkcji ceny rzeczywistej
    ax = axes[1, 1]
    rel_err = (y_pred_price - y_test_price) / y_test_price * 100
    ax.scatter(y_test_price, rel_err, s=8, alpha=0.3, color="#C44E52")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("cena rzeczywista (log skala)")
    ax.set_ylabel("błąd względny [%]")
    ax.set_title("Błąd względny predykcji")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "prediction_overlay.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Zapisano wykres: {out}")


if __name__ == "__main__":
    main()
