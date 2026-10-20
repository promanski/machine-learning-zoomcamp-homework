"""
Porównanie algorytmów predykcji ceny samochodu na podstawie data.csv.

Porównywane modele:
  - Ridge (regresja liniowa z regularyzacją L2)
  - Random Forest
  - Gradient Boosting
  - Hist Gradient Boosting

Cel: log1p(MSRP), metryka: RMSE.
"""

import numpy as np
import pandas as pd

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


def evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Trenuje model i zwraca RMSE na walidacji i teście."""
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    score_val = rmse(y_val, y_pred_val)
    score_test = rmse(y_test, y_pred_test)

    print(f"{name:<28} RMSE(val)={score_val:.4f}   RMSE(test)={score_test:.4f}")
    return score_val, score_test


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

    print(f"Liczba cech: {X_train.shape[1]}, "
          f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}\n")

    print("Porównanie modeli:")
    print("-" * 70)

    results = {}

    # 1. Ridge (regresja liniowa z regularyzacją L2)
    results["Ridge"] = evaluate(
        "Ridge (alpha=1.0)",
        Ridge(alpha=1.0),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # 2. Random Forest
    results["RandomForest"] = evaluate(
        "Random Forest",
        RandomForestRegressor(
            n_estimators=200, max_depth=None,
            random_state=42, n_jobs=-1,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # 3. Gradient Boosting
    results["GradientBoosting"] = evaluate(
        "Gradient Boosting",
        GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1,
            max_depth=3, random_state=42,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # 4. Hist Gradient Boosting (szybki, obsługuje duże zbiory)
    results["HistGradientBoosting"] = evaluate(
        "Hist Gradient Boosting",
        HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1,
            max_depth=None, random_state=42,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    print("-" * 70)

    # Najlepszy model wg RMSE na walidacji
    best = min(results, key=lambda k: results[k][0])
    print(f"\nNajlepszy model (val): {best}  "
          f"RMSE(val)={results[best][0]:.4f}  RMSE(test)={results[best][1]:.4f}")

    return results


if __name__ == "__main__":
    main()
