"""
Tuning hiperparametrów dla modeli predykcji (data.csv).

Zadania:
  1. Cena (msrp)          -> Hist Gradient Boosting
  2. Wiek bez ceny (age)  -> Random Forest

Metoda: RandomizedSearchCV z walidacją krzyżową (5-fold).
Metryka: RMSE (neg_mean_squared_error -> pierwiastek).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import RandomizedSearchCV, KFold

from car_price_prediction import (
    load_data, split_data, build_categorical_map, prepare_X, rmse,
)


# ----------------------------------------------------------------------
# Wspólne przygotowanie cech
# ----------------------------------------------------------------------
CATEGORICAL_COLUMNS = [
    "make", "model", "engine_fuel_type", "transmission_type",
    "driven_wheels", "market_category", "vehicle_size", "vehicle_style",
]


def build_cat_map(df_train):
    return {c: list(df_train[c].value_counts().head().index)
            for c in CATEGORICAL_COLUMNS}


def prepare_features(df, categorical, numeric_features):
    df = df.copy()
    features = list(numeric_features)
    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)
    return df[features].fillna(0).values


def rmse_scorer(estimator, X, y):
    """Scorer RMSE dla GridSearchCV (im wyżej tym lepiej -> negatyw)."""
    pred = estimator.predict(X)
    return -rmse(y, pred)


# ----------------------------------------------------------------------
# 1. Tuning ceny (Hist Gradient Boosting)
# ----------------------------------------------------------------------
def tune_price(df_train, df_val, df_test):
    print("=" * 70)
    print("1. TUNING CENY (Hist Gradient Boosting)")
    print("=" * 70)

    y_train = np.log1p(df_train["msrp"].values)
    y_val = np.log1p(df_val["msrp"].values)
    y_test = np.log1p(df_test["msrp"].values)

    for d in (df_train, df_val, df_test):
        del d["msrp"]

    categorical = build_cat_map(df_train)
    numeric = ["engine_hp", "engine_cylinders", "highway_mpg",
               "city_mpg", "popularity", "year"]

    X_train = prepare_features(df_train, categorical, numeric)
    X_val = prepare_features(df_val, categorical, numeric)
    X_test = prepare_features(df_test, categorical, numeric)

    param_dist = {
        "max_iter": [200, 300, 500, 800],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    }

    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=30, cv=KFold(5, shuffle=True, random_state=42),
        scoring=rmse_scorer, random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"Najlepsze parametry: {search.best_params_}")
    print(f"CV RMSE (train): {(-search.best_score_):.4f}")

    best = search.best_estimator_
    val_score = rmse(y_val, best.predict(X_val))
    test_score = rmse(y_test, best.predict(X_test))
    print(f"RMSE(val)  = {val_score:.4f}")
    print(f"RMSE(test) = {test_score:.4f}")

    # baseline (domyślne parametry)
    base = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    base.fit(X_train, y_train)
    base_test = rmse(y_test, base.predict(X_test))
    print(f"\nBaseline RMSE(test) = {base_test:.4f}")
    print(f"Poprawa: {base_test - test_score:+.4f}")

    return {
        "baseline": base_test,
        "tuned": test_score,
        "params": search.best_params_,
    }


# ----------------------------------------------------------------------
# 2. Tuning wieku bez ceny (Random Forest)
# ----------------------------------------------------------------------
def tune_age(df_train, df_val, df_test):
    print("\n" + "=" * 70)
    print("2. TUNING WIEKU BEZ CENY (Random Forest)")
    print("=" * 70)

    for d in (df_train, df_val, df_test):
        d["age"] = 2017 - d["year"]

    y_train = df_train["age"].values
    y_val = df_val["age"].values
    y_test = df_test["age"].values

    categorical = build_cat_map(df_train)
    numeric = ["engine_hp", "engine_cylinders", "number_of_doors",
               "highway_mpg", "city_mpg", "popularity"]

    X_train = prepare_features(df_train, categorical, numeric)
    X_val = prepare_features(df_val, categorical, numeric)
    X_test = prepare_features(df_test, categorical, numeric)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30, cv=KFold(5, shuffle=True, random_state=42),
        scoring=rmse_scorer, random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"Najlepsze parametry: {search.best_params_}")
    print(f"CV RMSE (train): {(-search.best_score_):.4f}")

    best = search.best_estimator_
    val_score = rmse(y_val, best.predict(X_val))
    test_score = rmse(y_test, best.predict(X_test))
    print(f"RMSE(val)  = {val_score:.4f}")
    print(f"RMSE(test) = {test_score:.4f}")

    # baseline
    base = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1)
    base.fit(X_train, y_train)
    base_test = rmse(y_test, base.predict(X_test))
    print(f"\nBaseline RMSE(test) = {base_test:.4f}")
    print(f"Poprawa: {base_test - test_score:+.4f}")

    return {
        "baseline": base_test,
        "tuned": test_score,
        "params": search.best_params_,
    }


def main():
    df = load_data("data.csv")
    df_train, df_val, df_test = split_data(df)

    # kopia dla drugiego zadania (bo pierwsze usuwa msrp)
    df_train2 = df_train.copy()
    df_val2 = df_val.copy()
    df_test2 = df_test.copy()

    res_price = tune_price(df_train, df_val, df_test)
    res_age = tune_age(df_train2, df_val2, df_test2)

    # ------------------------------------------------------------------
    # Wykres porównania baseline vs tuned
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Cena (HistGB)", "Wiek (RandomForest)"]
    baseline = [res_price["baseline"], res_age["baseline"]]
    tuned = [res_price["tuned"], res_age["tuned"]]

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, baseline, w, label="baseline", color="#4C72B0")
    b2 = ax.bar(x + w / 2, tuned, w, label="po tuningu", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (test)")
    ax.set_title("Tuning hiperparametrów: baseline vs po tuningu")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = "tuning_results.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()
