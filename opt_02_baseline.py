"""
KROK 3: Benchmarking modeli bazowych i zaawansowanych.

Modele:
  - DummyRegressor (baseline)
  - Ridge (one-hot) / ElasticNet
  - CatBoost (ulica jako kategoria)
  - LightGBM (natywne kategorie)
  - XGBoost (natywne kategorie)
  - HistGradientBoosting (one-hot ulica)

Raport dla 3 podziałów (random / group / time):
  RMSE, MAE, R2 - w log ORAZ w PLN.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

from opt_data import (
    load_dataset, metrics, split_random, split_group, split_time,
    FEATURES, NUMERIC, CAT_FEATURE, SEED,
)


def make_features_onehot(df, encoder=None):
    """One-hot dla ulicy + cechy numeryczne."""
    X_num = df[NUMERIC].fillna(0).values
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = encoder.fit_transform(df[[CAT_FEATURE]])
    else:
        X_cat = encoder.transform(df[[CAT_FEATURE]])
    return np.hstack([X_num, X_cat]), encoder


def run_models(tr, va, te, label):
    Xtr, ytr = tr[FEATURES].copy(), np.log1p(tr["cena_2026"].values)
    Xte, yte = te[FEATURES].copy(), np.log1p(te["cena_2026"].values)

    results = {}

    # przygotowanie wersji one-hot (bez wycieku - encoder na train)
    Xtr_oh, enc = make_features_onehot(tr)
    Xte_oh, _ = make_features_onehot(te, enc)

    # 1. Dummy
    m = DummyRegressor(strategy="mean")
    m.fit(Xtr[NUMERIC].fillna(0), ytr)
    results["Dummy (mean)"] = metrics(yte, m.predict(Xte[NUMERIC].fillna(0)))

    # 2. Ridge
    m = Ridge(alpha=10.0)
    m.fit(Xtr_oh, ytr)
    results["Ridge (one-hot)"] = metrics(yte, m.predict(Xte_oh))

    # 3. ElasticNet
    m = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
    m.fit(Xtr_oh, ytr)
    results["ElasticNet (one-hot)"] = metrics(yte, m.predict(Xte_oh))

    # 4. HistGB (one-hot ulica)
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=SEED)
    m.fit(Xtr_oh, ytr)
    results["HistGB (one-hot)"] = metrics(yte, m.predict(Xte_oh))

    # 5. CatBoost (natywne kategorie)
    m = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=SEED, verbose=0, cat_features=[CAT_FEATURE])
    m.fit(Xtr, ytr)
    results["CatBoost (cat)"] = metrics(yte, m.predict(Xte))

    # 6. LightGBM (natywne kategorie)
    Xtr_l = Xtr.copy()
    Xte_l = Xte.copy()
    Xtr_l[CAT_FEATURE] = Xtr_l[CAT_FEATURE].astype("category")
    Xte_l[CAT_FEATURE] = pd.Categorical(
        Xte_l[CAT_FEATURE], categories=Xtr_l[CAT_FEATURE].cat.categories)
    m = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=SEED, verbose=-1, min_child_samples=10)
    m.fit(Xtr_l, ytr, categorical_feature=[CAT_FEATURE])
    results["LightGBM (cat)"] = metrics(yte, m.predict(Xte_l))

    # 7. XGBoost (natywne kategorie)
    Xtr_x = Xtr.copy()
    Xte_x = Xte.copy()
    Xtr_x[CAT_FEATURE] = Xtr_x[CAT_FEATURE].astype("category")
    Xte_x[CAT_FEATURE] = pd.Categorical(
        Xte_x[CAT_FEATURE], categories=Xtr_x[CAT_FEATURE].cat.categories)
    m = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        random_state=SEED, enable_categorical=True, verbosity=0)
    m.fit(Xtr_x, ytr)
    results["XGBoost (cat)"] = metrics(yte, m.predict(Xte_x))

    print(f"\n{'='*100}")
    print(f"PODZIAŁ: {label}   (train={len(tr)}, test={len(te)})")
    print(f"{'='*100}")
    print(f"{'Model':<24} {'RMSE(log)':>10} {'MAE(log)':>9} {'R2(log)':>8} "
          f"{'RMSE(zł)':>12} {'MAE(zł)':>11} {'R2(zł)':>7}")
    print("-" * 100)
    for name, r in sorted(results.items(), key=lambda x: x[1]["rmse_log"]):
        print(f"{name:<24} {r['rmse_log']:>10.4f} {r['mae_log']:>9.4f} "
              f"{r['r2_log']:>8.4f} {r['rmse_pln']:>12,.0f} "
              f"{r['mae_pln']:>11,.0f} {r['r2_pln']:>7.4f}")
    return results


def main():
    df = load_dataset()
    print(f"Zbiór: {len(df)} wierszy, {df['ulica'].nunique()} ulic")

    all_results = {}
    for name, splitter in [("random", split_random),
                           ("group (ulica)", split_group),
                           ("time (rok)", split_time)]:
        tr, va, te = splitter(df)
        all_results[name] = run_models(tr, va, te, name)

    # podsumowanie: najlepszy model per podział
    print(f"\n{'='*100}")
    print("PODSUMOWANIE (najlepszy model per podział, RMSE log):")
    print(f"{'='*100}")
    for split, res in all_results.items():
        best = min(res.items(), key=lambda x: x[1]["rmse_log"])
        print(f"  {split:<20} -> {best[0]:<24} RMSE(log)={best[1]['rmse_log']:.4f}")

    # zapis wyników
    import json
    with open("opt_results_baseline.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nZapisano: opt_results_baseline.json")


if __name__ == "__main__":
    main()