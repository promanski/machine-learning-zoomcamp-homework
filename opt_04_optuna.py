"""
KROK 5: Optymalizacja CatBoost z Optuna.

- Cel: minimalizacja RMSE na walidacji KRZYŻOWEJ grupowej (po ulicy),
  bo to uczciwy protokół dla generalizacji na nowe lokalizacje.
- Ustalone ziarna losowości, ten sam protokół dla wszystkich prób.
- Zapis najlepszych parametrów.

Dodatkowo: porównanie targetu log(cena) vs log(cena/m2).
"""

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

from opt_data import load_dataset, metrics, split_random, split_group, split_time, SEED


BASE_NUM = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
            "x", "y", "odl_centrum"]
FEATS = BASE_NUM + ["log_pow", "izb_na_m2", "maly_metraz",
                    "wysokie_pietro", "piętro_bin", "pow_bin", "ulica"]
CATS = ["ulica", "piętro_bin", "pow_bin"]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prep_xy(df, feats=FEATS, cats=CATS, target="cena_2026", per_m2=False):
    X = df[feats].copy()
    for c in feats:
        if c in cats:
            X[c] = X[c].astype(str)
        else:
            X[c] = X[c].fillna(0)
    if per_m2:
        y = np.log(df[target].values / df["lok_pow_uzyt"].values)
    else:
        y = np.log1p(df[target].values)
    return X, y


def cv_group_rmse(params, df, n_splits=5):
    """Grupowa walidacja krzyżowa po ulicy - zwraca średnie RMSE (log ceny)."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["ulica"].values
    rmses = []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        Xtr, ytr = prep_xy(tr)
        Xva, yva = prep_xy(va)
        m = CatBoostRegressor(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            random_seed=SEED, verbose=0, cat_features=CATS)
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        rmses.append(np.sqrt(np.mean((yva - pred) ** 2)))
    return float(np.mean(rmses)), float(np.std(rmses))


def objective(trial, df):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
    }
    mean_rmse, _ = cv_group_rmse(params, df)
    return mean_rmse


def main():
    df = load_dataset()
    print(f"Zbiór: {len(df)} wierszy, {df['ulica'].nunique()} ulic")
    print("Walidacja: GroupKFold(5) po ulicy\n")

    # --- Optuna ---
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, df), n_trials=60, show_progress_bar=False)

    best = study.best_params
    print("Najlepsze parametry (Optuna):")
    for k, v in best.items():
        print(f"  {k:<22} {v}")
    print(f"\nCV RMSE (group, log) = {study.best_value:.4f}")

    # --- porównanie z domyślnymi ---
    default = {"iterations": 500, "learning_rate": 0.05, "depth": 6,
               "l2_leaf_reg": 3, "random_strength": 1.0,
               "bagging_temperature": 1.0}
    d_map = dict(default)
    # usuń parametry nie będące w objective
    dm, ds = cv_group_rmse({k: d_map[k] for k in best}, df)
    print(f"CV RMSE (default)     = {dm:.4f}")
    print(f"Poprawa: {dm - study.best_value:+.4f}")

    save = {
        "best_params": best,
        "best_cv_rmse": study.best_value,
        "default_cv_rmse": dm,
        "n_trials": 60,
        "cv": "GroupKFold(5) by ulica",
    }
    with open("opt_best_params.json", "w") as f:
        json.dump(save, f, indent=2)
    print("\nZapisano: opt_best_params.json")

    # --- finalna ocena na 3 podziałach (train na pełnym, test na holdout) ---
    print(f"\n{'='*80}")
    print("OCENA FINALNA (CatBoost z najlepszymi parametrami):")
    print(f"{'='*80}")
    full = {k: best[k] for k in best}
    for sname, splitter in [("random", split_random),
                            ("group", split_group),
                            ("time", split_time)]:
        tr, va, te = splitter(df)
        Xtr, ytr = prep_xy(tr)
        Xte, yte = prep_xy(te)
        m = CatBoostRegressor(iterations=full["iterations"],
                              learning_rate=full["learning_rate"],
                              depth=full["depth"],
                              l2_leaf_reg=full["l2_leaf_reg"],
                              random_strength=full["random_strength"],
                              bagging_temperature=full["bagging_temperature"],
                              random_seed=SEED, verbose=0, cat_features=CATS)
        m.fit(Xtr, ytr)
        r = metrics(yte, m.predict(Xte))
        print(f"  {sname:<10} RMSE(log)={r['rmse_log']:.4f}  "
              f"MAE(log)={r['mae_log']:.4f}  R2(log)={r['r2_log']:.4f}  "
              f"RMSE(zł)={r['rmse_pln']:,.0f}")


if __name__ == "__main__":
    main()