"""
KROK 4: Inżynieria cech - eksperymenty addytywne.

Zestawy cech:
  A) bazowy: pow, izby, kond, x, y, odl_centrum, ulica
  B) A + nieliniowe: log_pow, pow_bin, piętro_bin, izb_na_m2
  C) B + interakcje: pow*ulica (przez cat), maly_metraz, wysokie_pietro
  D) C + geo: nr_porz

Ewaluacja na 3 podziałach (random/group/time), model CatBoost.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor

from opt_data import (
    load_dataset, metrics, split_random, split_group, split_time, SEED,
)

BASE_NUM = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
            "x", "y", "odl_centrum"]
CAT = "ulica"

FEATURE_SETS = {
    "A) bazowy": BASE_NUM + [CAT],
    "B) +nieliniowe": BASE_NUM + ["log_pow", "izb_na_m2",
                                  "maly_metraz", "wysokie_pietro",
                                  "piętro_bin", "pow_bin", CAT],
    "C) +geo_porz": BASE_NUM + ["log_pow", "izb_na_m2",
                                "maly_metraz", "wysokie_pietro",
                                "piętro_bin", "pow_bin", "nr_porz", CAT],
    "D) tylko_pow_ulica": ["lok_pow_uzyt", CAT],
    "E) bez_ulicy": BASE_NUM + ["log_pow", "izb_na_m2"],
}

# które kolumny są kategoryczne
CAT_COLS = {CAT, "piętro_bin", "pow_bin"}


def evaluate(df, feats, splitter):
    tr, va, te = splitter(df)
    Xtr = tr[feats].copy()
    Xte = te[feats].copy()
    ytr = np.log1p(tr["cena_2026"].values)
    yte = np.log1p(te["cena_2026"].values)

    cats = [c for c in feats if c in CAT_COLS]
    Xtr[cats] = Xtr[cats].astype(str)
    Xte[cats] = Xte[cats].astype(str)
    # uzupełnij braki
    for c in feats:
        if c not in cats:
            Xtr[c] = Xtr[c].fillna(0)
            Xte[c] = Xte[c].fillna(0)

    m = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=SEED, verbose=0, cat_features=cats)
    m.fit(Xtr, ytr)
    return metrics(yte, m.predict(Xte))


def main():
    df = load_dataset()
    print(f"Zbiór: {len(df)}\n")

    splits = [("random", split_random),
              ("group", split_group),
              ("time", split_time)]

    print(f"{'Zestaw cech':<22} " +
          " ".join(f"{s[0]:>10}" for s in splits))
    print("-" * 60)
    results = {}
    for fname, feats in FEATURE_SETS.items():
        row = []
        results[fname] = {}
        for sname, splitter in splits:
            r = evaluate(df, feats, splitter)
            row.append(r["rmse_log"])
            results[fname][sname] = r
        print(f"{fname:<22} " + " ".join(f"{v:>10.4f}" for v in row))

    # szczegóły najlepszego zestawu
    print(f"\n{'='*80}")
    print("SZCZEGÓŁY (RMSE log / MAE log / R2 log / RMSE zł):")
    print(f"{'='*80}")
    for fname, bys in results.items():
        print(f"\n{fname}:")
        for sname, r in bys.items():
            print(f"  {sname:<10} RMSE(log)={r['rmse_log']:.4f}  "
                  f"MAE(log)={r['mae_log']:.4f}  R2(log)={r['r2_log']:.4f}  "
                  f"RMSE(zł)={r['rmse_pln']:,.0f}")

    import json
    with open("opt_results_features.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nZapisano: opt_results_features.json")


if __name__ == "__main__":
    main()