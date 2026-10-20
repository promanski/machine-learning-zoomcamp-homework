"""
Najlepszy model CatBoost dla ul. Trzebińskiej (Chrzanów).

Strategia:
  - Trening na Trzebińskiej + sąsiednich ulicach (z eksperymentu: 5 ulic).
  - Ulica jako cecha kategoryczna (CatBoost natywnie).
  - Ceny skorygowane o inflację -> wartości 2026 (wycena "tu i teraz").
  - Porównanie z HistGB i zapis najlepszego modelu.

Cel: cena realna 2026 (log1p). Metryka: RMSE.
"""

import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# poprawne renderowanie polskich znaków
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import load_data, clean, rmse
from inflacja_korekta import wspolczynnik_deflacji


ULICA_TARGET = "Trzebińska"
ULICE_SASIEDNIE = ["Grzybowa", "Broniewskiego", "Mieszka I",
                   "Wyszyńskiego", "Sienkiewicza"]

NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum"]
CAT_FEATURE = "ulica"
MODEL_FILE = "model_trzebinska_cat.pkl"


def wyciagnij_ulice(adres) -> str:
    if not isinstance(adres, str):
        return "brak"
    m = re.search(r"UL:([^;]+)", adres)
    return m.group(1).strip() if m else "brak"


def main():
    df = load_data()
    df = clean(df, verbose=False)

    # korekta o inflację
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]

    # ulica
    df["ulica"] = df["lok_adres"].apply(wyciagnij_ulice)

    # zbiór: Trzebińska + sąsiednie
    mask = df["ulica"].isin([ULICA_TARGET] + ULICE_SASIEDNIE)
    df_opt = df[mask].copy()

    # odległość od centrum Trzebińskiej
    trz = df_opt[df_opt["ulica"] == ULICA_TARGET]
    cx, cy = trz["x"].median(), trz["y"].median()
    df_opt["odl_centrum"] = np.sqrt(
        (df_opt["x"] - cx) ** 2 + (df_opt["y"] - cy) ** 2)

    print(f"Zbiór: {len(df_opt)} transakcji")
    print(f"  Trzebińska: {len(trz)}")
    print(f"  sąsiednie: {len(df_opt) - len(trz)}")

    # podział: test = 20% Trzebińskiej
    n = len(trz)
    idx = np.arange(n)
    np.random.seed(2)
    np.random.shuffle(idx)
    n_test = int(n * 0.2)
    test_idx = trz.index[idx[:n_test]]

    df_test = df_opt.loc[test_idx].reset_index(drop=True)
    df_train = df_opt.drop(test_idx).reset_index(drop=True)

    print(f"\nTrening: {len(df_train)}  Test (Trzebińska): {len(df_test)}")

    y_train = np.log1p(df_train["cena_2026"].values)
    y_test = np.log1p(df_test["cena_2026"].values)

    features = NUMERIC + [CAT_FEATURE]
    X_train = df_train[features].fillna(0)
    X_test = df_test[features].fillna(0)

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    cat_model = CatBoostRegressor(
        iterations=800, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        cat_features=[CAT_FEATURE],
    )
    cat_model.fit(X_train, y_train)

    pred_cat = cat_model.predict(X_test)
    rmse_cat = rmse(y_test, pred_cat)
    pln_cat = np.sqrt(
        ((np.expm1(y_test) - np.expm1(pred_cat)) ** 2).mean())

    # ------------------------------------------------------------------
    # HistGB (baseline, bez ulicy)
    # ------------------------------------------------------------------
    hgb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    hgb.fit(df_train[NUMERIC].fillna(0), y_train)
    pred_hgb = hgb.predict(df_test[NUMERIC].fillna(0))
    rmse_hgb = rmse(y_test, pred_hgb)
    pln_hgb = np.sqrt(
        ((np.expm1(y_test) - np.expm1(pred_hgb)) ** 2).mean())

    print("\nWyniki (test Trzebińskiej):")
    print("-" * 70)
    print(f"{'CatBoost (ulica kat.)':<28} RMSE={rmse_cat:.4f}  "
          f"({pln_cat:,.0f} zł)")
    print(f"{'HistGB (bez ulicy)':<28} RMSE={rmse_hgb:.4f}  "
          f"({pln_hgb:,.0f} zł)")

    # ------------------------------------------------------------------
    # Zapis najlepszego modelu
    # ------------------------------------------------------------------
    best_model = cat_model if rmse_cat <= rmse_hgb else hgb
    best_name = "CatBoost" if rmse_cat <= rmse_hgb else "HistGB"
    best_rmse = min(rmse_cat, rmse_hgb)
    best_pln = pln_cat if rmse_cat <= rmse_hgb else pln_hgb

    # centra ulic (do wyceny)
    centra = {}
    for ul in [ULICA_TARGET] + ULICE_SASIEDNIE:
        d = df_opt[df_opt["ulica"] == ul]
        centra[ul] = {"x": float(d["x"].median()),
                      "y": float(d["y"].median()),
                      "n": int(len(d))}

    payload = {
        "model": best_model,
        "model_type": best_name,
        "features": features,
        "numeric": NUMERIC,
        "cat_feature": CAT_FEATURE,
        "centra": centra,
        "ulice": [ULICA_TARGET] + ULICE_SASIEDNIE,
        "rmse_test": best_rmse,
        "rmse_test_pln": best_pln,
        "n_train": len(df_train),
        "ceny_realne": True,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nZapisano najlepszy model ({best_name}): {MODEL_FILE}")
    print(f"RMSE(test) = {best_rmse:.4f}  ({best_pln:,.0f} zł)")

    # ------------------------------------------------------------------
    # Przykładowe wyceny
    # ------------------------------------------------------------------
    print("\nPrzykładowe wyceny (Trzebińska, ceny 2026):")
    for m2, izby, kond in [(50, 2, 1), (60, 3, 2), (65, 3, 0),
                           (70, 3, 3), (80, 4, 4)]:
        row = {
            "lok_pow_uzyt": m2, "lok_liczba_izb": izby, "lok_nr_kond": kond,
            "x": centra[ULICA_TARGET]["x"], "y": centra[ULICA_TARGET]["y"],
            "odl_centrum": 0.0, "ulica": ULICA_TARGET,
        }
        X = pd.DataFrame([row])[features].fillna(0)
        cena = float(np.expm1(best_model.predict(X)[0]))
        print(f"  {m2} m², {izby} izb, piętro {kond} -> "
              f"{cena:,.0f} zł  ({cena/m2:,.0f} zł/m²)")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Najlepszy model dla ul. Trzebińskiej (CatBoost)",
                 fontsize=15, fontweight="bold")

    # 1. predykcja vs rzeczywistość
    ax = axes[0]
    ax.scatter(y_test, pred_cat, s=40, alpha=0.6, color="#4C72B0",
               edgecolors="white", linewidths=0.5)
    lims = [min(y_test.min(), pred_cat.min()),
            max(y_test.max(), pred_cat.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna")
    ax.set_xlabel("log-cena rzeczywista (2026)")
    ax.set_ylabel("log-cena przewidywana")
    ax.set_title(f"CatBoost (RMSE={rmse_cat:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. porównanie modeli
    ax = axes[1]
    labels = ["CatBoost\n(z ulicą)", "HistGB\n(bez ulicy)"]
    vals = [rmse_cat, rmse_hgb]
    bars = ax.bar(labels, vals, color=["#55A868", "#DD8452"])
    ax.set_ylabel("RMSE (test, log-cena)")
    ax.set_title("Porównanie modeli")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", fontsize=10)

    # 3. ważność cech
    ax = axes[2]
    imp = cat_model.get_feature_importance()
    order = np.argsort(imp)
    ax.barh([features[i] for i in order], [imp[i] for i in order],
            color="#C44E52")
    ax.set_xlabel("ważność")
    ax.set_title("Ważność cech (CatBoost)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = "trzebinska_catboost.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()