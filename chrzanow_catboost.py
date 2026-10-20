"""
CatBoost: wycena mieszkań w Chrzanowie z ulicą jako cechą kategoryczną.

- Wyciąga ulicę z adresu (UL:...) i normalizuje literówki.
- Używa CatBoost z natywną obsługą cech kategorycznych (ulica).
- Ceny skorygowane o inflację -> wartości 2026 (wycena "tu i teraz").
- Porównuje z Hist Gradient Boosting (baseline).

Cel: cena realna 2026 (log1p). Metryka: RMSE.
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import load_data, clean, rmse
from inflacja_korekta import wspolczynnik_deflacji


# normalizacja nazw ulic (literówki, warianty)
ULICA_POPRAWKI = {
    "Partyzanów": "Partyzantów",
    "Al.Henryka": "Henryka",
    "Urbańczyka Tadeusza": "Urbańczyka",
    "Pęckowskiego Jana": "Pęckowskiego",
    "Kardynała Stefana Wyszyńskiego": "Wyszyńskiego",
    "Wincentego Witosa": "Witosa",
}

NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum"]


def wyciagnij_ulice(adres: str) -> str:
    m = re.search(r"UL:([^;]+)", adres)
    if not m:
        return "brak"
    u = m.group(1).strip()
    return ULICA_POPRAWKI.get(u, u)


def main():
    df = load_data()
    df = clean(df, verbose=False)

    # tylko Chrzanów
    df = df[df["lok_adres"].str.contains("Chrzanów", case=False,
                                         na=False)].copy()

    # korekta o inflację
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]

    # ulica jako kategoria
    df["ulica"] = df["lok_adres"].apply(wyciagnij_ulice)

    # odległość od centrum Chrzanowa (centroid)
    cx, cy = df["x"].median(), df["y"].median()
    df["odl_centrum"] = np.sqrt((df["x"] - cx) ** 2 + (df["y"] - cy) ** 2)

    print(f"Transakcje w Chrzanowie: {len(df)}")
    print(f"Unikalnych ulic (po normalizacji): {df['ulica'].nunique()}")

    # ulice z minimum transakcji
    counts = df["ulica"].value_counts()
    min_n = 5
    rzadkie = counts[counts < min_n].index
    df["ulica_grupa"] = df["ulica"].where(~df["ulica"].isin(rzadkie), "inna")
    print(f"Ulic z >= {min_n} transakcji: "
          f"{(counts >= min_n).sum()}")
    print(f"Ulic rzadkich (grupowane w 'inna'): {len(rzadkie)}")
    print()

    # podział train/test
    n = len(df)
    idx = np.arange(n)
    np.random.seed(2)
    np.random.shuffle(idx)
    n_test = int(n * 0.2)
    n_val = int(n * 0.2)
    df_train = df.iloc[idx[:n - n_val - n_test]].reset_index(drop=True)
    df_val = df.iloc[idx[n - n_val - n_test:n - n_test]].reset_index(drop=True)
    df_test = df.iloc[idx[n - n_test:]].reset_index(drop=True)

    y_train = np.log1p(df_train["cena_2026"].values)
    y_val = np.log1p(df_val["cena_2026"].values)
    y_test = np.log1p(df_test["cena_2026"].values)

    # ------------------------------------------------------------------
    # CatBoost (ulica jako cecha kategoryczna)
    # ------------------------------------------------------------------
    CAT_FEATURE = "ulica_grupa"
    features = NUMERIC + [CAT_FEATURE]

    X_train = df_train[features].fillna(0)
    X_val = df_val[features].fillna(0)
    X_test = df_test[features].fillna(0)

    cat_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=42, verbose=0,
        cat_features=[CAT_FEATURE],
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    pred_cat_val = cat_model.predict(X_val)
    pred_cat_test = cat_model.predict(X_test)

    rmse_cat_val = rmse(y_val, pred_cat_val)
    rmse_cat_test = rmse(y_test, pred_cat_test)
    pln_cat = np.sqrt(
        ((np.expm1(y_test) - np.expm1(pred_cat_test)) ** 2).mean())

    print("Wyniki:")
    print("-" * 70)
    print(f"{'CatBoost (ulica kat.)':<32} RMSE(val)={rmse_cat_val:.4f}  "
          f"RMSE(test)={rmse_cat_test:.4f}  ({pln_cat:,.0f} zł)")

    # ------------------------------------------------------------------
    # baseline: HistGB bez ulicy
    # ------------------------------------------------------------------
    Xh_train = df_train[NUMERIC].fillna(0)
    Xh_val = df_val[NUMERIC].fillna(0)
    Xh_test = df_test[NUMERIC].fillna(0)

    hgb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    hgb.fit(Xh_train, y_train)
    rmse_hgb_val = rmse(y_val, hgb.predict(Xh_val))
    rmse_hgb_test = rmse(y_test, hgb.predict(Xh_test))
    pln_hgb = np.sqrt(
        ((np.expm1(y_test) - np.expm1(hgb.predict(Xh_test))) ** 2).mean())
    print(f"{'HistGB (bez ulicy)':<32} RMSE(val)={rmse_hgb_val:.4f}  "
          f"RMSE(test)={rmse_hgb_test:.4f}  ({pln_hgb:,.0f} zł)")

    # ------------------------------------------------------------------
    # Ocena per ulica (CatBoost)
    # ------------------------------------------------------------------
    print("\nOcena per ulica (CatBoost, test):")
    print("-" * 70)
    df_test = df_test.copy()
    df_test["pred"] = np.expm1(pred_cat_test)
    df_test["real"] = np.expm1(y_test)
    per_street = []
    for ul, g in df_test.groupby("ulica_grupa"):
        if len(g) < 5:
            continue
        r = rmse(np.log1p(g["real"]), np.log1p(g["pred"]))
        per_street.append((ul, len(g), r, g["real"].median(),
                           g["pred"].median()))
    per_street.sort(key=lambda x: x[2])
    print(f"{'ulica':<26} {'n':>4} {'RMSE':>7} {'med.real':>11} "
          f"{'med.pred':>11}")
    for ul, n, r, mr, mp in per_street:
        print(f"{ul:<26} {n:>4} {r:>7.3f} {mr:>11,.0f} {mp:>11,.0f}")

    # ------------------------------------------------------------------
    # Ważność cech
    # ------------------------------------------------------------------
    print("\nWażność cech (CatBoost):")
    imp = cat_model.get_feature_importance()
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        print(f"  {f:<20} {v:>6.2f}")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CatBoost: wycena mieszkań w Chrzanowie (ceny 2026)",
                 fontsize=16, fontweight="bold")

    # 1. predykcja vs rzeczywistość
    ax = axes[0, 0]
    ax.scatter(y_test, pred_cat_test, s=12, alpha=0.4, color="#4C72B0")
    lims = [min(y_test.min(), pred_cat_test.min()),
            max(y_test.max(), pred_cat_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna")
    ax.set_xlabel("log-cena rzeczywista (2026)")
    ax.set_ylabel("log-cena przewidywana")
    ax.set_title(f"CatBoost: predykcja vs rzeczywistość "
                 f"(RMSE={rmse_cat_test:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. porównanie modeli
    ax = axes[0, 1]
    labels = ["CatBoost\n(z ulicą)", "HistGB\n(bez ulicy)"]
    vals = [rmse_cat_test, rmse_hgb_test]
    bars = ax.bar(labels, vals, color=["#55A868", "#DD8452"])
    ax.set_ylabel("RMSE (test, log-cena)")
    ax.set_title("Porównanie modeli")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", fontsize=10)

    # 3. mediana ceny/m2 per ulica (realna 2026)
    ax = axes[1, 0]
    street_m2 = []
    for ul, g in df.groupby("ulica_grupa"):
        if len(g) < 8:
            continue
        street_m2.append((ul, (g["cena_2026"] / g["lok_pow_uzyt"]).median(),
                          len(g)))
    street_m2.sort(key=lambda x: -x[1])
    top = street_m2[:15]
    ax.barh([s[0] for s in top][::-1], [s[1] for s in top][::-1],
            color="#4C72B0")
    ax.set_xlabel("mediana ceny za m² 2026 [zł]")
    ax.set_title("Najdroższe ulice Chrzanowa (mediana, ceny 2026)")
    ax.grid(axis="x", alpha=0.3)

    # 4. ważność cech
    ax = axes[1, 1]
    order = np.argsort(imp)
    ax.barh([features[i] for i in order], [imp[i] for i in order],
            color="#C44E52")
    ax.set_xlabel("ważność")
    ax.set_title("Ważność cech (CatBoost)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "chrzanow_catboost.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()