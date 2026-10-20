"""
Korekta cen historycznych o inflację (deflacja do wartości bieżącej).

Problem: cena mieszkania z 2015 i z 2026 to nie to samo "zł". Aby porównywać
i uczyć model uczciwie, sprowadzamy wszystkie ceny do wartości z 2026 roku.

Metoda: mnożymy cenę z roku Y przez skumulowany wskaźnik inflacji CPI
od roku Y+1 do roku docelowego (2026).

Źródło wskaźników: GUS (Główny Urząd Statystyczny), inflacja CPI rok do roku.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import (
    load_data, clean, split_data, build_cat_map, prepare_X, rmse,
)


# ----------------------------------------------------------------------
# Roczne wskaźniki inflacji CPI w Polsce (GUS), w procentach
# ----------------------------------------------------------------------
INFLACJA_CPI = {
    2005: 2.1, 2006: 1.0, 2007: 2.5, 2008: 4.2, 2009: 3.5,
    2010: 2.6, 2011: 4.3, 2012: 3.7, 2013: 0.9, 2014: 0.0,
    2015: -0.9, 2016: -0.6, 2017: 2.0, 2018: 1.6, 2019: 2.3,
    2020: 3.4, 2021: 5.1, 2022: 14.4, 2023: 11.4, 2024: 3.6,
    2025: 4.0, 2026: 3.0,  # 2025-2026: wartości przybliżone
}

ROK_BAZOWY = 2026


def wspolczynnik_deflacji(rok: int, rok_bazowy: int = ROK_BAZOWY) -> float:
    """Skumulowany wskaźnik inflacji od roku+1 do roku bazowego.

    Cena z roku Y * wspolczynnik = cena w złotówkach roku bazowego.
    """
    wsp = 1.0
    for y in range(rok + 1, rok_bazowy + 1):
        wsp *= (1 + INFLACJA_CPI.get(y, 0.0) / 100.0)
    return wsp


def main():
    df = load_data()
    df = clean(df, verbose=False)
    print(f"Mieszkania po czyszczeniu: {len(df)}\n")

    # --- korekta o inflację ---
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_realna_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]
    df["cena_za_m2_realna"] = df["cena_realna_2026"] / df["lok_pow_uzyt"]

    print("Współczynniki deflacji (do 2026):")
    for y in [2010, 2015, 2019, 2022, 2024]:
        print(f"  {y}: x{wspolczynnik_deflacji(y):.3f}  "
              f"(100 tys. zł z {y} = "
              f"{100000 * wspolczynnik_deflacji(y):,.0f} zł w 2026)")

    print("\nŚrednia cena za m2 wg roku (nominalna vs realna 2026):")
    grp = df.groupby("rok").agg(
        nominalna=("cena_za_m2", "mean"),
        realna=("cena_za_m2_realna", "mean"),
        n=("cena_za_m2", "size"),
    )
    for y, row in grp.iterrows():
        print(f"  {y}: nominalna {row.nominalna:>7,.0f} zł/m2  |  "
              f"realna(2026) {row.realna:>7,.0f} zł/m2  (n={int(row.n)})")

    # ------------------------------------------------------------------
    # Porównanie modeli: ceny nominalne vs realne
    # ------------------------------------------------------------------
    df_train, df_val, df_test = split_data(df)

    categorical = build_cat_map(df_train)
    numeric = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
               "x", "y", "odl_centrum", "rok"]

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    def train_eval(target_col, label):
        y_train = np.log1p(df_train[target_col].values)
        y_val = np.log1p(df_val[target_col].values)
        y_test = np.log1p(df_test[target_col].values)

        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        v = rmse(y_val, model.predict(X_val))
        t = rmse(y_test, model.predict(X_test))
        print(f"{label:<28} RMSE(val)={v:.4f}  RMSE(test)={t:.4f}")
        return model, y_test, model.predict(X_test)

    print("\nPorównanie modeli:")
    print("-" * 70)
    m_nom, yt_nom, yp_nom = train_eval("tran_cena_brutto", "ceny nominalne")
    m_real, yt_real, yp_real = train_eval("cena_realna_2026", "ceny realne (2026)")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Korekta cen o inflację (deflacja do 2026)",
                 fontsize=16, fontweight="bold")

    # 1. średnia cena za m2: nominalna vs realna
    ax = axes[0, 0]
    ax.plot(grp.index, grp["nominalna"], "o-", label="nominalna",
            color="#DD8452", linewidth=2)
    ax.plot(grp.index, grp["realna"], "s-", label="realna (2026)",
            color="#4C72B0", linewidth=2)
    ax.set_xlabel("rok transakcji")
    ax.set_ylabel("cena za m2 [zł]")
    ax.set_title("Cena za m2: nominalna vs realna")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. współczynnik inflacji
    ax = axes[0, 1]
    lata = sorted(INFLACJA_CPI.keys())
    ax.bar(lata, [INFLACJA_CPI[y] for y in lata], color="#C44E52")
    ax.set_xlabel("rok")
    ax.set_ylabel("inflacja CPI [%]")
    ax.set_title("Roczna inflacja CPI w Polsce (GUS)")
    ax.grid(axis="y", alpha=0.3)

    # 3. predykcja vs rzeczywistość (realne)
    ax = axes[1, 0]
    ax.scatter(yt_real, yp_real, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(yt_real.min(), yp_real.min()),
            max(yt_real.max(), yp_real.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywista log-cena (realna 2026)")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title("Model na cenach realnych")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. rozkład reszt (realne)
    ax = axes[1, 1]
    residuals = yt_real - yp_real
    ax.hist(residuals, bins=50, color="#55A868", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("reszta")
    ax.set_ylabel("liczba próbek")
    ax.set_title(f"Rozkład reszt (realne)\nśrednia={residuals.mean():.3f}, "
                 f"std={residuals.std():.3f}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "inflacja_korekta.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()
