"""
Wycena mieszkań na ul. Trzebińskiej "tu i teraz".

Wszystkie historyczne ceny (2009-2025) sprowadzone do wartości 2026
poprzez zsumowaną inflację. Jeden wykres: cena realna 2026 vs powierzchnia.

Bez kolorowania datami - interesuje nas wycena na dziś.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mieszkania_prediction import load_data, clean
from inflacja_korekta import wspolczynnik_deflacji


ULICA = "Trzebińska"


def main():
    df = load_data()
    df = clean(df, verbose=False)

    df = df[df["lok_adres"].str.contains(ULICA, case=False, na=False)].copy()

    # zsumowana inflacja -> cena realna 2026
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]
    df["cena_m2_2026"] = df["cena_2026"] / df["lok_pow_uzyt"]

    print(f"Transakcje na ul. {ULICA}: {len(df)}")
    print(f"Wszystkie ceny sprowadzone do wartości 2026.\n")

    # regresja liniowa cena ~ powierzchnia
    z = np.polyfit(df["lok_pow_uzyt"], df["cena_2026"], 1)
    a, b = z[0], z[1]
    xs = np.linspace(df["lok_pow_uzyt"].min(), df["lok_pow_uzyt"].max(), 100)

    print(f"Model liniowy (cena 2026 vs powierzchnia):")
    print(f"  cena = {a:,.0f} * m2 + {b:,.0f}")
    print(f"  czyli ~{a:,.0f} zł za każdy m2 + {b:,.0f} zł stałe")

    # ------------------------------------------------------------------
    # Wykres
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Wycena mieszkań 'tu i teraz' - ul. {ULICA} (Chrzanów)\n"
                 f"ceny historyczne sprowadzone do wartości 2026",
                 fontsize=15, fontweight="bold")

    # 1. cena 2026 vs powierzchnia
    ax = axes[0]
    ax.scatter(df["lok_pow_uzyt"], df["cena_2026"], s=55, alpha=0.6,
               color="#4C72B0", edgecolors="white", linewidths=0.6,
               label="transakcje (cena 2026)")
    ax.plot(xs, np.polyval(z, xs), "--", color="#C44E52", linewidth=2.5,
            label=f"trend: {a:,.0f} zł/m² + {b:,.0f} zł")
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena w złotówkach 2026 [zł]")
    ax.set_title("Cena 'tu i teraz' vs powierzchnia")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. cena za m2 2026 vs powierzchnia
    ax = axes[1]
    ax.scatter(df["lok_pow_uzyt"], df["cena_m2_2026"], s=55, alpha=0.6,
               color="#55A868", edgecolors="white", linewidths=0.6)
    med = df["cena_m2_2026"].median()
    ax.axhline(med, color="#C44E52", linestyle="--", linewidth=2,
               label=f"mediana: {med:,.0f} zł/m²")
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena za m² w złotówkach 2026 [zł]")
    ax.set_title("Cena za m² 'tu i teraz' vs powierzchnia")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = "trzebinska_wycenia_2026.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # statystyki
    print(f"\nStatystyki (ceny 2026):")
    print(f"  mediana ceny:      {df.cena_2026.median():,.0f} zł")
    print(f"  mediana ceny/m²:   {df.cena_m2_2026.median():,.0f} zł")
    print(f"  średnia ceny/m²:   {df.cena_m2_2026.mean():,.0f} zł")
    print(f"  zakres ceny/m²:    {df.cena_m2_2026.min():,.0f} - "
          f"{df.cena_m2_2026.max():,.0f} zł")

    # przykładowe wyceny
    print(f"\nPrzykładowe wyceny 'tu i teraz' (wg trendu):")
    for m2 in [40, 50, 60, 70, 80]:
        cena = a * m2 + b
        print(f"  {m2} m² -> {cena:,.0f} zł  ({cena/m2:,.0f} zł/m²)")


if __name__ == "__main__":
    main()
