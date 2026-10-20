"""
Wszystkie transakcje mieszkań na ul. Trzebińskiej (Chrzanów).

Wykres: cena vs powierzchnia, z podziałem na:
  - ceny nominalne (jak zapłacono w danym roku)
  - ceny realne (skorygowane o inflację do 2026)

Kolor punktów = rok transakcji.
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

    # tylko Trzebińska
    df = df[df["lok_adres"].str.contains(ULICA, case=False, na=False)].copy()

    # korekta o inflację
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_realna_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]
    df["cena_m2_nom"] = df["tran_cena_brutto"] / df["lok_pow_uzyt"]
    df["cena_m2_real"] = df["cena_realna_2026"] / df["lok_pow_uzyt"]

    print(f"Transakcje na ul. {ULICA}: {len(df)}")
    print(f"Zakres lat: {df.rok.min()} - {df.rok.max()}")
    print(f"Zakres powierzchni: {df.lok_pow_uzyt.min():.1f} - "
          f"{df.lok_pow_uzyt.max():.1f} m2")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Wszystkie transakcje mieszkań - ul. {ULICA} (Chrzanów)",
                 fontsize=16, fontweight="bold")

    # wspólna skala kolorów (rok)
    vmin, vmax = df.rok.min(), df.rok.max()

    # 1. cena nominalna vs powierzchnia
    ax = axes[0, 0]
    sc = ax.scatter(df["lok_pow_uzyt"], df["tran_cena_brutto"],
                    c=df["rok"], cmap="viridis", s=45, alpha=0.8,
                    vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena nominalna [zł]")
    ax.set_title("Cena NOMINALNA vs powierzchnia")
    plt.colorbar(sc, ax=ax, label="rok transakcji")
    ax.grid(alpha=0.3)

    # 2. cena realna (2026) vs powierzchnia
    ax = axes[0, 1]
    sc = ax.scatter(df["lok_pow_uzyt"], df["cena_realna_2026"],
                    c=df["rok"], cmap="viridis", s=45, alpha=0.8,
                    vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena realna 2026 [zł]")
    ax.set_title("Cena REALNA (skorygowana o inflację) vs powierzchnia")
    plt.colorbar(sc, ax=ax, label="rok transakcji")
    ax.grid(alpha=0.3)

    # 3. cena za m2 nominalna vs realna
    ax = axes[1, 0]
    ax.scatter(df["lok_pow_uzyt"], df["cena_m2_nom"],
               c=df["rok"], cmap="viridis", s=45, alpha=0.8,
               vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.5,
               label="nominalna")
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena za m² [zł]")
    ax.set_title("Cena za m² NOMINALNA vs powierzchnia")
    plt.colorbar(sc, ax=ax, label="rok transakcji")
    ax.grid(alpha=0.3)

    # 4. cena za m2 realna
    ax = axes[1, 1]
    sc = ax.scatter(df["lok_pow_uzyt"], df["cena_m2_real"],
                    c=df["rok"], cmap="viridis", s=45, alpha=0.8,
                    vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena za m² realna 2026 [zł]")
    ax.set_title("Cena za m² REALNA (2026) vs powierzchnia")
    plt.colorbar(sc, ax=ax, label="rok transakcji")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "trzebinska_transakcje.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # ------------------------------------------------------------------
    # Dodatkowy wykres: nominalna vs realna (bezpośrednie porównanie)
    # ------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(12, 8))
    sc1 = ax.scatter(df["lok_pow_uzyt"], df["tran_cena_brutto"],
                     c="#DD8452", s=50, alpha=0.6, label="nominalna",
                     edgecolors="white", linewidths=0.5)
    sc2 = ax.scatter(df["lok_pow_uzyt"], df["cena_realna_2026"],
                     c="#4C72B0", s=50, alpha=0.6, label="realna 2026",
                     edgecolors="white", linewidths=0.5)
    # linie trendu
    for col, color in [("tran_cena_brutto", "#DD8452"),
                       ("cena_realna_2026", "#4C72B0")]:
        z = np.polyfit(df["lok_pow_uzyt"], df[col], 1)
        xs = np.linspace(df["lok_pow_uzyt"].min(),
                         df["lok_pow_uzyt"].max(), 100)
        ax.plot(xs, np.polyval(z, xs), "--", color=color, linewidth=2)
    ax.set_xlabel("powierzchnia [m²]")
    ax.set_ylabel("cena [zł]")
    ax.set_title(f"Ul. {ULICA}: ceny nominalne vs realne (2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out2 = "trzebinska_nom_vs_real.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"Zapisano wykres: {out2}")

    # statystyki
    print(f"\nŚrednia cena nominalna: {df.tran_cena_brutto.mean():,.0f} zł")
    print(f"Średnia cena realna 2026: {df.cena_realna_2026.mean():,.0f} zł")
    print(f"Średnia cena/m² nominalna: {df.cena_m2_nom.mean():,.0f} zł")
    print(f"Średnia cena/m² realna: {df.cena_m2_real.mean():,.0f} zł")


if __name__ == "__main__":
    main()
