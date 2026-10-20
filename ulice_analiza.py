"""
Analiza cen mieszkań tylko z ulic Trzebińska i Niepodległości (Chrzanów).

Filtruje transakcje do dwóch konkretnych ulic i porównuje:
  - statystyki cen (nominalne i realne 2026)
  - rozkład cen za m2
  - trend w czasie
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mieszkania_prediction import load_data, clean
from inflacja_korekta import wspolczynnik_deflacji


ULICE = ["Trzebińska", "Niepodległości"]


def main():
    df = load_data()
    df = clean(df, verbose=False)

    # filtr: tylko wybrane ulice
    mask = df["lok_adres"].str.contains("|".join(ULICE), case=False, na=False)
    df = df[mask].copy()

    # przypisz ulicę
    def ktora_ulica(adres):
        for u in ULICE:
            if u.lower() in adres.lower():
                return u
        return "inne"

    df["ulica"] = df["lok_adres"].apply(ktora_ulica)

    # korekta o inflację
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_realna_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]
    df["cena_za_m2_realna"] = df["cena_realna_2026"] / df["lok_pow_uzyt"]

    print(f"Transakcje na wybranych ulicach: {len(df)}\n")

    # --- statystyki per ulica ---
    print("Statystyki cen (nominalne):")
    print("-" * 75)
    for u in ULICE:
        d = df[df["ulica"] == u]
        print(f"\n{u} (n={len(d)}):")
        print(f"  cena:        min {d.tran_cena_brutto.min():>10,.0f}  "
              f"mediana {d.tran_cena_brutto.median():>10,.0f}  "
              f"max {d.tran_cena_brutto.max():>10,.0f} zł")
        print(f"  cena za m2:  min {d.cena_za_m2.min():>10,.0f}  "
              f"mediana {d.cena_za_m2.median():>10,.0f}  "
              f"max {d.cena_za_m2.max():>10,.0f} zł")
        print(f"  powierzchnia: mediana {d.lok_pow_uzyt.median():.1f} m2")
        print(f"  cena realna (2026) mediana: "
              f"{d.cena_realna_2026.median():,.0f} zł")

    # --- trend w czasie per ulica ---
    print("\n\nŚrednia cena za m2 wg roku:")
    print("-" * 75)
    for u in ULICE:
        d = df[df["ulica"] == u]
        grp = d.groupby("rok")["cena_za_m2"].agg(["mean", "size"])
        print(f"\n{u}:")
        for y, row in grp.iterrows():
            print(f"  {y}: {row['mean']:>7,.0f} zł/m2  (n={int(row['size'])})")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Mieszkania: ul. Trzebińska i Niepodległości (Chrzanów)",
                 fontsize=16, fontweight="bold")

    colors = {"Trzebińska": "#4C72B0", "Niepodległości": "#DD8452"}

    # 1. rozkład ceny za m2
    ax = axes[0, 0]
    for u in ULICE:
        d = df[df["ulica"] == u]
        ax.hist(d["cena_za_m2"], bins=25, alpha=0.55, label=u,
                color=colors[u], density=True)
    ax.set_xlabel("cena za m2 [zł]")
    ax.set_ylabel("gęstość")
    ax.set_title("Rozkład ceny za m2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. cena vs powierzchnia
    ax = axes[0, 1]
    for u in ULICE:
        d = df[df["ulica"] == u]
        ax.scatter(d["lok_pow_uzyt"], d["tran_cena_brutto"],
                   s=20, alpha=0.5, label=u, color=colors[u])
    ax.set_xlabel("powierzchnia [m2]")
    ax.set_ylabel("cena [zł]")
    ax.set_title("Cena vs powierzchnia")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. trend ceny za m2 w czasie
    ax = axes[1, 0]
    for u in ULICE:
        d = df[df["ulica"] == u]
        grp = d.groupby("rok")["cena_za_m2"].mean()
        ax.plot(grp.index, grp.values, "o-", label=u, color=colors[u],
                linewidth=2)
    ax.set_xlabel("rok")
    ax.set_ylabel("średnia cena za m2 [zł]")
    ax.set_title("Trend ceny za m2 (nominalny)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. boxplot ceny za m2
    ax = axes[1, 1]
    data = [df[df["ulica"] == u]["cena_za_m2"].values for u in ULICE]
    bp = ax.boxplot(data, tick_labels=ULICE, patch_artist=True)
    for patch, u in zip(bp["boxes"], ULICE):
        patch.set_facecolor(colors[u])
        patch.set_alpha(0.6)
    ax.set_ylabel("cena za m2 [zł]")
    ax.set_title("Rozrzut ceny za m2")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "ulice_analiza.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()
