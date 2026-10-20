"""
KROK 1: Weryfikacja ewaluacji i wykrywanie data leakage.

Sprawdza:
  - duplikaty transakcji
  - potencjalny leakage (cena za m2 jako cecha, cena budynku/dzialki)
  - zakres i spójność danych
  - rozkład czasowy i przestrzenny (pod kątem grupowego podziału)
"""

import numpy as np
import pandas as pd
import re

from mieszkania_prediction import load_data, clean
from inflacja_korekta import wspolczynnik_deflacji


def main():
    df_raw = load_data()
    print(f"Surowych wierszy (lokale): {len(df_raw)}")

    # --- braki w kluczowych kolumnach ---
    print("\nBraki w kluczowych kolumnach (surowych):")
    key = ["tran_cena_brutto", "lok_pow_uzyt", "lok_liczba_izb",
           "lok_nr_kond", "lok_cena_brutto", "lok_pow_przyn",
           "nier_cena_brutto", "lok_adres"]
    for c in key:
        print(f"  {c:<22} {df_raw[c].isnull().sum():>5} / {len(df_raw)}")

    # --- duplikaty (po pełnym rekordzie i po kluczu) ---
    print("\nDuplikaty:")
    print(f"  pełne duplikaty wierszy: "
          f"{df_raw.drop(columns='geometry').duplicated().sum()}")
    dup_key = df_raw.duplicated(
        subset=["lok_adres", "tran_cena_brutto", "lok_pow_uzyt",
                "dok_data"], keep=False).sum()
    print(f"  duplikaty (adres+cena+pow+data): {dup_key}")

    # --- leakage check: kolumny z ceną zależne od targetu ---
    print("\nPotencjalny LEAKAGE (kolumny cenowe):")
    d = df_raw[df_raw["lok_funkcja"] == "mieszkalna"].copy()
    d = d[d["tran_cena_brutto"].notna() & (d["tran_cena_brutto"] > 0)]
    for c in ["lok_cena_brutto", "nier_cena_brutto"]:
        sub = d[d[c].notna()]
        if len(sub):
            corr = np.corrcoef(np.log1p(sub[c]), np.log1p(sub["tran_cena_brutto"]))[0, 1]
            same = np.isclose(sub[c], sub["tran_cena_brutto"]).mean()
            print(f"  {c}: n={len(sub)}, corr_z_targetem={corr:.3f}, "
                  f"identyczne_z_targetem={same:.1%}  <- NIE używać!")

    # --- czyszczenie (jak w pipeline) ---
    df = clean(df_raw, verbose=False)
    print(f"\nPo czyszczeniu: {len(df)}")

    # --- korekta inflacyjna ---
    df["wsp"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_2026"] = df["tran_cena_brutto"] * df["wsp"]

    # --- rozkład czasowy ---
    print("\nRozkład transakcji po latach:")
    yc = df["rok"].value_counts().sort_index()
    for y, n in yc.items():
        print(f"  {y}: {n:>4}")

    # --- rozkład cen (outliery w oryginalnej skali) ---
    print("\nRozkład ceny 2026 (percentyle):")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  {p:>2}%: {np.percentile(df['cena_2026'], p):>12,.0f} zł")

    print("\nRozkład ceny/m2 2026 (percentyle):")
    m2 = df["cena_2026"] / df["lok_pow_uzyt"]
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  {p:>2}%: {np.percentile(m2, p):>10,.0f} zł/m2")

    # --- spójność jednostek (dziwne powierzchnie) ---
    print("\nPodejrzane powierzchnie:")
    print(f"  < 15 m2: {(df['lok_pow_uzyt'] < 15).sum()}")
    print(f"  > 200 m2: {(df['lok_pow_uzyt'] > 200).sum()}")
    print(f"  max: {df['lok_pow_uzyt'].max():.1f} m2")

    # --- ulica jako grupa ---
    def ul(a):
        if not isinstance(a, str):
            return "brak"
        m = re.search(r"UL:([^;]+)", a)
        return m.group(1).strip() if m else "brak"
    df["ulica"] = df["lok_adres"].apply(ul)
    print(f"\nUnikalnych ulic (grupy): {df['ulica'].nunique()}")
    print(f"Ulic z 1 transakcją: {(df['ulica'].value_counts() == 1).sum()}")
    print(f"Ulic z >= 10 transakcji: {(df['ulica'].value_counts() >= 10).sum()}")

    # --- czy są daty (do podziału czasowego) ---
    df["dok_data"] = pd.to_datetime(df_raw["dok_data"], errors="coerce")
    print(f"\nZakres dat: {df['dok_data'].min()} ... {df['dok_data'].max()}")


if __name__ == "__main__":
    main()