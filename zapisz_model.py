"""
Zapis optymalnego modelu wyceny mieszkań dla ul. Trzebińskiej.

Optymalny zestaw (z eksperymentu przyrostowego): Trzebińska + 5 ulic
(Grzybowa, Broniewskiego, Mieszka I, Wyszyńskiego, Sienkiewicza).

Ceny treningowe są SKORYGOWANE O INFLACJĘ (sprowadzone do wartości 2026),
żeby model wyceniał mieszkanie "tu i teraz" - bez trendu czasowego.

Zapisuje model + metadane do pliku model_trzebinska.pkl.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import (
    load_data, clean, build_cat_map, rmse,
)
from inflacja_korekta import wspolczynnik_deflacji


ULICA_TARGET = "Trzebińska"
ULICE_OPTYMALNE = ["Grzybowa", "Broniewskiego", "Mieszka I", "Wyszyńskiego",
                   "Sienkiewicza"]

NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum"]
CATEGORICAL = ["miasto"]

MODEL_FILE = "model_trzebinska.pkl"


def prepare_X(df, categorical, numeric=NUMERIC):
    """Spójna z app.py funkcja budowy macierzy cech."""
    df = df.copy()
    features = list(numeric)
    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)
    return df[features].fillna(0).values


def main():
    df = load_data()
    df = clean(df, verbose=False)

    # korekta cen o inflację -> wartości realne 2026
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_realna_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]

    # zbiór: Trzebińska + optymalne ulice
    mask = df["lok_adres"].str.contains(
        "|".join([ULICA_TARGET] + ULICE_OPTYMALNE), case=False, na=False)
    df_opt = df[mask].copy()

    # test = 20% Trzebińskiej (do oceny)
    df_trz = df_opt[df_opt["lok_adres"].str.contains(
        ULICA_TARGET, case=False, na=False)].copy()
    n = len(df_trz)
    n_test = int(n * 0.2)
    idx = np.arange(n)
    np.random.seed(2)
    np.random.shuffle(idx)
    test_idx = idx[:n_test]

    df_test = df_trz.iloc[test_idx].reset_index(drop=True)
    df_train = df_opt.drop(df_trz.index[test_idx]).reset_index(drop=True)

    print(f"Trening: {len(df_train)} transakcji")
    print(f"Test (Trzebińska): {len(df_test)} transakcji")

    # cel: cena realna (skorygowana o inflację do 2026)
    y_train = np.log1p(df_train["cena_realna_2026"].values)
    y_test = np.log1p(df_test["cena_realna_2026"].values)

    categorical = build_cat_map(df_train)

    X_train = prepare_X(df_train, categorical)
    X_test = prepare_X(df_test, categorical)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    test_rmse = rmse(y_test, model.predict(X_test))
    test_pln = np.sqrt(
        ((np.expm1(y_test) - np.expm1(model.predict(X_test))) ** 2).mean())
    print(f"\nRMSE(test) = {test_rmse:.4f}  ({test_pln:,.0f} zł)")

    # centra ulic (do wyceny - mediana współrzędnych)
    centra = {}
    for ul in [ULICA_TARGET] + ULICE_OPTYMALNE:
        d = df_opt[df_opt["lok_adres"].str.contains(ul, case=False, na=False)]
        centra[ul] = {"x": float(d["x"].median()),
                      "y": float(d["y"].median()),
                      "n": int(len(d))}

    # zapis modelu + metadanych
    payload = {
        "model": model,
        "categorical": categorical,
        "numeric": NUMERIC,
        "centra": centra,
        "ulice": [ULICA_TARGET] + ULICE_OPTYMALNE,
        "rmse_test": test_rmse,
        "rmse_test_pln": test_pln,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "ceny_realne": True,  # model trenowany na cenach realnych 2026
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nZapisano model: {MODEL_FILE}")
    print(f"Ulice w modelu: {payload['ulice']}")
    print(f"Centra ulic:")
    for ul, c in centra.items():
        print(f"  {ul:<16} n={c['n']:>3}  x={c['x']:.0f}  y={c['y']:.0f}")


if __name__ == "__main__":
    main()
