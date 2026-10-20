"""
KROK 2: Wspólny moduł danych + protokół ewaluacji.

- Jednolita inżynieria cech (bez leakage).
- Trzy podziały: losowy, grupowy (po ulicy), czasowy.
- Metryki: RMSE, MAE, R² — w logarytmie ORAZ w PLN.
- Ustalone ziarna losowości.
"""

import re
import numpy as np
import pandas as pd

from mieszkania_prediction import load_data, clean
from inflacja_korekta import wspolczynnik_deflacji


SEED = 42

# cechy numeryczne (bez leakage - bez lok_cena_brutto, nier_cena_brutto)
NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum"]
CAT_FEATURE = "ulica"


def wyciagnij_ulice(adres):
    if not isinstance(adres, str):
        return "brak"
    m = re.search(r"UL:([^;]+)", adres)
    return m.group(1).strip() if m else "brak"


def wyciagnij_nr(adres):
    if not isinstance(adres, str):
        return np.nan
    m = re.search(r"NR_PORZ:([^;]+)", adres)
    if not m:
        return np.nan
    mm = re.search(r"\d+", m.group(1))
    return float(mm.group()) if mm else np.nan


def load_dataset(deduplicate: bool = True) -> pd.DataFrame:
    """Ładuje i przygotowuje dane z korektą inflacyjną i cechami."""
    df = clean(load_data(), verbose=False)

    # odrzuć błędne daty (rok < 2000)
    df = df[df["rok"].between(2000, 2026)].copy()

    # korekta inflacyjna -> cena realna 2026 (TARGET)
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]

    # ulica, numer porządkowy
    df["ulica"] = df["lok_adres"].apply(wyciagnij_ulice)
    df["nr_porz"] = df["lok_adres"].apply(wyciagnij_nr)

    # odległość od centrum Chrzanowa
    ch = df[df["lok_adres"].str.contains("Chrzanów", case=False, na=False)]
    cx, cy = ch["x"].median(), ch["y"].median()
    df["odl_centrum"] = np.sqrt((df["x"] - cx) ** 2 + (df["y"] - cy) ** 2)

    # dodatkowe cechy inżynieryjne
    df["piętro_bin"] = pd.cut(df["lok_nr_kond"], bins=[-2, 0, 2, 5, 99],
                              labels=["parter", "1-2", "3-5", "6+"])
    df["piętro_bin"] = df["piętro_bin"].astype(str)
    df["pow_bin"] = pd.cut(df["lok_pow_uzyt"],
                           bins=[0, 35, 45, 55, 65, 80, 999],
                           labels=["<35", "35-45", "45-55", "55-65",
                                   "65-80", "80+"]).astype(str)
    df["log_pow"] = np.log1p(df["lok_pow_uzyt"])
    df["izb_na_m2"] = df["lok_liczba_izb"] / df["lok_pow_uzyt"]
    df["maly_metraz"] = (df["lok_pow_uzyt"] < 40).astype(int)
    df["wysokie_pietro"] = (df["lok_nr_kond"] >= 4).astype(int)

    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(
            subset=["lok_adres", "tran_cena_brutto", "lok_pow_uzyt",
                    "dok_data"], keep="first").copy()
        print(f"Usunięto duplikatów: {before - len(df)}")

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Metryki
# ----------------------------------------------------------------------
def metrics(y_true_log, y_pred_log):
    """Zwraca RMSE/MAE/R2 w log oraz w PLN (po odwrotnej transformacji)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def r2(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return float(1 - ss_res / ss_tot)

    return {
        "rmse_log": rmse(y_true_log, y_pred_log),
        "mae_log": mae(y_true_log, y_pred_log),
        "r2_log": r2(y_true_log, y_pred_log),
        "rmse_pln": rmse(y_true, y_pred),
        "mae_pln": mae(y_true, y_pred),
        "r2_pln": r2(y_true, y_pred),
    }


# ----------------------------------------------------------------------
# Podziały danych
# ----------------------------------------------------------------------
def split_random(df, test_frac=0.2, val_frac=0.2, seed=SEED):
    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    tr = df.iloc[idx[:n - n_val - n_test]].reset_index(drop=True)
    va = df.iloc[idx[n - n_val - n_test:n - n_test]].reset_index(drop=True)
    te = df.iloc[idx[n - n_test:]].reset_index(drop=True)
    return tr, va, te


def split_group(df, group_col="ulica", test_frac=0.2, val_frac=0.2, seed=SEED):
    """Grupowy podział po ulicy - cała ulica trafia do jednego zbioru."""
    groups = np.array(df[group_col].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    te = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    va = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    tr = df[~df[group_col].isin(test_groups | val_groups)].reset_index(drop=True)
    return tr, va, te


def split_time(df, time_col="rok", test_frac=0.2, val_frac=0.2):
    """Czasowy podział - najnowsze dane do testu."""
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    tr = df.iloc[:n - n_val - n_test].reset_index(drop=True)
    va = df.iloc[n - n_val - n_test:n - n_test].reset_index(drop=True)
    te = df.iloc[n - n_test:].reset_index(drop=True)
    return tr, va, te


FEATURES = NUMERIC + [CAT_FEATURE]


def get_xy(df, target="cena_2026", features=None):
    features = features or FEATURES
    X = df[features].copy()
    y = np.log1p(df[target].values)
    return X, y


if __name__ == "__main__":
    df = load_dataset()
    print(f"Zbiór: {len(df)} wierszy, {df['ulica'].nunique()} ulic")
    print(f"Cechy: {FEATURES}")

    for name, splitter in [("random", split_random),
                           ("group", split_group),
                           ("time", split_time)]:
        tr, va, te = splitter(df)
        print(f"\n{name}: train={len(tr)}, val={len(va)}, test={len(te)}")

    print("\nWeryfikacja polskich znaków: Trzebińska, m², zł")