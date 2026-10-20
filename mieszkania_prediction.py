"""
Predykcja ceny mieszkania na podstawie 1203_transakcje_ceny.gpkg.

Dane: Rejestr Cen Nieruchomości (RCN), warstwa 'transakcje_lokale'.
Cel: tran_cena_brutto (cena transakcji brutto).
Model: Hist Gradient Boosting. Metryka: RMSE.

Uwaga: GeoPackage to baza SQLite - czytamy przez sqlite3/pandas.
"""

import sqlite3
import struct
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

GPKG = "1203_transakcje_ceny.gpkg"


# ----------------------------------------------------------------------
# 0. Parser geometrii WKB (GeoPackage, EPSG:2180)
# ----------------------------------------------------------------------
def parse_geom(blob):
    """Zwraca (x, y) centroidu z bloba geometrii GeoPackage."""
    if blob is None:
        return None, None
    flags = blob[3]
    env = (flags >> 1) & 0x07
    off = 8
    if env == 1:
        off += 32
    elif env in (2, 3):
        off += 48
    elif env == 4:
        off += 64
    bo = blob[off]
    endian = "<" if bo == 1 else ">"
    gtype = struct.unpack(endian + "I", blob[off + 1:off + 5])[0]
    x, y = struct.unpack(endian + "dd", blob[off + 5:off + 21])
    return x, y


# ----------------------------------------------------------------------
# 1. Wczytanie danych z GeoPackage
# ----------------------------------------------------------------------
def load_data(path: str = GPKG) -> pd.DataFrame:
    con = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM transakcje_lokale", con)
    con.close()

    # sparsuj geometrię -> współrzędne x, y
    coords = df["geometry"].apply(parse_geom)
    df["x"] = [c[0] for c in coords]
    df["y"] = [c[1] for c in coords]

    return df


# ----------------------------------------------------------------------
# 2. Czyszczenie i inżynieria cech
# ----------------------------------------------------------------------
def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    def step(mask, label):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        if verbose:
            print(f"  {label:<45} -{before - len(df):>5}  (zostaje {len(df)})")

    if verbose:
        print("Czyszczenie danych:")

    # tylko mieszkania
    step(df["lok_funkcja"] == "mieszkalna", "tylko mieszkania")

    # cel: cena transakcji brutto
    step(df["tran_cena_brutto"].notna(), "cena transakcji istnieje")
    step(df["tran_cena_brutto"] > 0, "cena > 0")

    # powierzchnia użytkowa
    step(df["lok_pow_uzyt"].notna(), "powierzchnia istnieje")
    step(df["lok_pow_uzyt"] > 0, "powierzchnia > 0")

    # --- FILTROWANIE OUTLIERÓW ---
    # rozsądny zakres powierzchni mieszkania
    step(df["lok_pow_uzyt"].between(15, 200), "powierzchnia 15-200 m2")

    # rozsądna liczba izb
    step(df["lok_liczba_izb"].between(1, 10), "liczba izb 1-10")

    # cena za m2 w rozsądnym zakresie (rynek wtórny/pierwotny)
    df["cena_za_m2"] = df["tran_cena_brutto"] / df["lok_pow_uzyt"]
    step(df["cena_za_m2"].between(1000, 20000), "cena za m2 1000-20000 zł")

    # odrzuć skrajne ceny bezwzględne
    step(df["tran_cena_brutto"].between(20_000, 2_000_000),
         "cena 20 tys. - 2 mln zł")

    # data -> rok i miesiąc
    df["dok_data"] = pd.to_datetime(df["dok_data"], errors="coerce")
    step(df["dok_data"].notna(), "poprawna data")
    df["rok"] = df["dok_data"].dt.year
    df["miesiac"] = df["dok_data"].dt.month

    # odrzuć błędne daty (przyszłość / bardzo stare)
    step(df["rok"].between(2000, 2026), "rok transakcji 2000-2026")

    # miasto z adresu (MSC:...)
    df["miasto"] = df["lok_adres"].str.extract(r"MSC:([^;]+)")
    df["miasto"] = df["miasto"].fillna("nieznane")

    # --- CECHY PRZESTRZENNE ---
    # odległość od centrum miasta (centroid = mediana współrzędnych w mieście)
    df["x"] = df["x"].fillna(df["x"].median())
    df["y"] = df["y"].fillna(df["y"].median())

    centra = df.groupby("miasto")[["x", "y"]].median()
    df = df.join(centra, on="miasto", rsuffix="_centrum")
    df["odl_centrum"] = np.sqrt(
        (df["x"] - df["x_centrum"]) ** 2 +
        (df["y"] - df["y_centrum"]) ** 2)

    # liczba izb i kondygnacja - uzupełnij medianą
    df["lok_liczba_izb"] = df["lok_liczba_izb"].fillna(
        df["lok_liczba_izb"].median())
    df["lok_nr_kond"] = df["lok_nr_kond"].fillna(
        df["lok_nr_kond"].median())

    if verbose:
        print(f"  {'RAZEM':<45} -{n0 - len(df):>5}  (zostaje {len(df)})")

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. Podział train/val/test
# ----------------------------------------------------------------------
def split_data(df: pd.DataFrame, seed: int = 2):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    return (
        df.iloc[idx[:n_train]].reset_index(drop=True),
        df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True),
        df.iloc[idx[n_train + n_val:]].reset_index(drop=True),
    )


# ----------------------------------------------------------------------
# 4. Przygotowanie macierzy cech
# ----------------------------------------------------------------------
NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond", "rok", "miesiac",
           "x", "y", "odl_centrum"]
CATEGORICAL = ["miasto"]


def build_cat_map(df_train):
    return {c: list(df_train[c].value_counts().head(10).index)
            for c in CATEGORICAL}


def prepare_X(df, categorical):
    df = df.copy()
    features = list(NUMERIC)
    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)
    return df[features].fillna(0).values


def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())


# ----------------------------------------------------------------------
# 5. Główny przepływ
# ----------------------------------------------------------------------
def main():
    df = load_data()
    print(f"Wczytano {len(df)} transakcji lokali")

    df = clean(df)
    print(f"Po czyszczeniu (mieszkania z ceną): {len(df)}")
    print(f"Zakres cen: {df.tran_cena_brutto.min():,.0f} - "
          f"{df.tran_cena_brutto.max():,.0f}")
    print(f"Średnia cena za m2: {df.cena_za_m2.mean():,.0f} zł\n")

    df_train, df_val, df_test = split_data(df)

    y_train = np.log1p(df_train["tran_cena_brutto"].values)
    y_val = np.log1p(df_val["tran_cena_brutto"].values)
    y_test = np.log1p(df_test["tran_cena_brutto"].values)

    categorical = build_cat_map(df_train)

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print("Predykcja ceny mieszkania:")
    print("-" * 60)
    print(f"RMSE(val)  = {rmse(y_val, y_pred_val):.4f}")
    print(f"RMSE(test) = {rmse(y_test, y_pred_test):.4f}")

    # RMSE w złotówkach
    y_test_pln = np.expm1(y_test)
    y_pred_pln = np.expm1(y_pred_test)
    rmse_pln = np.sqrt(((y_test_pln - y_pred_pln) ** 2).mean())
    print(f"RMSE(test) w złotówkach = {rmse_pln:,.0f} zł")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Predykcja ceny mieszkania (RCN, rejon 1203)",
                 fontsize=16, fontweight="bold")

    # 1. predykcja vs rzeczywistość (log)
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywista log-cena")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title("Predykcja vs rzeczywistość")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. nakładka rozkładów
    ax = axes[0, 1]
    ax.hist(y_test, bins=50, alpha=0.55, label="prawdziwe",
            color="#4C72B0", density=True)
    ax.hist(y_pred_test, bins=50, alpha=0.55, label="przewidywane",
            color="#DD8452", density=True)
    ax.set_xlabel("log-cena")
    ax.set_ylabel("gęstość")
    ax.set_title("Nakładka rozkładów cen")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. cena vs powierzchnia (kolor = predykcja)
    ax = axes[1, 0]
    sc = ax.scatter(df_test["lok_pow_uzyt"], y_test_pln,
                    c=y_pred_pln, s=12, alpha=0.5, cmap="viridis")
    ax.set_xlabel("powierzchnia użytkowa [m2]")
    ax.set_ylabel("cena rzeczywista [zł]")
    ax.set_title("Cena vs powierzchnia (kolor = predykcja)")
    plt.colorbar(sc, ax=ax, label="predykcja [zł]")
    ax.grid(alpha=0.3)

    # 4. rozkład reszt
    ax = axes[1, 1]
    residuals = y_test - y_pred_test
    ax.hist(residuals, bins=50, color="#55A868", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("reszta (rzeczywista - przewidywana)")
    ax.set_ylabel("liczba próbek")
    ax.set_title(f"Rozkład reszt\nśrednia={residuals.mean():.3f}, "
                 f"std={residuals.std():.3f}")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "mieszkania_prediction.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # ------------------------------------------------------------------
    # Wykres przestrzenny (mapa)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(11, 9))
    sc2 = ax2.scatter(
        df_test["x"], df_test["y"],
        c=df_test["cena_za_m2"], s=18, alpha=0.7, cmap="plasma")
    ax2.set_xlabel("x [m] (EPSG:2180)")
    ax2.set_ylabel("y [m] (EPSG:2180)")
    ax2.set_title("Rozkład przestrzenny cen mieszkań (kolor = cena za m2)")
    ax2.set_aspect("equal")
    plt.colorbar(sc2, ax=ax2, label="cena za m2 [zł]")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    out2 = "mieszkania_mapa.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"Zapisano mapę: {out2}")

    # przykładowa predykcja
    print("\nPrzykład (wiersz 20):")
    print(f"  powierzchnia: {df_test['lok_pow_uzyt'].iloc[20]:.1f} m2, "
          f"izby: {df_test['lok_liczba_izb'].iloc[20]:.0f}, "
          f"miasto: {df_test['miasto'].iloc[20]}")
    print(f"  przewidywana cena: {y_pred_pln[20]:,.0f} zł")
    print(f"  rzeczywista cena : {y_test_pln[20]:,.0f} zł")


if __name__ == "__main__":
    main()
