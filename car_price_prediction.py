"""
Predykcja ceny samochodu na podstawie data.csv.

Model: regresja liniowa (normal equation) z regularyzacją (ridge).
Cel: log1p(MSRP), metryka: RMSE.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Wczytanie i przygotowanie danych
# ----------------------------------------------------------------------
def load_data(path: str = "data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalizacja nazw kolumn
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # normalizacja wartości tekstowych
    for col in df.dtypes[df.dtypes == "object"].index:
        df[col] = df[col].str.lower().str.replace(" ", "_")

    return df


# ----------------------------------------------------------------------
# 2. Podział na zbiory train / val / test
# ----------------------------------------------------------------------
def split_data(df: pd.DataFrame, seed: int = 2):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    return df_train, df_val, df_test


# ----------------------------------------------------------------------
# 3. Inżynieria cech
# ----------------------------------------------------------------------
BASE = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]

CATEGORICAL_COLUMNS = [
    "make", "model", "engine_fuel_type", "driven_wheels",
    "market_category", "vehicle_size", "vehicle_style",
]


def build_categorical_map(df_train: pd.DataFrame) -> dict:
    """Zwraca słownik: kolumna -> lista najczęstszych wartości."""
    categorical = {}
    for c in CATEGORICAL_COLUMNS:
        categorical[c] = list(df_train[c].value_counts().head().index)
    return categorical


def prepare_X(df: pd.DataFrame, categorical: dict) -> np.ndarray:
    df = df.copy()

    # wiek samochodu
    df["age"] = 2017 - df["year"]
    features = BASE + ["age"]

    # liczba drzwi jako one-hot
    for v in [2, 3, 4]:
        col = f"num_doors_{v}"
        df[col] = (df["number_of_doors"] == v).astype(int)
        features.append(col)

    # zmienne kategoryczne jako one-hot
    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)

    df_num = df[features].fillna(0)
    return df_num.values


# ----------------------------------------------------------------------
# 4. Trening modelu (normal equation + regularyzacja)
# ----------------------------------------------------------------------
def train_linear_regression_reg(X: np.ndarray, y: np.ndarray, r: float = 0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# ----------------------------------------------------------------------
# 5. Metryka RMSE
# ----------------------------------------------------------------------
def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    se = (y - y_pred) ** 2
    return np.sqrt(se.mean())


# ----------------------------------------------------------------------
# 6. Główny przepływ
# ----------------------------------------------------------------------
def main():
    df = load_data("data.csv")
    df_train, df_val, df_test = split_data(df)

    # logarytmujemy cenę (target)
    y_train = np.log1p(df_train["msrp"].values)
    y_val = np.log1p(df_val["msrp"].values)
    y_test = np.log1p(df_test["msrp"].values)

    # usuwamy target z cech
    for d in (df_train, df_val, df_test):
        del d["msrp"]

    # mapa kategorii budowana TYLKO na zbiorze treningowym
    categorical = build_categorical_map(df_train)

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    # --- tuning parametru regularyzacji ---
    print("Tuning r:")
    best_r, best_score = None, float("inf")
    for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        y_pred = w0 + X_val.dot(w)
        score = rmse(y_val, y_pred)
        print(f"  r={r:<8} w0={w0:8.4f}  RMSE={score:.4f}")
        if score < best_score:
            best_score, best_r = score, r

    print(f"\nNajlepsze r = {best_r}, RMSE(val) = {best_score:.4f}")

    # --- trening na train+val, ocena na test ---
    df_full = pd.concat([df_train, df_val]).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_val])

    X_full = prepare_X(df_full, categorical)
    w0, w = train_linear_regression_reg(X_full, y_full, r=best_r)

    y_pred_test = w0 + X_test.dot(w)
    print(f"RMSE(test) = {rmse(y_test, y_pred_test):.4f}")

    # --- predykcja dla pojedynczego samochodu ---
    car = df_test.iloc[20].to_dict()
    df_small = pd.DataFrame([car])
    X_small = prepare_X(df_small, categorical)

    y_pred_log = (w0 + X_small.dot(w))[0]
    print(f"\nPrzykład (wiersz 20):")
    print(f"  przewidywana cena : {np.expm1(y_pred_log):,.0f}")
    print(f"  rzeczywista cena  : {np.expm1(y_test[20]):,.0f}")

    return w0, w, categorical


if __name__ == "__main__":
    main()
