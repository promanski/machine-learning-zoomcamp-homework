"""
Predykcja zużycia paliwa (highway MPG / city mpg) na podstawie data.csv.

Model: Hist Gradient Boosting (najlepszy z porównania).
Cel: highway_mpg oraz city_mpg, metryka: RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from car_price_prediction import load_data, split_data, rmse


# cechy numeryczne używane do predykcji MPG
NUMERIC_FEATURES = [
    "year", "engine_hp", "engine_cylinders", "number_of_doors",
    "popularity", "msrp",
]

CATEGORICAL_COLUMNS = [
    "make", "model", "engine_fuel_type", "transmission_type",
    "driven_wheels", "market_category", "vehicle_size", "vehicle_style",
]


def build_categorical_map(df_train: pd.DataFrame) -> dict:
    categorical = {}
    for c in CATEGORICAL_COLUMNS:
        categorical[c] = list(df_train[c].value_counts().head().index)
    return categorical


def prepare_X(df: pd.DataFrame, categorical: dict) -> np.ndarray:
    df = df.copy()

    df["age"] = 2017 - df["year"]
    features = NUMERIC_FEATURES + ["age"]

    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)

    df_num = df[features].fillna(0)
    return df_num.values


def train_and_eval(target: str, df_train, df_val, df_test, categorical):
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print(f"[{target}]  RMSE(val)={rmse(y_val, y_pred_val):.4f}   "
          f"RMSE(test)={rmse(y_test, y_pred_test):.4f}")

    return model, y_test, y_pred_test


def main():
    df = load_data("data.csv")
    df_train, df_val, df_test = split_data(df)

    categorical = build_categorical_map(df_train)

    print("Predykcja zużycia paliwa (MPG):")
    print("-" * 60)

    model_hw, y_hw, pred_hw = train_and_eval(
        "highway_mpg", df_train, df_val, df_test, categorical)
    model_city, y_city, pred_city = train_and_eval(
        "city_mpg", df_train, df_val, df_test, categorical)

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Predykcja zużycia paliwa (MPG)",
                 fontsize=16, fontweight="bold")

    # 1. highway: predykcja vs rzeczywistość
    ax = axes[0, 0]
    ax.scatter(y_hw, pred_hw, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(y_hw.min(), pred_hw.min()), max(y_hw.max(), pred_hw.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywiste highway MPG")
    ax.set_ylabel("przewidywane highway MPG")
    ax.set_title("Highway MPG: predykcja vs rzeczywistość")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. city: predykcja vs rzeczywistość
    ax = axes[0, 1]
    ax.scatter(y_city, pred_city, s=8, alpha=0.3, color="#55A868")
    lims = [min(y_city.min(), pred_city.min()), max(y_city.max(), pred_city.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywiste city MPG")
    ax.set_ylabel("przewidywane city MPG")
    ax.set_title("City MPG: predykcja vs rzeczywistość")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. nakładka rozkładów highway
    ax = axes[1, 0]
    ax.hist(y_hw, bins=50, alpha=0.55, label="prawdziwe",
            color="#4C72B0", density=True)
    ax.hist(pred_hw, bins=50, alpha=0.55, label="przewidywane",
            color="#DD8452", density=True)
    ax.set_xlabel("highway MPG")
    ax.set_ylabel("gęstość")
    ax.set_title("Nakładka rozkładów (highway MPG)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. nakładka rozkładów city
    ax = axes[1, 1]
    ax.hist(y_city, bins=50, alpha=0.55, label="prawdziwe",
            color="#55A868", density=True)
    ax.hist(pred_city, bins=50, alpha=0.55, label="przewidywane",
            color="#C44E52", density=True)
    ax.set_xlabel("city MPG")
    ax.set_ylabel("gęstość")
    ax.set_title("Nakładka rozkładów (city MPG)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "mpg_prediction.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # przykładowa predykcja
    print("\nPrzykład (wiersz 20):")
    print(f"  highway MPG: przewidywane={pred_hw[20]:.1f}  "
          f"rzeczywiste={y_hw[20]:.1f}")
    print(f"  city MPG   : przewidywane={pred_city[20]:.1f}  "
          f"rzeczywiste={y_city[20]:.1f}")


if __name__ == "__main__":
    main()
