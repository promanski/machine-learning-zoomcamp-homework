"""
Porównanie modeli predykcji ceny mieszkania BEZ trendu czasowego (rok).

Cel: sprawdzić, jak dobrze da się wycenić mieszkanie wyłącznie na podstawie
cech nieruchomości (bez informacji o roku transakcji).

Porównanie: z rokiem vs bez roku.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from mieszkania_prediction import (
    load_data, clean, split_data, build_cat_map, rmse,
)


# cechy numeryczne BEZ roku i miesiąca
NUMERIC_NO_TIME = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
                   "x", "y", "odl_centrum"]
CATEGORICAL = ["miasto"]


def prepare_X(df, categorical, numeric):
    df = df.copy()
    features = list(numeric)
    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)
    return df[features].fillna(0).values


def run(models, X_train, y_train, X_val, y_val, X_test, y_test):
    names, val_scores, test_scores, test_pln = [], [], [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        v = rmse(y_val, model.predict(X_val))
        t = rmse(y_test, model.predict(X_test))
        t_pln = np.sqrt(
            ((np.expm1(y_test) - np.expm1(model.predict(X_test))) ** 2).mean())
        names.append(name)
        val_scores.append(v)
        test_scores.append(t)
        test_pln.append(t_pln)
    return names, val_scores, test_scores, test_pln


def main():
    df = load_data()
    df = clean(df, verbose=False)
    print(f"Mieszkania po czyszczeniu: {len(df)}\n")

    df_train, df_val, df_test = split_data(df)

    y_train = np.log1p(df_train["tran_cena_brutto"].values)
    y_val = np.log1p(df_val["tran_cena_brutto"].values)
    y_test = np.log1p(df_test["tran_cena_brutto"].values)

    categorical = build_cat_map(df_train)

    def make_models():
        return {
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=3,
                random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.1, random_state=42),
        }

    # --- WARIANT A: z rokiem ---
    numeric_with = NUMERIC_NO_TIME + ["rok", "miesiac"]
    X_train_a = prepare_X(df_train, categorical, numeric_with)
    X_val_a = prepare_X(df_val, categorical, numeric_with)
    X_test_a = prepare_X(df_test, categorical, numeric_with)

    print("WARIANT A: z rokiem (trend czasowy)")
    print("-" * 70)
    names, val_a, test_a, pln_a = run(
        make_models(), X_train_a, y_train, X_val_a, y_val, X_test_a, y_test)
    for n, v, t, p in zip(names, val_a, test_a, pln_a):
        print(f"{n:<24} RMSE(val)={v:.4f}  RMSE(test)={t:.4f}  ({p:,.0f} zł)")

    # --- WARIANT B: bez roku ---
    X_train_b = prepare_X(df_train, categorical, NUMERIC_NO_TIME)
    X_val_b = prepare_X(df_val, categorical, NUMERIC_NO_TIME)
    X_test_b = prepare_X(df_test, categorical, NUMERIC_NO_TIME)

    print("\nWARIANT B: bez roku (tylko cechy nieruchomości)")
    print("-" * 70)
    names, val_b, test_b, pln_b = run(
        make_models(), X_train_b, y_train, X_val_b, y_val, X_test_b, y_test)
    for n, v, t, p in zip(names, val_b, test_b, pln_b):
        print(f"{n:<24} RMSE(val)={v:.4f}  RMSE(test)={t:.4f}  ({p:,.0f} zł)")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Wpływ trendu czasowego (rok) na predykcję ceny mieszkania",
                 fontsize=15, fontweight="bold")

    x = np.arange(len(names))
    w = 0.38

    # 1. RMSE (log) - z rokiem vs bez
    ax = axes[0]
    ax.bar(x - w / 2, test_a, w, label="z rokiem", color="#4C72B0")
    ax.bar(x + w / 2, test_b, w, label="bez roku", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (test, log-cena)")
    ax.set_title("RMSE: z rokiem vs bez roku")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(test_a, test_b)):
        ax.text(i - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8)

    # 2. RMSE w złotówkach
    ax = axes[1]
    ax.bar(x - w / 2, pln_a, w, label="z rokiem", color="#4C72B0")
    ax.bar(x + w / 2, pln_b, w, label="bez roku", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (test) [zł]")
    ax.set_title("RMSE w złotówkach: z rokiem vs bez roku")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(pln_a, pln_b)):
        ax.text(i - w / 2, a + 1000, f"{a/1000:.0f}k", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1000, f"{b/1000:.0f}k", ha="center", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "mieszkania_bez_roku.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # podsumowanie różnicy
    print("\nRóżnica (bez roku - z rokiem), RMSE w zł:")
    for n, a, b in zip(names, pln_a, pln_b):
        print(f"  {n:<24} {b - a:+,.0f} zł")


if __name__ == "__main__":
    main()
