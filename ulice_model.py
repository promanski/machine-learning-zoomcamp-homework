"""
Dedykowany model wyceny mieszkań dla ulic Trzebińska i Niepodległości.

Trenuje model TYLKO na transakcjach z tych dwóch ulic i porównuje
z modelem ogólnym (cały powiat). Sprawdza, czy zawężenie danych pomaga.

Cel: tran_cena_brutto (log1p). Metryka: RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import (
    load_data, clean, split_data, build_cat_map, prepare_X, rmse,
)


ULICE = ["Trzebińska", "Niepodległości"]

NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum", "rok"]
CATEGORICAL = ["miasto"]


def train_eval(df_train, df_val, df_test, categorical, label):
    y_train = np.log1p(df_train["tran_cena_brutto"].values)
    y_val = np.log1p(df_val["tran_cena_brutto"].values)
    y_test = np.log1p(df_test["tran_cena_brutto"].values)

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    v = rmse(y_val, model.predict(X_val))
    t = rmse(y_test, model.predict(X_test))
    t_pln = np.sqrt(
        ((np.expm1(y_test) - np.expm1(model.predict(X_test))) ** 2).mean())

    print(f"{label:<32} RMSE(val)={v:.4f}  RMSE(test)={t:.4f}  "
          f"({t_pln:,.0f} zł)")
    return model, v, t, t_pln


def main():
    df = load_data()
    df = clean(df, verbose=False)

    # --- model OGÓLNY (cały powiat) ---
    df_tr, df_va, df_te = split_data(df)
    cat_all = build_cat_map(df_tr)

    print("Porównanie modeli:")
    print("-" * 78)
    m_all, v_all, t_all, pln_all = train_eval(
        df_tr, df_va, df_te, cat_all, "OGÓLNY (cały powiat)")

    # --- model DEDYKOWANY (tylko 2 ulice) ---
    mask = df["lok_adres"].str.contains("|".join(ULICE), case=False, na=False)
    df_st = df[mask].copy()
    print(f"\nTransakcje na wybranych ulicach: {len(df_st)}")

    df_tr2, df_va2, df_te2 = split_data(df_st)
    cat_st = build_cat_map(df_tr2)

    m_st, v_st, t_st, pln_st = train_eval(
        df_tr2, df_va2, df_te2, cat_st, "DEDYKOWANY (2 ulice)")

    # --- model dedykowany oceniany na CAŁYM zbiorze testowym ---
    # (żeby porównać jabłka do jabłek)
    print("\nOcena modelu dedykowanego na zbiorze testowym CAŁEGO powiatu:")
    y_test_all = np.log1p(df_te["tran_cena_brutto"].values)
    X_test_all = prepare_X(df_te, cat_st)
    pred = m_st.predict(X_test_all)
    t_cross = rmse(y_test_all, pred)
    t_cross_pln = np.sqrt(
        ((np.expm1(y_test_all) - np.expm1(pred)) ** 2).mean())
    print(f"{'DEDYKOWANY na teście ogólnym':<32} RMSE(test)={t_cross:.4f}  "
          f"({t_cross_pln:,.0f} zł)")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Model dedykowany: ul. Trzebińska i Niepodległości",
                 fontsize=15, fontweight="bold")

    # 1. porównanie RMSE
    ax = axes[0]
    labels = ["OGÓLNY\n(cały powiat)", "DEDYKOWANY\n(2 ulice)"]
    vals = [t_all, t_st]
    plns = [pln_all, pln_st]
    x = np.arange(2)
    w = 0.38
    ax.bar(x - w / 2, vals, w, label="RMSE (log)", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, plns, w, label="RMSE (zł)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE (log-cena)", color="#4C72B0")
    ax2.set_ylabel("RMSE [zł]", color="#DD8452")
    ax.set_title("Ogólny vs dedykowany")
    ax.grid(axis="y", alpha=0.3)
    for i, (v, p) in enumerate(zip(vals, plns)):
        ax.text(i - w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        ax2.text(i + w / 2, p + 1000, f"{p/1000:.0f}k", ha="center", fontsize=9)

    # 2. predykcja vs rzeczywistość (dedykowany)
    ax = axes[1]
    y_test_st = np.log1p(df_te2["tran_cena_brutto"].values)
    pred_st = m_st.predict(prepare_X(df_te2, cat_st))
    ax.scatter(y_test_st, pred_st, s=20, alpha=0.5, color="#55A868")
    lims = [min(y_test_st.min(), pred_st.min()),
            max(y_test_st.max(), pred_st.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywista log-cena")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title("Model dedykowany: predykcja vs rzeczywistość")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "ulice_model.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # --- przykładowa wycena 65 m2 na parterze ---
    print("\nPrzykład: 65 m2, parter, Chrzanów")
    for ul in ULICE:
        d = df_st[df_st["lok_adres"].str.contains(ul, case=False, na=False)]
        x_c = d["x"].median()
        y_c = d["y"].median()
        row = {
            "lok_pow_uzyt": 65, "lok_liczba_izb": 3, "lok_nr_kond": 0,
            "x": x_c, "y": y_c, "odl_centrum": 0.0, "rok": 2026,
            "miesiac": 6, "miasto": "Chrzanów",
        }
        X = prepare_X(pd.DataFrame([row]), cat_st)
        cena = float(np.expm1(m_st.predict(X)[0]))
        print(f"  {ul:<16} -> {cena:,.0f} zł  ({cena/65:,.0f} zł/m2)")


if __name__ == "__main__":
    main()
