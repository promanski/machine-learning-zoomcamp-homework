"""
Eksperyment: czy dotrenowanie modelu o dodatkowe ulice Chrzanowa
polepsza predykcję na ul. Trzebińskiej?

Porównanie:
  A) model tylko na Trzebińskiej
  B) model na Trzebińskiej + Grzybowa, Broniewskiego, Mieszka I, Wyszyńskiego

Oba oceniane na TYM SAMYM zbiorze testowym (transakcje z Trzebińskiej).
Cel: tran_cena_brutto (log1p). Metryka: RMSE.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import (
    load_data, clean, build_cat_map, prepare_X, rmse,
)


ULICA_TARGET = "Trzebińska"
ULICE_DODATKOWE = ["Grzybowa", "Broniewskiego", "Mieszka I", "Wyszyńskiego",
                   "Sienkiewicza", "Jordana", "Pęckowskiego", "Orkana"]

NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum", "rok"]
CATEGORICAL = ["miasto"]


def split(df, seed=2):
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


def train_eval(df_train, df_test, categorical, label):
    y_train = np.log1p(df_train["tran_cena_brutto"].values)
    y_test = np.log1p(df_test["tran_cena_brutto"].values)

    X_train = prepare_X(df_train, categorical)
    X_test = prepare_X(df_test, categorical)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    t = rmse(y_test, pred)
    t_pln = np.sqrt(
        ((np.expm1(y_test) - np.expm1(pred)) ** 2).mean())

    print(f"{label:<40} RMSE(test)={t:.4f}  ({t_pln:,.0f} zł)")
    return model, t, t_pln, y_test, pred


def main():
    df = load_data()
    df = clean(df, verbose=False)

    # --- zbiór Trzebińskiej ---
    df_trz = df[df["lok_adres"].str.contains(ULICA_TARGET, case=False,
                                             na=False)].copy()
    print(f"Trzebińska: {len(df_trz)} transakcji")

    # --- zbiór rozszerzony ---
    mask_ext = df["lok_adres"].str.contains(
        "|".join([ULICA_TARGET] + ULICE_DODATKOWE), case=False, na=False)
    df_ext = df[mask_ext].copy()
    print(f"Trzebińska + {len(ULICE_DODATKOWE)} ulic: {len(df_ext)} transakcji\n")

    # --- wspólny podział: test = 20% Trzebińskiej (ten sam dla obu modeli) ---
    df_trz_tr, df_trz_va, df_trz_te = split(df_trz)

    # zbiór rozszerzony: trening = wszystko oprócz testu Trzebińskiej
    test_idx = df_trz_te.index  # nie używamy - budujemy po adresie
    # trening rozszerzony: cała Trzebińska (train+val) + dodatkowe ulice
    df_trz_train_full = pd.concat([df_trz_tr, df_trz_va]).reset_index(drop=True)
    df_dod = df_ext[~df_ext["lok_adres"].str.contains(
        ULICA_TARGET, case=False, na=False)].copy()
    df_ext_train = pd.concat([df_trz_train_full, df_dod]).reset_index(drop=True)

    print("Porównanie (ocena na tym samym teście Trzebińskiej):")
    print("-" * 78)

    # A) tylko Trzebińska
    cat_a = build_cat_map(df_trz_tr)
    m_a, t_a, pln_a, y_te, pred_a = train_eval(
        df_trz_tr, df_trz_te, cat_a, "A) tylko Trzebińska")

    # B) Trzebińska + dodatkowe ulice
    cat_b = build_cat_map(df_ext_train)
    m_b, t_b, pln_b, _, pred_b = train_eval(
        df_ext_train, df_trz_te, cat_b,
        f"B) Trzebińska + {len(ULICE_DODATKOWE)} ulic")

    # --- porównanie przyrostowe: 0, 2, 4, 6, 8 ulic ---
    print("\nPorównanie przyrostowe (ile ulic dodawać):")
    print("-" * 78)
    incremental = []
    for k in range(0, len(ULICE_DODATKOWE) + 1):
        ulice_k = ULICE_DODATKOWE[:k]
        if k == 0:
            df_tr_k = df_trz_train_full
        else:
            mask_k = df["lok_adres"].str.contains(
                "|".join([ULICA_TARGET] + ulice_k), case=False, na=False)
            df_k = df[mask_k].copy()
            df_dod_k = df_k[~df_k["lok_adres"].str.contains(
                ULICA_TARGET, case=False, na=False)].copy()
            df_tr_k = pd.concat([df_trz_train_full, df_dod_k]).reset_index(
                drop=True)

        cat_k = build_cat_map(df_tr_k)
        y_tr_k = np.log1p(df_tr_k["tran_cena_brutto"].values)
        X_tr_k = prepare_X(df_tr_k, cat_k)
        X_te_k = prepare_X(df_trz_te, cat_k)
        mk = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=42)
        mk.fit(X_tr_k, y_tr_k)
        pred_k = mk.predict(X_te_k)
        t_k = rmse(y_te, pred_k)
        pln_k = np.sqrt(
            ((np.expm1(y_te) - np.expm1(pred_k)) ** 2).mean())
        incremental.append((k, len(df_tr_k), t_k, pln_k))
        print(f"  +{k} ulic (n={len(df_tr_k):>4}): "
              f"RMSE={t_k:.4f}  ({pln_k:,.0f} zł)")

    best_k = min(incremental, key=lambda r: r[2])[0]
    print(f"\nNajlepsza liczba dodatkowych ulic: {best_k}")

    print(f"\nRóżnica (B - A): RMSE {t_b - t_a:+.4f}  "
          f"({pln_b - pln_a:+,.0f} zł)")
    if t_b < t_a:
        print("=> Dotrenowanie POPRAWIŁO predykcję na Trzebińskiej")
    else:
        print("=> Dotrenowanie NIE poprawiło predykcji na Trzebińskiej")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Dotrenowanie o dodatkowe ulice - wpływ na Trzebińską",
                 fontsize=15, fontweight="bold")

    # 1. RMSE
    ax = axes[0]
    labels = ["A) tylko\nTrzebińska", f"B) Trzebińska\n+ {len(ULICE_DODATKOWE)} ulic"]
    vals = [t_a, t_b]
    bars = ax.bar(labels, vals, color=["#4C72B0", "#55A868"])
    ax.set_ylabel("RMSE (test, log-cena)")
    ax.set_title("RMSE na teście Trzebińskiej")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", fontsize=10)

    # 2. predykcja vs rzeczywistość (A)
    ax = axes[1]
    ax.scatter(y_te, pred_a, s=20, alpha=0.5, color="#4C72B0")
    lims = [min(y_te.min(), pred_a.min()), max(y_te.max(), pred_a.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("rzeczywista log-cena")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title(f"A) tylko Trzebińska (RMSE={t_a:.3f})")
    ax.grid(alpha=0.3)

    # 3. predykcja vs rzeczywistość (B)
    ax = axes[2]
    ax.scatter(y_te, pred_b, s=20, alpha=0.5, color="#55A868")
    lims = [min(y_te.min(), pred_b.min()), max(y_te.max(), pred_b.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("rzeczywista log-cena")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title(f"B) Trzebińska + {len(ULICE_DODATKOWE)} ulic (RMSE={t_b:.3f})")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "ulice_dotrenowanie.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")

    # --- wykres przyrostowy ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    ks = [r[0] for r in incremental]
    ts = [r[2] for r in incremental]
    ax.plot(ks, ts, "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax.axhline(t_a, color="red", linestyle="--", linewidth=1.5,
               label=f"tylko Trzebińska ({t_a:.3f})")
    ax.scatter([best_k], [min(ts)], color="#55A868", s=150, zorder=5,
               label=f"najlepszy: +{best_k} ulic")
    ax.set_xlabel("liczba dodatkowych ulic")
    ax.set_ylabel("RMSE (test Trzebińskiej)")
    ax.set_title("Wpływ liczby dodatkowych ulic na predykcję Trzebińskiej")
    ax.legend()
    ax.grid(alpha=0.3)
    for k, t in zip(ks, ts):
        ax.annotate(f"{t:.3f}", (k, t), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    out2 = "ulice_przyrostowo.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"Zapisano wykres: {out2}")


if __name__ == "__main__":
    main()
