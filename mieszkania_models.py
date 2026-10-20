"""
Porównanie modeli predykcji ceny mieszkania (1203_transakcje_ceny.gpkg).

Modele: Ridge, Random Forest, Gradient Boosting, Hist Gradient Boosting.
Cel: tran_cena_brutto (log1p). Metryka: RMSE.
Wykorzystuje pipeline z mieszkania_prediction.py (parser WKB + cechy przestrzenne).
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
    load_data, clean, split_data, build_cat_map, prepare_X, rmse,
)


def main():
    df = load_data()
    df = clean(df, verbose=False)
    print(f"Mieszkania po czyszczeniu: {len(df)}\n")

    df_train, df_val, df_test = split_data(df)

    y_train = np.log1p(df_train["tran_cena_brutto"].values)
    y_val = np.log1p(df_val["tran_cena_brutto"].values)
    y_test = np.log1p(df_test["tran_cena_brutto"].values)

    categorical = build_cat_map(df_train)

    X_train = prepare_X(df_train, categorical)
    X_val = prepare_X(df_val, categorical)
    X_test = prepare_X(df_test, categorical)

    print(f"Liczba cech: {X_train.shape[1]}, "
          f"train={X_train.shape[0]}, val={X_val.shape[0]}, "
          f"test={X_test.shape[0]}\n")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=42),
    }

    print("Porównanie modeli (cena mieszkania):")
    print("-" * 70)

    names, val_scores, test_scores, test_pln = [], [], [], []
    preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        p_val = model.predict(X_val)
        p_test = model.predict(X_test)

        v = rmse(y_val, p_val)
        t = rmse(y_test, p_test)
        # RMSE w złotówkach
        t_pln = np.sqrt(((np.expm1(y_test) - np.expm1(p_test)) ** 2).mean())

        names.append(name)
        val_scores.append(v)
        test_scores.append(t)
        test_pln.append(t_pln)
        preds[name] = p_test

        print(f"{name:<24} RMSE(val)={v:.4f}  RMSE(test)={t:.4f}  "
              f"({t_pln:,.0f} zł)")

    best_name = names[int(np.argmin(val_scores))]
    best_pred = preds[best_name]
    print(f"\nNajlepszy model: {best_name}")

    # ------------------------------------------------------------------
    # Wykresy
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Porównanie modeli - predykcja ceny mieszkania",
                 fontsize=16, fontweight="bold")

    # 1. RMSE słupkowy (log)
    ax = axes[0, 0]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, val_scores, w, label="walidacja", color="#4C72B0")
    ax.bar(x + w / 2, test_scores, w, label="test", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (log-cena)")
    ax.set_title("RMSE modeli (niżej = lepiej)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (v, t) in enumerate(zip(val_scores, test_scores)):
        ax.text(i - w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, t + 0.005, f"{t:.3f}", ha="center", fontsize=8)

    # 2. RMSE w złotówkach
    ax = axes[0, 1]
    bars = ax.bar(names, test_pln, color="#55A868")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE (test) [zł]")
    ax.set_title("RMSE w złotówkach (niżej = lepiej)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, test_pln):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1000,
                f"{v:,.0f}", ha="center", fontsize=8)

    # 3. predykcja vs rzeczywistość (najlepszy)
    ax = axes[1, 0]
    ax.scatter(y_test, best_pred, s=8, alpha=0.3, color="#4C72B0")
    lims = [min(y_test.min(), best_pred.min()),
            max(y_test.max(), best_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="idealna predykcja")
    ax.set_xlabel("rzeczywista log-cena")
    ax.set_ylabel("przewidywana log-cena")
    ax.set_title(f"Predykcja vs rzeczywistość ({best_name})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. ważność cech (Random Forest)
    ax = axes[1, 1]
    rf = models["Random Forest"]
    feature_names = (
        ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond", "rok", "miesiac",
         "x", "y", "odl_centrum"]
        + [f"miasto_{v}" for v in categorical["miasto"]]
    )
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    ax.barh(
        [feature_names[i] for i in order][::-1],
        importances[order][::-1],
        color="#C44E52",
    )
    ax.set_xlabel("ważność")
    ax.set_title("Top 15 najważniejszych cech (Random Forest)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "mieszkania_models.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nZapisano wykres: {out}")


if __name__ == "__main__":
    main()
