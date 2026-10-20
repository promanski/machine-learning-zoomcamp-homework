"""
Prosty frontend (Flask) do predykcji ceny mieszkania w powiecie chrzanowskim.

Model: Hist Gradient Boosting trenowany na 1203_transakcje_ceny.gpkg.
Wycena "tu i teraz" - ceny skorygowane o inflację (realne 2026),
bez trendu czasowego (bez roku transakcji), z cechami przestrzennymi.

Uruchomienie:  python app.py
Następnie otwórz:  http://127.0.0.1:5000
"""

import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from sklearn.ensemble import HistGradientBoostingRegressor

from mieszkania_prediction import (
    load_data, clean, split_data, build_cat_map, rmse,
)
from inflacja_korekta import wspolczynnik_deflacji

app = Flask(__name__)

# ----------------------------------------------------------------------
# Trening modelu przy starcie
# ----------------------------------------------------------------------
# cechy bez roku (wycena "tu i teraz")
NUMERIC = ["lok_pow_uzyt", "lok_liczba_izb", "lok_nr_kond",
           "x", "y", "odl_centrum"]
CATEGORICAL = ["miasto"]

MODEL = None
CATEGORICAL_MAP = None
CENTRA = None
CITIES = None
STATS = None
YEARS = None

# model dedykowany dla ulic (Trzebińska + 5 ulic)
STREET_MODEL = None
STREET_CAT_MAP = None
STREET_CENTRA = None
STREETS = None
STREET_STATS = None
STREET_FEATURES = None
STREET_CAT_FEATURE = None
MODEL_FILE = "model_trzebinska_cat.pkl"


def prepare_X(df, categorical, numeric=NUMERIC):
    df = df.copy()
    features = list(numeric)
    for name, values in categorical.items():
        for value in values:
            col = f"{name}_{value}"
            df[col] = (df[name] == value).astype(int)
            features.append(col)
    return df[features].fillna(0).values


def train_model():
    global MODEL, CATEGORICAL_MAP, CENTRA, CITIES, STATS, YEARS

    df = load_data()
    df = clean(df, verbose=False)

    # korekta cen o inflację -> wartości realne 2026
    df["wsp_inflacji"] = df["rok"].apply(wspolczynnik_deflacji)
    df["cena_realna_2026"] = df["tran_cena_brutto"] * df["wsp_inflacji"]

    df_train, df_val, df_test = split_data(df)

    # cel: cena realna (skorygowana o inflację do 2026)
    y_train = np.log1p(df_train["cena_realna_2026"].values)
    y_val = np.log1p(df_val["cena_realna_2026"].values)
    y_test = np.log1p(df_test["cena_realna_2026"].values)

    CATEGORICAL_MAP = build_cat_map(df_train)

    X_train = prepare_X(df_train, CATEGORICAL_MAP)
    X_val = prepare_X(df_val, CATEGORICAL_MAP)
    X_test = prepare_X(df_test, CATEGORICAL_MAP)

    MODEL = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=42)
    MODEL.fit(X_train, y_train)

    val_rmse = rmse(y_val, MODEL.predict(X_val))
    test_rmse = rmse(y_test, MODEL.predict(X_test))

    # centra miast (do liczenia odległości od centrum)
    CENTRA = df.groupby("miasto")[["x", "y"]].median().to_dict("index")

    # lista miast (posortowana)
    CITIES = sorted(df["miasto"].unique().tolist())

    # statystyki do wyświetlenia
    STATS = {
        "n": len(df),
        "rmse_val": round(val_rmse, 4),
        "rmse_test": round(test_rmse, 4),
        "cena_min": int(df["cena_realna_2026"].min()),
        "cena_max": int(df["cena_realna_2026"].max()),
        "cena_sr_m2": int((df["cena_realna_2026"] / df["lok_pow_uzyt"]).mean()),
    }

    print(f"Model wytrenowany. RMSE(val)={val_rmse:.4f} "
          f"RMSE(test)={test_rmse:.4f}")
    print(f"Miasta: {CITIES}")


def load_street_model():
    """Ładuje zapisany model dedykowany dla ulic (CatBoost)."""
    global STREET_MODEL, STREET_CAT_MAP, STREET_CENTRA, STREETS, STREET_STATS
    global STREET_FEATURES, STREET_CAT_FEATURE
    try:
        with open(MODEL_FILE, "rb") as f:
            payload = pickle.load(f)
        STREET_MODEL = payload["model"]
        STREET_CENTRA = payload["centra"]
        STREETS = payload["ulice"]
        STREET_FEATURES = payload["features"]
        STREET_CAT_FEATURE = payload["cat_feature"]
        STREET_STATS = {
            "rmse_test": round(payload["rmse_test"], 4),
            "rmse_test_pln": int(payload["rmse_test_pln"]),
            "n_train": payload["n_train"],
            "model_type": payload.get("model_type", "CatBoost"),
        }
        print(f"Załadowano model ulic ({STREET_STATS['model_type']}): "
              f"{STREETS}")
    except FileNotFoundError:
        STREET_MODEL = None
        STREETS = []
        print(f"Brak pliku {MODEL_FILE} - model ulic niedostępny")


# ----------------------------------------------------------------------
# Endpointy
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        cities=CITIES,
        stats=STATS,
        streets=STREETS or [],
        street_stats=STREET_STATS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        powierzchnia = float(data["powierzchnia"])
        izby = int(data["izby"])
        kondygnacja = int(data["kondygnacja"])
        miasto = data["miasto"]
        ulica = data.get("ulica", "")  # opcjonalny wybór ulicy

        # walidacja
        if not (15 <= powierzchnia <= 200):
            return jsonify({"error": "Powierzchnia musi być w zakresie 15-200 m2"}), 400
        if not (1 <= izby <= 10):
            return jsonify({"error": "Liczba izb musi być w zakresie 1-10"}), 400

        # --- wybór modelu: dedykowany (ulica) lub ogólny (miasto) ---
        if ulica and STREET_MODEL is not None and ulica in STREET_CENTRA:
            model = STREET_MODEL
            x = STREET_CENTRA[ulica]["x"]
            y = STREET_CENTRA[ulica]["y"]
            odl_centrum = 0.0
            miasto_used = "Chrzanów"
            model_label = f"ulica {ulica} ({STREET_STATS['model_type']})"

            row = {
                "lok_pow_uzyt": powierzchnia,
                "lok_liczba_izb": izby,
                "lok_nr_kond": kondygnacja,
                "x": x,
                "y": y,
                "odl_centrum": odl_centrum,
                STREET_CAT_FEATURE: ulica,
            }
            X = pd.DataFrame([row])[STREET_FEATURES].fillna(0)
            cena = float(np.expm1(model.predict(X)[0]))
        else:
            if miasto not in CENTRA:
                return jsonify({"error": "Nieznane miasto"}), 400
            x = CENTRA[miasto]["x"]
            y = CENTRA[miasto]["y"]
            odl_centrum = 0.0
            model_label = f"miasto {miasto}"

            row = {
                "lok_pow_uzyt": powierzchnia,
                "lok_liczba_izb": izby,
                "lok_nr_kond": kondygnacja,
                "x": x,
                "y": y,
                "odl_centrum": odl_centrum,
                "miasto": miasto,
            }
            X = prepare_X(pd.DataFrame([row]), CATEGORICAL_MAP)
            cena = float(np.expm1(MODEL.predict(X)[0]))

        cena_m2 = cena / powierzchnia

        return jsonify({
            "cena": round(cena),
            "cena_m2": round(cena_m2),
            "cena_min": round(cena * 0.85),
            "cena_max": round(cena * 1.15),
            "model": model_label,
        })

    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Błędne dane: {e}"}), 400


if __name__ == "__main__":
    train_model()
    load_street_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
