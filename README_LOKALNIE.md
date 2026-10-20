# Uruchomienie projektu lokalnie (bez Codespaces)

Cały projekt działa w 100% lokalnie. Potrzebujesz tylko Pythona i pliku
`1203_transakcje_ceny.gpkg`. Żadne dane nie są wysyłane do chmury.

## 1. Wymagania

- **Python 3.11+** (testowane na 3.14)
- ~500 MB miejsca na biblioteki
- Plik `1203_transakcje_ceny.gpkg` w katalogu projektu

## 2. Instalacja (Windows / macOS / Linux)

```bash
# wejdź do katalogu projektu
cd sciezka/do/machine-learning-zoomcamp-homework

# utwórz środowisko wirtualne
python -m venv .venv

# aktywuj (Linux/macOS)
source .venv/bin/activate
# aktywuj (Windows PowerShell)
# .venv\Scripts\Activate.ps1

# zainstaluj zależności
pip install -r requirements.txt
```

## 3. Uruchomienie pipeline'u optymalizacji

Kolejność ma znaczenie:

```bash
python opt_01_verify.py      # weryfikacja danych + leakage
python opt_02_baseline.py    # modele bazowe (Dummy, Ridge, CatBoost, ...)
python opt_03_features.py    # inżynieria cech
python opt_04_optuna.py      # strojenie CatBoost (Optuna)
```

Wyniki zapisują się do plików `opt_results_*.json`.

## 4. Uruchomienie aplikacji webowej (frontend)

```bash
# najpierw wytrenuj i zapisz model
python trzebinska_catboost.py

# uruchom serwer
python app.py
```

Otwórz w przeglądarce: **http://127.0.0.1:5000**

## 5. Uwagi

- **Nie commituj** katalogu `.venv` ani plików `*.pkl` (są duże).
- Jeśli używasz innej wersji Pythona, wersje bibliotek mogą się różnić —
  `requirements.txt` zawiera minimalne wersje.
- Wszystkie skrypty czytają dane z `1203_transakcje_ceny.gpkg` w katalogu
  bieżącym, więc uruchamiaj je z katalogu projektu.
