"""
Generator edukacyjnego PDF: "Machine Learning od podstaw"
na bazie skryptów predykcji ceny samochodu i zużycia paliwa.

Uruchomienie:  python generate_ml_guide.py
Wynik:         ML_przewodnik.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Preformatted, KeepTogether,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ----------------------------------------------------------------------
# Rejestracja czcionek z polskimi znakami (DejaVu)
# ----------------------------------------------------------------------
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVu", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuMono", f"{FONT_DIR}/DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuMono-Bold", f"{FONT_DIR}/DejaVuSansMono-Bold.ttf"))
pdfmetrics.registerFontFamily(
    "DejaVu", normal="DejaVu", bold="DejaVu-Bold",
    italic="DejaVu", boldItalic="DejaVu-Bold",
)


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleX", parent=styles["Title"], fontName="DejaVu-Bold",
    fontSize=26, leading=32,
    textColor=colors.HexColor("#1a3a5c"), spaceAfter=6, alignment=TA_CENTER,
)
SUBTITLE = ParagraphStyle(
    "SubtitleX", parent=styles["Normal"], fontName="DejaVu",
    fontSize=13, leading=18,
    textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=20,
)
H1 = ParagraphStyle(
    "H1X", parent=styles["Heading1"], fontName="DejaVu-Bold",
    fontSize=18, leading=24,
    textColor=colors.HexColor("#1a3a5c"), spaceBefore=16, spaceAfter=8,
)
H2 = ParagraphStyle(
    "H2X", parent=styles["Heading2"], fontName="DejaVu-Bold",
    fontSize=14, leading=19,
    textColor=colors.HexColor("#2c5f8a"), spaceBefore=12, spaceAfter=6,
)
H3 = ParagraphStyle(
    "H3X", parent=styles["Heading3"], fontName="DejaVu-Bold",
    fontSize=12, leading=16,
    textColor=colors.HexColor("#3a7ca5"), spaceBefore=8, spaceAfter=4,
)
BODY = ParagraphStyle(
    "BodyX", parent=styles["Normal"], fontName="DejaVu",
    fontSize=10.5, leading=15.5,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
BULLET = ParagraphStyle(
    "BulletX", parent=BODY, fontName="DejaVu",
    leftIndent=16, bulletIndent=6, spaceAfter=3,
)
CODE = ParagraphStyle(
    "CodeX", parent=styles["Code"], fontName="DejaVuMono",
    fontSize=8.5, leading=11.5,
    textColor=colors.HexColor("#1a1a1a"),
    backColor=colors.HexColor("#f4f4f4"),
    borderColor=colors.HexColor("#cccccc"), borderWidth=0.5,
    borderPadding=6, spaceBefore=4, spaceAfter=8,
)
NOTE = ParagraphStyle(
    "NoteX", parent=BODY, fontName="DejaVu",
    fontSize=10, leading=14,
    textColor=colors.HexColor("#333333"),
    backColor=colors.HexColor("#eef6fb"),
    borderColor=colors.HexColor("#3a7ca5"), borderWidth=0.5,
    borderPadding=8, spaceBefore=6, spaceAfter=8,
)


def P(text, style=BODY):
    return Paragraph(text, style)


def code(text):
    return Preformatted(text.strip("\n"), CODE)


def bullets(items):
    return [Paragraph(f"&bull;&nbsp; {t}", BULLET) for t in items]


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ----------------------------------------------------------------------
# Treść dokumentu
# ----------------------------------------------------------------------
def build_story():
    s = []

    # ---------------- Strona tytułowa ----------------
    s.append(Spacer(1, 3 * cm))
    s.append(P("Machine Learning od podstaw", TITLE))
    s.append(P("Praktyczny przewodnik na bazie projektu<br/>"
               "predykcji ceny samochodu i zużycia paliwa", SUBTITLE))
    s.append(Spacer(1, 1 * cm))
    s.append(P(
        "Ten dokument wyjaśnia krok po kroku, jak działa uczenie maszynowe "
        "(Machine Learning) na przykładzie realnego zbioru danych o samochodach. "
        "Wszystkie koncepcje są zilustrowane kodem, który sam uruchamiałeś, "
        "oraz wynikami, które otrzymałeś.", BODY))
    s.append(Spacer(1, 0.5 * cm))
    s.append(P(
        "<b>Zawartość:</b> dane i ich przygotowanie &bull; eksploracja (EDA) &bull; "
        "podział train/val/test &bull; regresja liniowa &bull; metryka RMSE &bull; "
        "inżynieria cech &bull; zmienne kategoryczne &bull; regularyzacja &bull; "
        "modele drzewiaste &bull; porównanie modeli &bull; interpretacja wyników.",
        NOTE))
    s.append(PageBreak())

    # ---------------- 1. Czym jest ML ----------------
    s.append(P("1. Czym jest uczenie maszynowe?", H1))
    s.append(P(
        "Uczenie maszynowe to sposób, w którym komputer <b>uczy się z danych</b>, "
        "zamiast być programowany regułami ręcznie. Zamiast pisać "
        "\"jeśli moc silnika &gt; 300, to cena wysoka\", pokazujemy modelowi "
        "tysiące przykładów (samochód + jego cena), a on sam odkrywa zależności.", BODY))
    s.append(P("Rodzaje uczenia:", H2))
    s.extend(bullets([
        "<b>Uczenie nadzorowane</b> &mdash; mamy dane wejściowe (X) i znane "
        "odpowiedzi (y). Model uczy się mapowania X &rarr; y. "
        "Nasz projekt jest właśnie taki.",
        "<b>Regresja</b> &mdash; przewidujemy liczbę (np. cena, MPG). "
        "To robimy w tym projekcie.",
        "<b>Klasyfikacja</b> &mdash; przewidujemy kategorię (np. czy auto jest drogie: tak/nie).",
        "<b>Uczenie nienadzorowane</b> &mdash; brak odpowiedzi, szukamy struktury "
        "(np. grupowanie podobnych aut).",
    ]))
    s.append(P(
        "W naszym projekcie: <b>X</b> = cechy samochodu (moc, marka, rok...), "
        "<b>y</b> = cena (MSRP). Model uczy się przewidywać cenę na podstawie cech.", NOTE))

    # ---------------- 2. Dane ----------------
    s.append(P("2. Dane i ich przygotowanie", H1))
    s.append(P(
        "Wszystko zaczyna się od danych. Nasz plik <font face='DejaVuMono'>data.csv</font> "
        "zawiera ok. 12 000 samochodów, każdy opisany 16 kolumnami:", BODY))
    s.append(table([
        ["Kolumna", "Znaczenie", "Typ"],
        ["Make, Model", "marka, model", "tekst"],
        ["Year", "rok produkcji", "liczba"],
        ["Engine HP", "moc silnika (konie mechaniczne)", "liczba"],
        ["Engine Cylinders", "liczba cylindrów", "liczba"],
        ["highway MPG, city mpg", "zużycie paliwa", "liczba"],
        ["Popularity", "popularność modelu", "liczba"],
        ["MSRP", "cena (nasz cel!)", "liczba"],
    ], col_widths=[4.5 * cm, 8 * cm, 3 * cm]))
    s.append(Spacer(1, 0.3 * cm))
    s.append(P("Normalizacja nazw kolumn", H2))
    s.append(P(
        "Nazwy kolumn mają spacje i wielkie litery. Zamieniamy je na małe litery "
        "i podkreślenia, żeby łatwiej się do nich odwoływać:", BODY))
    s.append(code(
        "df.columns = df.columns.str.lower().str.replace(' ', '_')\n"
        "# 'Engine HP'  ->  'engine_hp'"))
    s.append(P(
        "To samo robimy z wartościami tekstowymi (marki, typy paliwa), żeby "
        "np. 'BMW' i 'bmw' były traktowane identycznie.", BODY))

    # ---------------- 3. EDA ----------------
    s.append(P("3. Eksploracja danych (EDA)", H1))
    s.append(P(
        "EDA (Exploratory Data Analysis) to poznanie danych przed modelowaniem. "
        "Sprawdzamy rozkłady, braki i dziwne wartości.", BODY))
    s.append(P("Długi ogon w rozkładzie ceny", H2))
    s.append(P(
        "Ceny samochodów mają rozkład z <b>długim ogonem</b>: większość aut jest "
        "tania, ale kilka jest ekstremalnie drogich (Ferrari, Maybach). "
        "To psuje modele liniowe. Rozwiązanie: <b>logarytm</b>.", BODY))
    s.append(code(
        "price_logs = np.log1p(df.msrp)   # log(1 + cena)\n"
        "# log1p jest bezpieczne dla zera i ściska duże wartości"))
    s.append(P(
        "Po logarytmowaniu rozkład staje się bardziej symetryczny (dzwonowy), "
        "co bardzo pomaga modelom. Pamiętaj: przewidujemy log(cenę), a na końcu "
        "wracamy do złotówek przez <font face='DejaVuMono'>np.expm1()</font>.", NOTE))
    s.append(P("Braki danych", H2))
    s.append(P(
        "Niektóre auta nie mają podanej mocy silnika. Zamiast usuwać wiersze, "
        "uzupełniamy braki zerem (proste i skuteczne w tym przypadku):", BODY))
    s.append(code("df.isnull().sum()          # ile braków w każdej kolumnie\n"
                  "df_num = df_num.fillna(0)  # uzupełnij braki zerem"))

    s.append(PageBreak())

    # ---------------- 4. Podział danych ----------------
    s.append(P("4. Podział danych: train / val / test", H1))
    s.append(P(
        "To jeden z najważniejszych kroków. <b>Nigdy nie oceniamy modelu na danych, "
        "na których się uczył</b> &mdash; to jak sprawdzanie ucznia na tych samych "
        "pytaniach, które widział. Dzielimy dane na trzy części:", BODY))
    s.append(table([
        ["Zbiór", "Udział", "Do czego służy"],
        ["train", "60%", "model uczy się na tych danych"],
        ["val (walidacja)", "20%", "wybieramy najlepszy model / parametry"],
        ["test", "20%", "ostateczna, uczciwa ocena (dotykamy raz!)"],
    ], col_widths=[3.5 * cm, 2.5 * cm, 9.5 * cm]))
    s.append(Spacer(1, 0.3 * cm))
    s.append(code(
        "n = len(df)\n"
        "n_val = int(n * 0.2)\n"
        "n_test = int(n * 0.2)\n"
        "n_train = n - n_val - n_test\n\n"
        "idx = np.arange(n)\n"
        "np.random.seed(2)          # powtarzalność wyników\n"
        "np.random.shuffle(idx)     # losowe przetasowanie\n\n"
        "df_train = df.iloc[idx[:n_train]]\n"
        "df_val   = df.iloc[idx[n_train:n_train+n_val]]\n"
        "df_test  = df.iloc[idx[n_train+n_val:]]"))
    s.append(P(
        "<b>Dlaczego tasujemy?</b> Bo dane w pliku są posortowane (np. wszystkie BMW "
        "razem). Bez tasowania model uczyłby się na jednych markach, a testowany był "
        "na innych &mdash; wynik byłby fałszywie zły.", NOTE))

    # ---------------- 5. Regresja liniowa ----------------
    s.append(P("5. Regresja liniowa &mdash; pierwszy model", H1))
    s.append(P(
        "Najprostszy model: przewidywana cena to suma ważona cech plus stała:", BODY))
    s.append(P(
        "y = w<sub>0</sub> + w<sub>1</sub>&middot;x<sub>1</sub> + "
        "w<sub>2</sub>&middot;x<sub>2</sub> + ... + w<sub>n</sub>&middot;x<sub>n</sub>",
        NOTE))
    s.append(P(
        "gdzie <b>w<sub>0</sub></b> to wyraz wolny (bias), a "
        "<b>w<sub>1</sub>...w<sub>n</sub></b> to wagi (jak bardzo każda cecha "
        "wpływa na cenę). Uczenie = znalezienie takich wag, żeby predykcje były "
        "jak najbliżej prawdy.", BODY))
    s.append(P("Jak znaleźć wagi? Równanie normalne", H2))
    s.append(P(
        "Istnieje elegancki wzór matematyczny, który od razu daje optymalne wagi:", BODY))
    s.append(P("w = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup> y", NOTE))
    s.append(P(
        "gdzie X to macierz cech (z dodatkową kolumną jedynek dla w<sub>0</sub>), "
        "a y to wektor cen. To nie magia &mdash; to rozwiązanie problemu "
        "najmniejszych kwadratów.", BODY))
    s.append(code(
        "def train_linear_regression(X, y):\n"
        "    ones = np.ones(X.shape[0])\n"
        "    X = np.column_stack([ones, X])   # dodaj kolumnę jedynek\n\n"
        "    XTX = X.T.dot(X)\n"
        "    XTX_inv = np.linalg.inv(XTX)     # macierz odwrotna\n"
        "    w_full = XTX_inv.dot(X.T).dot(y)\n\n"
        "    return w_full[0], w_full[1:]     # (w0, pozostałe wagi)"))

    s.append(PageBreak())

    # ---------------- 6. RMSE ----------------
    s.append(P("6. RMSE &mdash; jak mierzyć błąd?", H1))
    s.append(P(
        "Potrzebujemy liczby, która powie nam \"jak bardzo model się myli\". "
        "Najczęściej używamy <b>RMSE</b> (Root Mean Squared Error):", BODY))
    s.append(P("RMSE = &radic;( średnia( (y - y_pred)<sup>2</sup> ) )", NOTE))
    s.append(P("Krok po kroku:", H3))
    s.extend(bullets([
        "Policz różnicę między prawdą a predykcją dla każdego auta.",
        "Podnieś każdą różnicę do kwadratu (żeby minusy nie znikały).",
        "Uśrednij te kwadraty (MSE).",
        "Wyciągnij pierwiastek (RMSE) &mdash; wraca do jednostek ceny.",
    ]))
    s.append(code(
        "def rmse(y, y_pred):\n"
        "    se = (y - y_pred) ** 2\n"
        "    mse = se.mean()\n"
        "    return np.sqrt(mse)"))
    s.append(P(
        "Im <b>niższe RMSE</b>, tym lepszy model. RMSE = 0 oznaczałoby idealne "
        "predykcje. W naszym projekcie (na log-cenie) RMSE ~0.11 dla modeli "
        "drzewiastych to bardzo dobry wynik.", NOTE))

    # ---------------- 7. Inżynieria cech ----------------
    s.append(P("7. Inżynieria cech (feature engineering)", H1))
    s.append(P(
        "Często sami tworzymy nowe, mądrzejsze cechy z istniejących. To jedna "
        "z najskuteczniejszych technik w ML.", BODY))
    s.append(P("Przykład 1: wiek samochodu", H2))
    s.append(P(
        "Rok produkcji (np. 2015) sam w sobie niewiele mówi. Ale <b>wiek</b> "
        "(2017 - rok) jest bezpośrednio związany z ceną:", BODY))
    s.append(code("df['age'] = 2017 - df['year']"))
    s.append(P("Przykład 2: zmienne kategoryczne (one-hot encoding)", H2))
    s.append(P(
        "Model nie rozumie tekstu jak \"bmw\" czy \"sedan\". Zamieniamy każdą "
        "kategorię na kolumnę 0/1:", BODY))
    s.append(code(
        "df['make_bmw'] = (df['make'] == 'bmw').astype(int)\n"
        "# BMW -> 1, inne -> 0"))
    s.append(P(
        "Dla każdej ważnej kategorii tworzymy osobną kolumnę. To się nazywa "
        "<b>one-hot encoding</b>. Bierzemy tylko najczęstsze wartości "
        "(np. top 5 marek), żeby nie tworzyć tysięcy kolumn.", BODY))

    s.append(PageBreak())

    # ---------------- 8. Regularyzacja ----------------
    s.append(P("8. Regularyzacja &mdash; lekarstwo na niestabilność", H1))
    s.append(P(
        "Czasem macierz X<sup>T</sup>X jest <b>osobliwa</b> (nie da się jej "
        "odwrócić), bo cechy są silnie skorelowane. Wtedy wagi eksplodują do "
        "absurdalnych wartości. Widzieliśmy to w wynikach:", BODY))
    s.append(code(
        "r=0.0   ->  w0 = -292975507   RMSE = 113.55   (katastrofa!)\n"
        "r=1e-05 ->  w0 = 7.88         RMSE = 0.46     (świetnie!)"))
    s.append(P(
        "Rozwiązanie: <b>regularyzacja</b>. Dodajemy małą wartość do przekątnej "
        "macierzy, co czyni ją odwracalną:", BODY))
    s.append(P("X<sup>T</sup>X &rarr; X<sup>T</sup>X + r&middot;I", NOTE))
    s.append(code(
        "def train_linear_regression_reg(X, y, r=0.001):\n"
        "    ones = np.ones(X.shape[0])\n"
        "    X = np.column_stack([ones, X])\n\n"
        "    XTX = X.T.dot(X)\n"
        "    XTX = XTX + r * np.eye(XTX.shape[0])   # <-- regularyzacja\n\n"
        "    XTX_inv = np.linalg.inv(XTX)\n"
        "    w_full = XTX_inv.dot(X.T).dot(y)\n"
        "    return w_full[0], w_full[1:]"))
    s.append(P(
        "<b>r</b> to parametr, który dostrajamy. Za mały &rarr; niestabilność. "
        "Za duży &rarr; model traci zdolność uczenia. Szukamy złotego środka "
        "na zbiorze walidacyjnym.", NOTE))

    # ---------------- 9. Modele drzewiaste ----------------
    s.append(P("9. Modele drzewiaste &mdash; mocniejsze niż liniowe", H1))
    s.append(P(
        "Regresja liniowa zakłada, że zależności są <b>liniowe</b>. Ale cena "
        "samochodu zależy od cech w sposób nieliniowy (np. moc silnika podnosi "
        "cenę tylko dla marek premium). Tu wchodzą <b>drzewa decyzyjne</b>.", BODY))
    s.append(P("Jak działa drzewo?", H2))
    s.append(P(
        "Drzewo zadaje serię pytań: \"czy moc &gt; 200?\", \"czy wiek &lt; 5?\" "
        "i dzieli dane na grupy. Każda grupa dostaje swoją przewidywaną cenę. "
        "To naturalnie wychwytuje nieliniowości i interakcje cech.", BODY))
    s.append(P("Ensemble &mdash; siła w liczbie", H2))
    s.append(P(
        "Pojedyncze drzewo jest słabe. Ale <b>wiele drzew razem</b> (ensemble) "
        "daje świetne wyniki:", BODY))
    s.extend(bullets([
        "<b>Random Forest</b> &mdash; buduje setki drzew na losowych "
        "podzbiorach danych i uśrednia ich predykcje.",
        "<b>Gradient Boosting</b> &mdash; buduje drzewa po kolei, każde "
        "poprawia błędy poprzedniego.",
        "<b>Hist Gradient Boosting</b> &mdash; szybsza wersja boosting, "
        "świetna dla dużych zbiorów. Nasz zwycięzca!",
    ]))
    s.append(code(
        "from sklearn.ensemble import HistGradientBoostingRegressor\n\n"
        "model = HistGradientBoostingRegressor(\n"
        "    max_iter=300, learning_rate=0.1, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "y_pred = model.predict(X_test)"))

    s.append(PageBreak())

    # ---------------- 10. Porównanie wyników ----------------
    s.append(P("10. Porównanie modeli &mdash; wyniki", H1))
    s.append(P(
        "Uruchomiliśmy cztery modele na tych samych danych. Oto wyniki "
        "(RMSE na log-cenie, niżej = lepiej):", BODY))
    s.append(table([
        ["Model", "RMSE (walidacja)", "RMSE (test)"],
        ["Ridge (regresja liniowa)", "0.4608", "0.4605"],
        ["Random Forest", "0.1134", "0.1203"],
        ["Gradient Boosting", "0.1508", "0.1550"],
        ["Hist Gradient Boosting", "0.1112", "0.1156"],
    ], col_widths=[7 * cm, 4.5 * cm, 4 * cm]))
    s.append(Spacer(1, 0.3 * cm))
    s.append(P("Wnioski:", H3))
    s.extend(bullets([
        "Modele drzewiaste są <b>~4x lepsze</b> od liniowego (0.11 vs 0.46).",
        "<b>Hist Gradient Boosting</b> wygrywa &mdash; najniższy błąd.",
        "Regresja liniowa jest najsłabsza, bo nie wychwytuje nieliniowości.",
    ]))
    s.append(P(
        "Ważne: RMSE na walidacji i teście są <b>zbliżone</b> (np. 0.1112 vs "
        "0.1156). To znaczy, że model <b>nie przeucza się</b> &mdash; dobrze "
        "generalizuje na nowe dane.", NOTE))

    # ---------------- 11. Interpretacja ----------------
    s.append(P("11. Interpretacja wyników", H1))
    s.append(P("Co wpływa na cenę? (ważność cech)", H2))
    s.append(P(
        "Random Forest potrafi powiedzieć, które cechy są najważniejsze. "
        "W naszym modelu:", BODY))
    s.extend(bullets([
        "<b>age (wiek)</b> &mdash; dominująca cecha (~75%). Starsze auto = tańsze.",
        "<b>engine_hp (moc)</b> &mdash; druga najważniejsza (~20%).",
        "Pozostałe cechy mają marginalne znaczenie.",
    ]))
    s.append(P("Nakładka predykcji i prawdy", H2))
    s.append(P(
        "Gdy nałożymy rozkład przewidywanych cen na prawdziwe, histogramy "
        "<b>pokrywają się niemal idealnie</b>. To dowód, że model nie ma "
        "systematycznego obciążenia (bias).", BODY))
    s.append(P("Gdzie model się myli?", H2))
    s.extend(bullets([
        "Największe błędy przy <b>skrajnych cenach</b> (bardzo tanich i drogich) "
        "&mdash; jest ich mało w danych.",
        "Auta elektryczne/hybrydowe (skrajne MPG) są trudne do przewidzenia.",
        "W głównym zakresie cen model jest bardzo dokładny.",
    ]))

    # ---------------- 12. Predykcja MPG ----------------
    s.append(P("12. Drugi projekt: predykcja zużycia paliwa", H1))
    s.append(P(
        "Ten sam warsztat zastosowaliśmy do przewidywania MPG (zużycia paliwa). "
        "Zmieniliśmy tylko cel (y) z ceny na MPG:", BODY))
    s.append(table([
        ["Cel", "RMSE (walidacja)", "RMSE (test)"],
        ["highway_mpg", "1.8052", "1.5207"],
        ["city_mpg", "1.4908", "1.3434"],
    ], col_widths=[7 * cm, 4.5 * cm, 4 * cm]))
    s.append(Spacer(1, 0.3 * cm))
    s.append(P(
        "RMSE ~1.5 MPG to ok. 5-7% typowego błędu &mdash; bardzo dobry wynik. "
        "City MPG jest łatwiejsze do przewidzenia niż highway.", NOTE))

    s.append(PageBreak())

    # ---------------- 13. Podsumowanie ----------------
    s.append(P("13. Podsumowanie &mdash; cały proces ML", H1))
    s.append(P("Powtórzmy pełny przepływ pracy, który wykonaliśmy:", BODY))
    s.append(table([
        ["Krok", "Co robimy", "Dlaczego"],
        ["1. Dane", "wczytanie, normalizacja", "spójny format"],
        ["2. EDA", "rozkłady, braki", "poznać dane"],
        ["3. Target", "log1p(cena)", "symetryczny rozkład"],
        ["4. Podział", "train/val/test", "uczciwa ocena"],
        ["5. Model", "regresja / drzewa", "nauczyć zależności"],
        ["6. Metryka", "RMSE", "zmierzyć błąd"],
        ["7. Cechy", "age, one-hot", "lepsze sygnały"],
        ["8. Regularyzacja", "r * I", "stabilność"],
        ["9. Tuning", "wybór r / modelu", "najlepszy wynik"],
        ["10. Ocena", "RMSE na teście", "ostateczna prawda"],
    ], col_widths=[3 * cm, 6 * cm, 6.5 * cm]))
    s.append(Spacer(1, 0.5 * cm))
    s.append(P("Złote zasady ML", H2))
    s.extend(bullets([
        "<b>Nigdy</b> nie oceniaj modelu na danych treningowych.",
        "Zawsze trzymaj zbiór testowy w tajemnicy do samego końca.",
        "Zacznij od prostego modelu (baseline), potem komplikuj.",
        "Mierz błąd na walidacji, wybieraj najlepszy wariant.",
        "Dbaj o to, by walidacja i test dawały podobne wyniki (brak przeuczenia).",
        "Interpretuj wyniki &mdash; sprawdzaj, czy mają sens biznesowy.",
    ]))
    s.append(Spacer(1, 0.5 * cm))
    s.append(P(
        "Gratulacje! Przeszedłeś przez kompletny projekt uczenia maszynowego "
        "&mdash; od surowych danych do działającego modelu predykcyjnego. "
        "Te same kroki stosuje się w każdym projekcie ML, niezależnie od "
        "dziedziny.", NOTE))

    return s


def main():
    doc = SimpleDocTemplate(
        "ML_przewodnik.pdf", pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Machine Learning od podstaw",
        author="Przewodnik edukacyjny",
    )
    doc.build(build_story())
    print("Zapisano: ML_przewodnik.pdf")


if __name__ == "__main__":
    main()
