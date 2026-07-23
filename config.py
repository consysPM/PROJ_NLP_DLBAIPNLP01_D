"""
config.py
Projektbericht DLBAIPNLP01_D - Sentimentanalyse von Produktrezensionen

Gemeinsame Konfiguration fuer alle Pipeline-Stufen (download-01, prepare-02, ...).
Zentrale Ablage von Datensatzname, Kategorien und Verzeichnissen, damit alle
Stufen dieselben Werte verwenden ("eine Quelle der Wahrheit").
"""

from pathlib import Path

# Wurzelverzeichnis des Projekts (Ordner, in dem diese Datei liegt).
PROJECT_DIR = Path(__file__).parent

# Datensatz auf HuggingFace.
DATASET = "McAuley-Lab/Amazon-Reviews-2023"

# Ausgewaehlte Produktkategorien (Schreibweise wie im Datensatz, mit "_").
CATEGORIES = [
    "Health_and_Personal_Care",
    "Digital_Music",
    "Video_Games",
]

# Ordner fuer die heruntergeladenen Rohdaten bzw. den datasets-Cache (Stufe 1).
DATA_DIR = PROJECT_DIR / "data"

# Ordner fuer die aufbereiteten Train-/Test-Dateien (Stufe 2).
PROCESSED_DIR = DATA_DIR / "processed"

# --- Parameter fuer Stufe 2 (prepare-02.py) --------------------------------

# Maximale Anzahl der insgesamt uebernommenen Rezensionen (0 = alle).
SAMPLE_SIZE = 50_000

# Anteil der Testdaten am Train-/Test-Split.
TEST_SIZE = 0.2

# Fester Zufalls-Seed fuer reproduzierbares Sampling und Splitten.
SEED = 1337

# --- Textvorverarbeitung (Stufe 3+) ----------------------------------------
# Liegt hier (und nicht in den Train-Skripten), damit der Tokenizer beim
# Speichern/Laden der Modelle mit joblib auffindbar bleibt: joblib pickelt nur
# eine Referenz "config.stem_tokens" - eine lokale Funktion in einem
# Train-Skript waere in predict-05.py nicht importierbar.

import re
from nltk.stem import SnowballStemmer

_stemmer = SnowballStemmer("english")
_token_pattern = re.compile(r"(?u)\b\w\w+\b")   # Woerter aus mind. 2 Zeichen

def stem_tokens(text):
    """Zerlegt den Text in Woerter und stemmt jedes einzelne."""
    return [_stemmer.stem(tok) for tok in _token_pattern.findall(text)]

# --- Gespeicherte Modelle (Stufe 3/4 -> predict-05.py) ----------------------
# Achtung: Der Vektorisierer wird von beiden Train-Skripten mit identischer
# Konfiguration erzeugt; das zuletzt gelaufene Skript ueberschreibt die Datei.

MODELS_DIR = DATA_DIR / "models"
VECTORIZER_FILE = MODELS_DIR / "vectorizer.joblib"
NB_MODEL_FILE = MODELS_DIR / "model_nb.joblib"
MLP_MODEL_FILE = MODELS_DIR / "model_mlp.joblib"
