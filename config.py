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
