# Sentimentanalyse von Amazon-Rezensionen

Projektarbeit für den IU-Kurs DLBAIPNLP01_D (Projekt: NLP), Aufgabenstellung 1.
Sagt aus Titel + Text einer Amazon-Rezension die Sternebewertung (1-5) vorher.
Verglichen werden Naive Bayes (ComplementNB) und ein neuronales Netz (MLP),
beide auf denselben TF-IDF-Features.

## Setup (Windows)

```
setup.bat -> CMD
```

Legt die .venv an und installiert die requirements. Alternativ von Hand:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ausführen

Die Skripte bauen aufeinander auf und werden der Reihe nach gestartet:

```
python download-01.py     # lädt die Rohdaten von HuggingFace (dauert beim ersten Mal)
python prepare-02.py      # Stichprobe ziehen + Train/Test-Split (data/processed/)
python train-nb-03.py     # Naive Bayes trainieren + evaluieren
python train-mlp-04.py    # MLP trainieren + evaluieren
python predict-05.py      # eigene Rezensionen interaktiv bewerten, simples chat interface
```

Die Train-Skripte geben Accuracy, Macro-F1, classification_report und die
Konfusionsmatrix aus und speichern Modell + Vektorisierer nach data/models/.

Alle Einstellungen (Kategorien, Stichprobengröße, Seed usw.) liegen zentral
in config.py. Der Seed ist fixiert (1337), die Ergebnisse sind damit
reproduzierbar.

Hinweis: data/ und .venv/ sind bewusst nicht im Repo (siehe .gitignore),
die Daten werden von download-01.py selbst geladen.
