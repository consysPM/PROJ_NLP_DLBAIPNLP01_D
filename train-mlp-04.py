from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from config import (
    CATEGORIES, DATASET, DATA_DIR, PROCESSED_DIR,
    SAMPLE_SIZE, TEST_SIZE, SEED,
    stem_tokens, MODELS_DIR, VECTORIZER_FILE, MLP_MODEL_FILE,
)

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
TEST_FILE  = PROCESSED_DIR / "test.parquet"

def main():
    print("")

    # TRAIN
    df_train = pd.read_parquet(TRAIN_FILE)
    corpus = df_train["text"].fillna("") + " " + df_train["title"].fillna("")

    # TEST
    df_test = pd.read_parquet(TEST_FILE)
    corpus_test = df_test["title"].fillna("") + " " + df_test["text"].fillna("")

    #todo: lemming -> spacy?

    print(corpus)
    #stop_words='english'
    vectorizer = TfidfVectorizer(tokenizer=stem_tokens, 
                                 token_pattern=None,
                                 min_df=5, #ignoriert bzw. entfernt seltene wörter
                                 ngram_range=(1,2))


    #MultinomialNB
    #Naive Bayes classifier for multinomial models.
    #The multinomial Naive Bayes classifier is suitable for classification with 
    # discrete features (e.g., word counts for text classification). 
    # The multinomial distribution normally requires integer feature counts. 
    # However, in practice, fractional counts such as tf-idf may also work.
    # sollte d.h. mit TF-IDF funktionieren

    X = vectorizer.fit_transform(corpus)
    print(X)
    print(X.shape)
    print(vectorizer.get_stop_words())
    y = df_train["rating"]
    print(vectorizer.get_feature_names_out())

    model = MLPClassifier(hidden_layer_sizes=(8,), early_stopping=True, 
                          verbose=True,
                          max_iter=40,
                          batch_size=512,
                          random_state=SEED,
                          validation_fraction=0.1)
    model.fit(X, y)

    X_test = vectorizer.transform(corpus_test)   # transform, NICHT fit_transform!
    y_test = df_test["rating"]

    y_pred = model.predict(X_test)

    # --- Auswertung ---
    # Genauigkeit (ein einzelner Wert zwischen 0 und 1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # F1 SCORE
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Verwirrungsmatrix (5x5)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
    print(cm)

    # --- Modell speichern (fuer predict-05.py) ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(model, MLP_MODEL_FILE)
    print(f"Gespeichert: {VECTORIZER_FILE} und {MLP_MODEL_FILE}")
    
    disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=[1, 2, 3, 4, 5], cmap="Blues")
    disp.ax_.set_ylabel("Ground Trueth (Rating)")
    disp.ax_.set_xlabel("Predicted (Rating)")

    # optional: vorhergesagte Achse nach oben
    disp.ax_.xaxis.tick_top()
    disp.ax_.xaxis.set_label_position("top")
    #plt.savefig("konfusionsmatrix.png", dpi=150, bbox_inches="tight")
    print(disp)
    plt.show()



if __name__ == "__main__":
    main()