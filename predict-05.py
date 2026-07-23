import joblib

# stem_tokens wird hier nicht direkt aufgerufen, muss aber importierbar sein,
# damit joblib den Vektorisierer (tokenizer=config.stem_tokens) laden kann.
from config import stem_tokens  # noqa: F401
from config import VECTORIZER_FILE, NB_MODEL_FILE, MLP_MODEL_FILE


def main():
    if not VECTORIZER_FILE.exists():
        print("Kein Vektorisierer gefunden - zuerst train-nb-03.py oder "
              "train-mlp-04.py ausfuehren.")
        return

    vectorizer = joblib.load(VECTORIZER_FILE)

    models = {}
    if NB_MODEL_FILE.exists():
        models["Naive Bayes"] = joblib.load(NB_MODEL_FILE)
    if MLP_MODEL_FILE.exists():
        models["MLP"] = joblib.load(MLP_MODEL_FILE)

    if not models:
        print("Keine Modelle gefunden - zuerst die Train-Skripte ausfuehren.")
        return

    print(f"Geladene Modelle: {', '.join(models)}")
    print("Rezension (englisch) eingeben, leere Eingabe beendet.\n")

    while True:
        text = input("Rezension: ").strip()
        if not text:
            break
        X = vectorizer.transform([text])
        for name, model in models.items():
            stars = int(model.predict(X)[0])
            print(f"  {name}: {stars} Sterne ({'*' * stars})")
        print("")


if __name__ == "__main__":
    main()
