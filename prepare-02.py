
import pandas as pd

from datasets import concatenate_datasets, load_dataset

from config import (
    CATEGORIES, DATASET, DATA_DIR, PROCESSED_DIR,
    SAMPLE_SIZE, TEST_SIZE, SEED,
)

# Spalten, die aus den Rohdaten uebernommen werden (Rest wird verworfen).
KEEP_COLUMNS = ["rating", "title", "text"]


def load_category(category):
    """Laedt eine Kategorie aus dem Cache, reduziert sie auf die benoetigten
    Spalten und ergaenzt eine Spalte 'category'."""
    config_name = f"raw_review_{category}"
    print(f"Lade {config_name} aus dem Cache ...")
    ds = load_dataset(
        DATASET,
        config_name,
        split="full",
        cache_dir=str(DATA_DIR),
        trust_remote_code=True,
        token=False,
    )
    ds = ds.select_columns(KEEP_COLUMNS)
    ds = ds.add_column("category", [category] * len(ds))
    return ds


def build_sample():
    """Fuehrt alle Kategorien zusammen, mischt sie und zieht die Stichprobe.
    SAMPLE_SIZE = 0 bedeutet: alle verfuegbaren Rezensionen verwenden."""
    combined = concatenate_datasets(
        [load_category(c) for c in CATEGORIES]
    ).shuffle(seed=SEED)
    if 0 < SAMPLE_SIZE < len(combined):
        return combined.select(range(SAMPLE_SIZE))
    return combined


def split_and_save(sample):
    """Teilt die Stichprobe stratifiziert nach Bewertung in Train/Test und
    speichert beide als Parquet."""
    df = sample.to_pandas()[["category", "title", "text", "rating"]].copy()
    df["rating"] = df["rating"].round().astype(int)

    # Stratifizierter Split: je Bewertungsstufe denselben Testanteil ziehen,
    # damit Train und Test dieselbe Verteilung besitzen.
    # stochastisch ziehen würde evtl. dazu führen das seltene klassen fast nie gezogen werden und daher die verteilung
    # im test-set nicht der realitit entspricht
    test = df.groupby("rating", group_keys=False).sample(
        frac=TEST_SIZE, random_state=SEED
    )
    train = df.drop(index=test.index)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    return train, test


def print_summary(train, test):
    """Gibt die Kennzahlen fuer Tabelle 2 und Tabelle 3 des Berichts aus."""
    total = len(train) + len(test)
    print(f"\nAnzahl Rezensionen gesamt: {total:,}")
    print(f"  Trainingssatz: {len(train):,}")
    print(f"  Testsatz:      {len(test):,}")
    print("Verteilung der Bewertungen (Train | Test):")
    for stars in range(1, 6):
        n_tr = int((train["rating"] == stars).sum())
        n_te = int((test["rating"] == stars).sum())
        print(f"  {stars} Sterne: {n_tr:>8,} | {n_te:>8,}")
    print("Rezensionen je Kategorie (Train | Test):")
    for category in CATEGORIES:
        n_tr = int((train["category"] == category).sum())
        n_te = int((test["category"] == category).sum())
        print(f"  {category}: {n_tr:>8,} | {n_te:>8,}")

    # Bewertungsverteilung je Kategorie (Anteil in %), um zu erkennen, ob
    # eine Kategorie die Gesamtverteilung in eine Richtung zieht.
    full = pd.concat([train, test], ignore_index=True)
    verteilung = pd.crosstab(
        full["category"], full["rating"], normalize="index"
    ).reindex(CATEGORIES) * 100
    print("\nBewertungsverteilung je Kategorie (Anteil % je Zeile):")
    print(verteilung.round(1).to_string())


def main():
    train, test = split_and_save(build_sample())
    print_summary(train, test)
    print(f"\nGespeichert unter: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    main()
