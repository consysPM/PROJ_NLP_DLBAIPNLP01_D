from datasets import load_dataset

from config import CATEGORIES, DATASET, DATA_DIR


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        config_name = f"raw_review_{category}"
        print(f"Lade {config_name} ...")
   
        load_dataset(
            DATASET,
            config_name,
            split="full",
            cache_dir=str(DATA_DIR),
            trust_remote_code=True,
            #bin mit huggingface cli angemeldet, wirft fehler wenn nicht angegeben
            token=False,
        )

    print("\nDownload abgeschlossen. Cache-Ordner:", DATA_DIR.resolve())


if __name__ == "__main__":
    main()
