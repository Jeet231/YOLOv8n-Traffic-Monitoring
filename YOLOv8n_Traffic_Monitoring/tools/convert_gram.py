from pathlib import Path

# convert_gram.py is inside Traffic_Research
PROJECT_ROOT = Path(__file__).resolve().parent

GRAM_ROOT = PROJECT_ROOT / "datasets" / "GRAM"

DATASETS = {
    "M-30": {
        "images": GRAM_ROOT / "Images" / "M-30",
        "annotations": GRAM_ROOT / "Annotations" / "M-30" / "xml",
    },

    "M-30-HD": {
        "images": GRAM_ROOT / "Images" / "M-30-HD",
        "annotations": GRAM_ROOT / "Annotations" / "M-30-HD" / "xml",
    },

    "Urban1": {
        "images": GRAM_ROOT / "Images" / "Urban1",
        "annotations": GRAM_ROOT / "Annotations" / "Urban1" / "xml",
    },
}


for name, paths in DATASETS.items():

    print("\nDataset:", name)

    print("Images:")
    print(" ", paths["images"])
    print(" Exists:", paths["images"].exists())

    print("Annotations:")
    print(" ", paths["annotations"])
    print(" Exists:", paths["annotations"].exists())