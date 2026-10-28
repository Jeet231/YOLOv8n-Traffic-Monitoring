# Dataset setup

The GRAM road-traffic dataset is **not included** in this repository. Obtain it from its original authorized source and follow its usage terms.

The dataset-preparation script expects the following directory structure:

```text
datasets/GRAM/
├── Images/
│   ├── M-30/
│   ├── M-30-HD/
│   └── Urban1/
└── Annotations/
    ├── M-30/xml/
    ├── M-30-HD/xml/
    └── Urban1/xml/
```

Run `python scripts/prepare_gram.py` from the repository root. The script creates `prepared_data/GRAM/` with YOLO labels, split images, metadata and a generated `data.yaml`. It uses a chronological 70/15/15 split within each source subset.

**Note:** The generated `data.yaml` contains an absolute path to the machine that ran preparation. Regenerate it on a new machine; do not commit a stale machine-specific path.
