# GitHub upload checklist

## Upload these (included in this starter ZIP)

- [x] `README.md` — project overview, results, setup and usage
- [x] `requirements.txt` — Python dependencies
- [x] `.gitignore` — excludes datasets, videos, weights and generated outputs
- [x] `scripts/` — GRAM preparation, training, inference and tracking scripts
- [x] `scripts/original/` — archived original tracking scripts for traceability
- [x] `tools/` — dataset inspection and conversion helpers
- [x] `configs/` — ByteTrack YAML configuration
- [x] `assets/` — five project-report diagrams/graphs
- [x] `datasets/README.md` — dataset layout instructions (no data)

## Add only if you have and can share them

- [ ] Trained `best.pt` checkpoint — NOT found in supplied archive. Prefer a GitHub Release or Git LFS rather than ordinary Git; check model/dataset license conditions.
- [ ] A short **anonymized and authorized** demonstration video or screenshot showing real vehicle tracking — not in supplied code archive.
- [ ] Selected tracking summary JSON/CSV or training plots — not in supplied code archive; remove any personal or machine-specific information first.
- [ ] Final project report and PPT — optional; review personal names/enrollment numbers, third-party figures, and publication restrictions before public upload.
- [ ] Paper manuscript — optional **only after** checking conference/journal policy, review anonymity and publication rights.
- [ ] `LICENSE` — decide jointly with project contributors; do not copy an upstream license as your own without checking.

## Do not upload blindly

- [ ] Raw GRAM images/annotations or repackaged dataset without redistribution permission
- [ ] Full original `test_1.mov` personal traffic recording without consent/rights review
- [ ] Full copied ByteTrack, DeepSORT or OC-SORT upstream Git clones (including nested `.git` histories)
- [ ] `.venv/`, `runs/`, `results/`, large generated data, model caches or temporary files
- [ ] Passwords, tokens, API keys, local absolute paths or personally identifying data
- [ ] Every weekly college report (usually unnecessary and may expose personal details)

## Publish

1. Extract the GitHub-ready ZIP.
2. Create a new empty repository on GitHub, e.g. `yolov8n-traffic-monitoring`.
3. Upload the **contents of the extracted folder** (not the ZIP itself).
4. Ensure `README.md` is in the repository root.
5. Replace the placeholder GitHub URL in the README after creating the repository.
6. Review the repository on GitHub before making it public.

> Source code was syntax-checked. Complete execution and model metrics were **not** rerun during packaging. OC-SORT requires additional integration.
