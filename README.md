# m7G-DOLL

RNA-FM + DNABERT-2 dual-PLM m7G site prediction with LoRA fine-tuning and
out-of-fold probability fusion.

Each site is scored independently by two pre-trained nucleotide language models
adapted with LoRA; the two probability streams are fused with a weight selected
on out-of-fold cross-validation predictions, and the fused model is evaluated once
on the independent test split.

## Repository contents

| Path | Description |
|---|---|
| `dual_plm.py` | Training / evaluation pipeline: LoRA injection, 5-fold CV, epoch selection, OOF fusion-weight search, independent-test metrics |
| `dataset/` | Processed sequence-window CSVs (401 nt full-transcript, 201 nt mature RNA), plus `manifest.json`, `SHA256SUMS` and provenance notes |
| `.env.example` | Documented environment variables (documentation only; no `.env` auto-loading) |
| `requirements.txt` | Direct dependencies |

## Data

| File | Role | Rows | Length |
|---|---|---:|---:|
| `train_mostdistant401.csv` | Full transcript training | 8988 | 401 nt |
| `test_mostdistant401.csv` | Full transcript independent test | 2248 | 401 nt |
| `train_mature_mostdistant201.csv` | Mature RNA training | 2386 | 201 nt |
| `test_mature_mostdistant201.csv` | Mature RNA independent test | 596 | 201 nt |

Columns: `chr_id`, `Sequence` (A/C/G/T, central G), `Label` (0/1). File bytes are
treated as authoritative — see `dataset/manifest.json` for per-file SHA-256 and
`dataset/README.md` for the schema and provenance status. Keep the CSV line
endings unchanged (`.gitattributes` marks them `-text`) so hashes reproduce.

## Method summary

- **Branches**: RNA-FM (T→U normalized input) and DNABERT-2 (U→T normalized input),
  both with LoRA adapters on the attention projections and a 2-class head.
- **Model selection**: stratified 5-fold CV on the training split only; per-fold
  best epoch by validation AUC; fold-level mean ± std reported.
- **Fusion**: `p = w · p_rnafm + (1 − w) · p_dnabert2`, `w` chosen on OOF
  predictions; the test split is scored after the weight is frozen.
- **Metrics**: ACC, SN, SP, PREC, F1, MCC, AUC (and AUPRC per fold) at threshold
  0.5, plus an optimal-threshold view.

## Environment variables

| Variable | Meaning |
|---|---|
| `M7G_DATA_DIR` | Dataset directory (default `./dataset`) |
| `M7G_OUTPUT_DIR` | Output directory (default `./results`) |
| `M7G_DATASET` | `full`, `mature`, or `both` |
| `DNABERT_LOCAL_DIR` | Optional local DNABERT-2 snapshot; unset to pull from upstream |
| `M7G_DNABERT_REPO_ID` / `M7G_DNABERT_REVISION` | Pinned model source and revision |
| `M7G_OFFLINE` | `1` for `local_files_only` loading |
| `HF_ENDPOINT` | Optional Hugging Face endpoint |

## Running

```bash
pip install -r requirements.txt   # unpinned, direct deps only
python dual_plm.py                # entry point of the pipeline module
```

`dual_plm.py` is the pipeline module in this snapshot; the CLI launcher
(`run.py`, with `--audit-only` / `--smoke-test` / `--train`) referenced by
`dataset/README.md` and `.env.example` ships in the full working package and is
not included here. Outputs are written under `results/<dataset>/{rnafm,dnabert2}`
together with `all_results.json`.

## Data availability and licence

The CSVs are processed sequence windows supplied by the authors, not raw
sequencing reads. Source citation, redistribution permission and the
label/negative-sampling protocol are still to be documented — see
`dataset/README.md`. No data licence is assigned by this repository.
