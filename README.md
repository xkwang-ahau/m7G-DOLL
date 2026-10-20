# m7G-DOLL

m7G site prediction with two pre-trained nucleotide language models — RNA-FM and
DNABERT-2 — each adapted with LoRA, whose out-of-fold probabilities are fused
into a single score.

## Repository contents

| Path | Description |
|---|---|
| `dual_plm.py` | The whole pipeline: LoRA injection, 5-fold CV, epoch selection, OOF fusion-weight search, independent-test evaluation |
| `dataset/` | Four processed sequence-window CSVs (no manifest, no derived files) |

## Data

| File | Role | Rows | Window |
|---|---|---:|---:|
| `train_mostdistant401.csv` | Full-transcript training | 8988 | 401 nt |
| `test_mostdistant401.csv` | Full-transcript independent test | 2248 | 401 nt |
| `train_mature_mostdistant201.csv` | Mature-RNA training | 2386 | 201 nt |
| `test_mature_mostdistant201.csv` | Mature-RNA independent test | 596 | 201 nt |

Columns: `chr_id`, `Sequence` (A/C/G/T, central G), `Label` (0/1). Every file is
balanced 1:1.

## Requirements

Python 3.10+, and:

```bash
pip install numpy pandas scikit-learn torch rna-fm transformers einops
```

A CUDA GPU is effectively required: 5 folds × 20 epochs × 2 models × 2 datasets
does not finish in reasonable time on CPU.

## Before the first run: two paths inside the script

`dual_plm.py` has two absolute paths baked in near the top. Point them at your
own machine, otherwise the run finds no data and no model snapshot:

```python
DNABERT_LOCAL_DIR = '/home/lchen/plm-m7g/DNABERT-2-117M'   # line 40
DATA_DIR          = '/home/lchen/plm-m7g/dataset'         # line 81
```

`DATA_DIR` must contain the four CSVs from `dataset/`. If `DNABERT_LOCAL_DIR`
exists, DNABERT-2 is loaded offline from that snapshot; if it does not, the
script falls back to `zhihan1996/DNABERT-2-117M` through the Hugging Face mirror
`https://hf-mirror.com`.

## Running

```bash
python dual_plm.py
```

It loads RNA-FM (`rna_fm_t12`) and DNABERT-2, then for each dataset runs 5-fold
cross-validation, retrains on the full training split at the selected epoch, and
evaluates the independent test split once. A dataset whose CSVs are missing is
skipped with `[Skip] <name> — files not found.` instead of aborting the run.

Results go to `dual_plm_resultslora16/` next to the script.

## Method

- **Branches**: RNA-FM (sequences T→U) and DNABERT-2 (sequences U→T), each with a
  2-class head of hidden size 256 and dropout 0.4.
- **LoRA**: r=16, α=32, dropout 0.05; RNA-FM targets `q_proj/k_proj/v_proj/out_proj`,
  DNABERT-2 targets `Wqkv/dense`. Everything else stays frozen; if no target
  matches, the run fails loudly rather than silently training the head alone.
- **Optimisation**: AdamW at 3e-4 for the head and 1e-4 for the PLM, weight decay
  2e-3, 10% warm-up then cosine decay, gradient clipping 5.0, label smoothing 0.05,
  balanced class weights, batch 16, seed 42.
- **Model selection**: stratified 5-fold CV on the training split only; the epoch
  is chosen by the mean validation AUC across folds.
- **Fusion**: `p = w · p_rnafm + (1 − w) · p_dnabert2`, with `w` searched on a
  0.05 grid against pooled out-of-fold AUC. The independent test split is scored
  only after `w` is frozen, so test labels never influence selection.
- **Metrics** at threshold 0.5: ACC, SN, SP, PREC, F1, MCC, AUC, plus AUPRC per
  fold and an optimal-threshold view.

## Outputs

```
dual_plm_resultslora16/
├── all_results.json                  # everything, one entry per dataset
└── <dataset>/
    ├── rnafm/  dnabert2/             # per-branch CV tables, OOF/test scores, meta
    ├── ensemble_summary.json
    ├── ensemble_weight_search_by_oof_auc.csv
    ├── ensemble_cv_summary_mean_std_threshold_0.5.csv
    ├── paper_table_cv_mean_std.csv
    ├── paper_table_independent_test.csv
    └── pooled_oof_metrics_threshold_0.5.csv
```

Each branch directory holds `cv_per_epoch_metrics.csv`, the per-fold and summary
tables at the globally selected epoch, `oof_scores.npy`, `oof_labels.npy`,
`oof_fold_ids.npy`, `test_scores.npy`, `test_labels.npy`, the LoRA+head weights
`lora_head_state.pth`, and `pipeline_meta.json`.

**Caching**: `REUSE_CACHED = True` (line 84). When a branch directory already
contains the five `.npy` files, that branch is reused instead of retrained — the
same run mode as before. Set it to `False`, or delete the directory, to force a
fresh training.
