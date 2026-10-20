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
