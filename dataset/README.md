# Benchmark data dictionary and provenance status

## Scope and file inventory

These are processed sequence-window classification datasets supplied by the
authors, not raw sequencing reads. Files, row order and sequence strings were
copied without alteration. Original byte identities are listed in SHA256SUMS
and manifest.json.

| File | Role | Rows | Positive | Negative | Length |
|---|---|---:|---:|---:|---:|
| train_mostdistant401.csv | Full transcript training | 8988 | 4494 | 4494 | 401 nt |
| test_mostdistant401.csv | Full transcript independent test | 2248 | 1124 | 1124 | 401 nt |
| train_mature_mostdistant201.csv | Mature RNA training | 2386 | 1193 | 1193 | 201 nt |
| test_mature_mostdistant201.csv | Mature RNA independent test | 596 | 298 | 298 | 201 nt |

## Columns

| Field | Observed format / use | Author confirmation needed |
|---|---|---|
| chr_id | String supplied with each window; not used as a model feature | Exact identifier/coordinate convention, reference assembly and strand meaning |
| Sequence | Uppercase A/C/G/T string, exactly 401 or 201 characters | Extraction/reference transcript version and orientation protocol |
| Label | Integer 0 or 1; code predicts class-1 probability | Label 1 is used as positive and 0 as negative; document the experimental positive evidence and negative sampling/definition |

All supplied sequences have a central G (zero-based index 200 or 100). The CSV
alphabet is DNA-style even though the task concerns RNA. The model's RNA-FM
branch converts T to U; the DNABERT-2 branch uses U-to-T normalization. No data
file conversion, cropping or padding was performed for this release.

## Train/test use and audit

The existing train/test assignments are preserved. Only training rows enter
five-fold CV, epoch selection and OOF fusion-weight selection. Test labels are
used for final metrics. No within-file exact duplicate sequences or exact
train/test sequence overlaps were found within either benchmark in the supplied
files. This is not a homology or transcript-level leakage assessment and does
not validate the biological truth of the labels or the underlying split design.

Run `python3 run.py --audit-only --dataset both` from the repository root to
repeat the file identity and schema checks. Hashes also depend on file bytes,
including row order and line endings; preserve these files for reproduction.

## Original source and processing: TO BE COMPLETED BY THE AUTHORS

- Original dataset-generating paper / DOI: not yet confirmed for this release.
- Original database accession, repository version and download URL: not yet confirmed.
- Processing/extraction script and genome/transcript annotation version: not yet confirmed.
- Positive-site evidence and negative sampling protocol: to be documented.
- Meaning of `mostdistant` and the partition/deduplication protocol: to be documented.
- Transcript biotype composition and any exclusions: to be documented; do not
  infer that every row is mRNA or that tRNA is absent from file names alone.
- Permission or licence allowing redistribution of these processed files:
  not yet confirmed. A publicly accessible original paper is not itself a data licence.
- Dataset author/creator names and preferred citation: to be confirmed.

## Access and rights

Included in this local private release-preparation package for verification.
Do not make this package public until the authors establish redistribution rights
and complete the source citation. No data licence is assigned here. If the data
cannot be redistributed, replace the CSVs in the public release with approved
download/reconstruction instructions and preserve the file manifest; rerun
validation on that revised release.
