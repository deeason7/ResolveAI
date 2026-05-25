# Fine-Tuning Data

Working directories for the QLoRA pipeline. Everything below is gitignored — the artifacts are reproducible from the scripts plus a Groq API key.

## Layout

| Directory | Contents | Built by |
|---|---|---|
| `raw/` | The CFPB CSV sample (200K rows, ~190 MB). | `scripts/download_cfpb_data.py` |
| `processed/` | Intermediate parquet/csv after cleaning. | _(reserved — not currently populated)_ |
| `labeled/` | `gold_labels.jsonl` — teacher LLM classifications, one per complaint. | `fine_tuning/01_prepare_labels.py` |
| `formatted/` | `train.jsonl`, `val.jsonl`, `test.jsonl` — ChatML splits for the trainer. | `fine_tuning/02_format_training_data.py` |
| `audit/` | `distribution.json`, `audit_report.md`, `sample_for_review.csv` — label QA artifacts (distribution stats, leakage checks, stratified human-review sample). | `fine_tuning/audit_labels.py` |

## Regenerating

From a fresh checkout:

```bash
# 1. Pull the CFPB sample (~1.8 GB download, samples 200K rows)
docker compose exec api python scripts/download_cfpb_data.py

# 2. Ingest into Postgres
docker compose exec api python scripts/import_cfpb.py

# 3. Label 10K complaints with Groq (free tier ~50 days unattended, dev tier ~6 hours)
docker compose exec api python fine_tuning/01_prepare_labels.py --limit 10000

# 4. Audit the label distribution and skim a stratified human-review sample
docker compose exec api python fine_tuning/audit_labels.py \
  --label-source "bedrock:us.meta.llama3-3-70b-instruct-v1:0"

# 5. Format into train/val/test splits
docker compose exec api python fine_tuning/02_format_training_data.py
```

## Why gitignored

- `raw/` — 190 MB CFPB CSV. Public data, fetchable on demand.
- `labeled/` — contains complaint narratives, which CFPB scrubs of PII but may not be perfectly clean. Treat as sensitive.
- `formatted/` — derivable from `labeled/` plus the formatting script.
- `models/` — fine-tuned adapter weights and merged GGUF artifacts. Multi-GB; lives on HF Hub / Drive.
- `audit/` — derivable from `labeled/` plus the audit script; the CSV sample contains narrative excerpts.
