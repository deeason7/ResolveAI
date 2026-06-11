# Fine-tuning guide

How the classifier was built, end to end, and how to reproduce it. The
pipeline lives in `fine_tuning/` as six numbered scripts plus a Colab
notebook; every step is resumable and auditable.

## Why fine-tune at all

Classification runs on every complaint — at corpus scale, per-call cloud
pricing and latency dominate. A 3B model fine-tuned for exactly this task
runs locally for free, returns structured JSON reliably, and the cloud stays
available as a fallback rather than the default. The interesting engineering
is in making a small model trustworthy: rubric-anchored labels, leak audits,
class-weighted loss, and a deployment re-check after quantization.

## The pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_prepare_labels.py` | Teacher-labels ~10K complaints against a 2.2K-token anchored rubric. Provider-pluggable (`--provider groq\|bedrock`); the production run used **Bedrock Llama 3.3 70B** (9,956 labels, 0 errors, ~53 min, ≈$23). Resumable, daily-cap aware, provenance-tracked into `complaint_labels`. |
| 2 | `audit_labels.py` | Label QA before any training: class balance, narrative-length-per-class (length leak), top-tokens-per-class (lexical leak), sentiment↔urgency consistency, reasoning-template collapse, stratified sample CSV for human review. |
| 3 | `02_format_training_data.py` | ChatML conversations, stratified 80/10/10 split (preserves the 2% neutral class in every split). |
| 4 | `03_train_qlora.py` + `colab_train.ipynb` | QLoRA on **Qwen2.5-3B-Instruct**: 4-bit NF4 + double quantization, LoRA r=16 / α=16 on all linear layers, paged 8-bit AdamW, cosine LR + 5% warmup, 3 epochs at effective batch 16, **class-weighted cross-entropy** (√-inverse-frequency: neutral 4.00 / negative 0.79 / extreme 0.86), **`assistant_only_loss`** so the model is supervised only on its answer, not on the complaint text. |
| 5 | `04_evaluate.py` | Held-out eval: per-class P/R/F1, urgency MAE + Spearman, % valid strict JSON, confusion matrices, length-bucketed accuracy (a direct answer to the audit's length-leak finding), `--no-adapter` baseline for a clean delta. |
| 6 | `05_export_gguf.py` | Merge adapter → GGUF f16 → quantize **Q4_K_M (1.9 GB)** via llama.cpp; generates the Ollama Modelfile with the *training* system prompt (train/serve prompt parity is asserted by a locked test). |
| 7 | `06_smoke_test.py` | Hits the real Ollama API with the deployed model: structured-output parse rate + agreement vs gold. |

## Results

Held-out set (995 examples), adapter vs base:

| Metric | Fine-tuned | Base Qwen2.5-3B |
|---|---|---|
| Sentiment accuracy / macro-F1 | **0.90 / 0.84** | taxonomy non-conformant (macro-F1 ≈ 0) |
| Intent accuracy / macro-F1 | **0.85 / 0.76** | — |
| Urgency MAE / Spearman | **0.223 / 0.845** | — |
| Valid structured JSON | **99.7%** | 88% parseable, wrong vocabulary |

The base model "speaks" but doesn't conform — it answers in its own labels.
That's the fine-tune's real win: vocabulary + format discipline, not just
accuracy.

Post-quantization deployment check (100 samples through the real Ollama
API): 100% parseable, 89% sentiment accuracy, urgency MAE 0.25 —
**Q4_K_M held**.

## Reproducing it

1. **Labels** — needs Postgres up with the corpus imported, plus AWS creds
   (Bedrock) or a Groq key:

   ```bash
   docker compose exec api python /fine_tuning/01_prepare_labels.py \
     --provider bedrock --limit 10000
   docker compose exec api python /fine_tuning/audit_labels.py
   docker compose exec api python /fine_tuning/02_format_training_data.py
   ```

2. **Training** — upload `colab_train.ipynb` + the JSONL splits to Google
   Drive, run on a T4/A100 (the config flips bf16 on automatically when the
   GPU supports it). Training writes checkpoints to Drive, so a Colab VM
   death loses nothing.

3. **Sanity-check the loss curve early.** First-step loss should be single
   digits. A loss of ~19 means assistant-only masking isn't active and the
   model is being trained to predict the user's complaint text — fix the
   config, don't wait out the run.

4. **Evaluate** on the held-out split (the notebook's batched bf16 harness
   does adapter + baseline in ~40 min on A100; the per-sample 4-bit path is
   ~50× slower — don't use it for full runs).

5. **Export + deploy locally** (needs `brew install llama.cpp`, plus the
   repo clone for the `convert_hf_to_gguf.py` script — the brew formula
   doesn't ship it):

   ```bash
   python fine_tuning/05_export_gguf.py --quant q4_k_m
   ollama create resolveai-sentiment -f fine_tuning/models/Modelfile
   python fine_tuning/06_smoke_test.py --batch 100
   ```

## Gotchas that cost real time (so you don't pay twice)

- **`assistant_only_loss` is the whole ballgame** — see step 3 above.
- **TRL renames kwargs across minor versions** (`max_seq_length` →
  `max_length`); the trainer introspects `SFTConfig`'s signature and routes
  kwargs instead of pinning to one TRL release.
- **`load_best_model_at_end` can leave the top-level adapter dir empty** —
  the export and eval scripts fall back to the latest `checkpoint-*/`
  automatically.
- **Drive zips folders on download** — the exported `.gguf` may arrive
  inside a `gguf-*.zip`; the file you want is 1.9 GB, not 40 KB.
- **Class imbalance compounds**: 2% neutral survived only because the split
  is stratified *and* the loss is class-weighted *and* the audit checks
  per-class recall. Any one alone wasn't enough.
