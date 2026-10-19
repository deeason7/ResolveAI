# Fine-tuning guide

The repository contains a classifier training experiment based on
Qwen2.5-3B-Instruct and QLoRA, plus evaluation, GGUF export and Ollama smoke-test
scripts. Local inference is optional. Production environment examples set
`LLM_SKIP_LOCAL=true` and use cloud inference by default.

## Evidence status — reviewed 2026-10-17

The available experiment is **not a clean held-out evaluation**. The local split
artifacts contain 7,978 training, 995 validation and 995 test records. Of the test
set, 185 narratives occur in training (18.59%); 127 full user prompts also match
training prompts. Validation overlaps too. Teacher labels are not independently
established ground truth: the available 100-row human-review sheet has no completed
agreement entries. This does not establish whether other review occurred.

The local `fine_tuning/results/metrics_adapter.json` and `eval_summary.md` preserve
these historical results; result artifacts are gitignored and are not supplied by
a fresh clone. Their run date and exact model revision are not established by the
available metadata. Do not cite them as current model quality or a measured
improvement on unseen complaints.

| Historical artifact value | Interpretation and limit |
|---|---|
| Sentiment/intent macro-F1: 0.6291 / 0.6312 | Raw report averages include a zero-support `INVALID` class |
| Earlier documentation: approximately 0.84 / 0.76 | Excluding that extra class gives approximately 0.8388 / 0.7574; this explains the reporting difference, not a new evaluation |
| JSON objects parsed: 992/995 (99.7%) | Parseability; not strict schema validity, label correctness or safe output |
| Urgency MAE 0.223; Spearman 0.845 | Teacher-label agreement on the contaminated test split |

The base-model artifact contains 878 JSON objects and 117 missing/unparseable
outputs, but no requested taxonomy keys. Its prompt does not specify the complete
output contract. Zero baseline F1 therefore does not establish that fine-tuning
outperforms a properly prompted base model. Provide both models the same taxonomy,
schema, examples, input context and output limits before comparing them.

Earlier documentation also quoted a 100-sample quantization check and an
8,000-record runtime benchmark. Supporting predictions/model artifacts for those
claims were unavailable during this review. Their values are withdrawn as quality
or throughput evidence pending a reproducible rerun. Existing files are preserved;
no historical artifact has been rewritten to imply a better result.

## Existing scripts

| Step | Script | Purpose and current limit |
|---|---|---|
| Label | `01_prepare_labels.py` | Provider-backed teacher labeling, with resume/provenance support; calls can cost money and transmit complaint text |
| Audit | `audit_labels.py` | Class balance, narrative length, lexical patterns, consistency and human-review sample; generated audit rows are not completed human labels |
| Format | `02_format_training_data.py` | ChatML and stratified row split; currently lacks duplicate-group isolation |
| Train | `03_train_qlora.py`, `colab_train.ipynb` | QLoRA, class weights and assistant-token supervision; inspect the effective config because compatibility filtering can omit requested options |
| Evaluate | `04_evaluate.py` | Classification metrics and parse checks; fair prompts, strict schema checks and leakage controls need revision |
| Export | `05_export_gguf.py` | Merge adapter, convert to GGUF, quantize and write an Ollama Modelfile |
| Smoke test | `06_smoke_test.py` | Call the selected Ollama model and measure sample behavior; not a substitute for independent evaluation |

A small local model could reduce external inference needs, but hardware, energy,
maintenance and quality trade-offs remain. No total-cost or latency advantage has
been established against a fair task-specific baseline.

## Before another training or evaluation run

1. Normalize and hash narratives, group duplicate/near-duplicate cases, and create
   disjoint train/validation/test groups. Save input hashes, group assignments,
   seed and data provenance. Consider temporal/company holdouts where relevant.
2. Independently annotate a frozen evaluation sample with an explicit rubric.
   Record disagreements and adjudication; do not call teacher labels human gold.
3. Compare a simple rules/text baseline, a fully prompted base model and the
   adapter on the same contract. Count malformed, invalid-taxonomy and missing
   outputs separately, including them in failure denominators.
4. Record exact model, adapter, tokenizer, prompt, dependencies, quantization,
   runtime hardware and effective training configuration. Fail unsupported required
   settings instead of silently dropping them. Persist predictions and metrics.
5. Report minority-class support/recall, confusion matrices, urgency error and
   high-priority recall within a fixed reviewer budget. Measure uncertainty and
   cost/latency on the actual deployment configuration.
6. Only then evaluate the exported quantized model against the same frozen sample.
   Parse rate alone cannot establish that quantization preserved task quality.

The existing scripts can be inspected and run in an isolated experiment, but
repeating their current defaults reproduces the existing limitations. Training
and paid labeling are separate experiments, not prerequisites for repairing the
application's access control and review workflow.

## Runtime benchmark boundaries

`colab_live_classify.py` measures classifier serving under a particular GPU and
Ollama configuration. `agent_benchmark.py` exercises the orchestrator and checks
with stub vector/graph stores and seeded reference material. Available local
agent results include 23 scenarios, 20 passes and 3 judge-unavailable outcomes;
that is a small workflow experiment, not a legal-safety evaluation or evidence of
retrieval quality. Keep future reports tied to their input fixtures and failures.
