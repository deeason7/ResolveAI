#!/usr/bin/env python3
"""
GPU live-classification benchmark for the ResolveAI fine-tune (built for Colab).

Serves the fine-tuned Qwen2.5-3B (GGUF) on a Colab GPU via Ollama, classifies a
batch of real CFPB complaints exactly the way production does (same system
prompt, same user-prompt layout, same JSON schema), and reports the numbers that
ground the resume:

  * valid-JSON adherence at scale
  * sentiment / intent / urgency distribution over real complaints
  * single-stream serving latency  p50 / p95   (warm-up, concurrency=1)
  * MAX throughput (classifications/sec) under concurrency  (saturates the GPU)

It drives the GPU hard: Ollama is started with a high parallel limit and the
classifications are issued from a thread pool, so a big A100 gets fed instead of
sitting idle on one-at-a-time requests.

------------------------------------------------------------------------------
HOW TO RUN (Google Colab, GPU runtime)
------------------------------------------------------------------------------
  !apt-get -qq install -y zstd
  !pip -q install "instructor>=1.0" "openai>=1.0" pydantic requests
  # mount Drive (model + this script live in MyDrive/resolveai/):
  from google.colab import drive; drive.mount('/content/drive')
  !python /content/drive/MyDrive/resolveai/colab_live_classify.py --n 8000 --concurrency 48
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

# --- production parity: these must match the deployed classifier ---------------
SYSTEM_PROMPT = (
    "You are a financial complaint classifier. "
    "Analyze the complaint and output a structured JSON classification."
)
MODEL_NAME = "resolveai-sentiment"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_GGUF = "/content/drive/MyDrive/resolveai/resolveai-sentiment-q4_k_m.gguf"


def build_user_prompt(narrative, product=None, issue=None, company=None) -> str:
    """Mirror classifier.build_user_prompt byte-for-byte (train/serve parity)."""
    parts = [f"COMPLAINT: {str(narrative).strip()}"]
    if product:
        parts.append(f"PRODUCT: {product}")
    if issue:
        parts.append(f"ISSUE: {issue}")
    if company:
        parts.append(f"COMPANY: {company}")
    return "\n".join(parts)


def write_modelfile(gguf_path: str, path: str = "Modelfile") -> str:
    """Write the Ollama Modelfile (ChatML template + training system prompt)."""
    content = (
        f"FROM {gguf_path}\n\n"
        'TEMPLATE """<|im_start|>system\n'
        "{{ .System }}<|im_end|>\n"
        "<|im_start|>user\n"
        "{{ .Prompt }}<|im_end|>\n"
        "<|im_start|>assistant\n"
        '"""\n\n'
        f'SYSTEM "{SYSTEM_PROMPT}"\n\n'
        "PARAMETER temperature 0.1\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_predict 512\n"
        'PARAMETER stop "<|im_end|>"\n'
    )
    Path(path).write_text(content, encoding="utf-8")
    return path


def ensure_ollama(num_parallel: int) -> None:
    """Install (if needed) and start Ollama tuned for high concurrent throughput."""
    if shutil.which("ollama") is None:
        print("installing ollama...", flush=True)
        # Ollama's installer unpacks a zstd-compressed archive; Colab lacks zstd.
        subprocess.run("apt-get -qq install -y zstd", shell=True, check=False)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    # Tune the server BEFORE it starts: serve up to num_parallel requests at once
    # (continuous batching on the GPU), keep the model resident, flash-attention on.
    env = {
        **os.environ,
        "OLLAMA_NUM_PARALLEL": str(num_parallel),
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_MAX_QUEUE": "4096",
    }
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
            print(f"ollama is up (OLLAMA_NUM_PARALLEL={num_parallel}).", flush=True)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("ollama did not come up within ~2 minutes")


def create_model(gguf_path: str) -> None:
    if not Path(gguf_path).is_file():
        raise SystemExit(
            f"GGUF not found at {gguf_path}\n"
            "Upload resolveai-sentiment-q4_k_m.gguf to Drive (mount it) or pass --gguf PATH."
        )
    write_modelfile(gguf_path)
    print(f"creating ollama model '{MODEL_NAME}' from {gguf_path} ...", flush=True)
    subprocess.run(["ollama", "create", MODEL_NAME, "-f", "Modelfile"], check=True)
    # Warm the weights onto the GPU so the first timed call isn't a cold load.
    subprocess.run(["ollama", "run", MODEL_NAME, "warm up"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fetch_cfpb(n: int) -> list[dict]:
    """Pull up to n CFPB complaints that have narratives, via the public API."""
    # NOTE: no `format=` param — that switches the API into bulk-export mode and
    # streams the ENTIRE dataset (gigabytes), ignoring size/frm. Plain search
    # mode returns paginated JSON (hits.hits) and honors size/frm.
    base = (
        "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
        "?has_narrative=true&no_aggs=true"
    )
    rows: list[dict] = []
    page = 1000
    for frm in range(0, n, page):
        url = f"{base}&size={min(page, n - frm)}&frm={frm}"
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = json.load(r)
        except Exception as exc:  # noqa: BLE001
            print(f"CFPB fetch stopped at frm={frm}: {exc}", flush=True)
            break
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            s = h.get("_source", {})
            narr = s.get("complaint_what_happened")
            if narr and narr.strip():
                rows.append({
                    "narrative": narr, "product": s.get("product"),
                    "issue": s.get("issue"), "company": s.get("company"),
                })
        print(f"  fetched {len(rows)} narratives...", flush=True)
        if len(rows) >= n:
            break
    return rows[:n]


def load_jsonl(path: str, n: int) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= n:
            break
    return rows


def pct(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="how many complaints to classify")
    ap.add_argument("--concurrency", type=int, default=48, help="parallel in-flight requests")
    ap.add_argument("--gguf", default=DEFAULT_GGUF, help="path to the GGUF")
    ap.add_argument("--data", default=None, help="optional jsonl of {narrative,product,issue,company}")
    args = ap.parse_args()

    from pydantic import BaseModel, Field

    class Entity(BaseModel):
        entity: str
        type: str

    class ComplaintClassification(BaseModel):
        sentiment: Literal["neutral", "negative", "extreme_negative"]
        intent: Literal[
            "information_request", "dispute_resolution", "account_action",
            "fraud_report", "regulatory_complaint",
        ]
        urgency: int = Field(ge=1, le=5)
        key_entities: list[Entity] = []
        reasoning: str = ""

    import instructor
    from openai import OpenAI

    ensure_ollama(num_parallel=args.concurrency)
    create_model(args.gguf)

    print(f"\nacquiring {args.n} complaints...", flush=True)
    rows = load_jsonl(args.data, args.n) if args.data else fetch_cfpb(args.n)
    if not rows:
        raise SystemExit("no complaints to classify. Aborting.")

    # One shared client — the OpenAI SDK's httpx core is safe across threads.
    client = instructor.from_openai(
        OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", max_retries=0, timeout=120),
        mode=instructor.Mode.JSON,
    )

    def classify_one(row: dict):
        prompt = build_user_prompt(row.get("narrative"), row.get("product"),
                                   row.get("issue"), row.get("company"))
        t0 = time.time()
        try:
            obj = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                response_model=ComplaintClassification, temperature=0.1, max_retries=1,
            )
            return True, (time.time() - t0) * 1000.0, obj
        except Exception:
            return False, (time.time() - t0) * 1000.0, None

    sentiments: Counter = Counter()
    intents: Counter = Counter()
    urgencies: Counter = Counter()
    valid = 0

    def tally(obj) -> None:
        sentiments[obj.sentiment] += 1
        intents[obj.intent] += 1
        urgencies[obj.urgency] += 1

    # 1) Warm-up: a few sequential calls → clean single-stream latency.
    warm_n = min(40, len(rows))
    warm_lat: list[float] = []
    print(f"\nwarm-up: {warm_n} sequential calls (clean latency)...", flush=True)
    for row in rows[:warm_n]:
        ok, lat, obj = classify_one(row)
        if ok:
            warm_lat.append(lat)
            valid += 1
            tally(obj)

    # 2) Bulk: everything else, concurrent → max throughput.
    bulk = rows[warm_n:]
    print(f"bulk: {len(bulk)} complaints at concurrency={args.concurrency}...\n", flush=True)
    bulk_start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for ok, lat, obj in (f.result() for f in as_completed(ex.submit(classify_one, r) for r in bulk)):
            done += 1
            if ok:
                valid += 1
                tally(obj)
            if done % 200 == 0:
                print(f"  {done}/{len(bulk)}  valid={valid}  ~{done/(time.time()-bulk_start):.1f}/s", flush=True)
    bulk_elapsed = max(time.time() - bulk_start, 1e-6)

    total = len(rows)
    throughput = len(bulk) / bulk_elapsed

    print("\n" + "=" * 64)
    print("RESOLVEAI FINE-TUNE — A100 LIVE CLASSIFICATION")
    print("=" * 64)
    print(f"complaints classified : {total}")
    print(f"valid JSON            : {valid}/{total}  ({100*valid/total:.1f}%)")
    print(f"single-stream latency : p50 {pct(warm_lat,50):.0f} ms / p95 {pct(warm_lat,95):.0f} ms")
    print(f"throughput (conc={args.concurrency:>2}) : {throughput:.1f} classifications/sec "
          f"({len(bulk)} in {bulk_elapsed:.1f}s)")
    print("\n-- sentiment --")
    for k, v in sentiments.most_common():
        print(f"   {k:18} {v:6}  ({100*v/valid:.1f}%)")
    print("-- intent --")
    for k, v in intents.most_common():
        print(f"   {k:22} {v:6}  ({100*v/valid:.1f}%)")
    print("-- urgency (1-5) --")
    for k in sorted(urgencies):
        print(f"   {k}: {urgencies[k]}")
    print("=" * 64)

    Path("live_metrics.json").write_text(json.dumps({
        "n": total, "valid_json": valid, "valid_json_pct": round(100 * valid / total, 2),
        "single_stream_p50_ms": round(pct(warm_lat, 50), 1) if warm_lat else None,
        "single_stream_p95_ms": round(pct(warm_lat, 95), 1) if warm_lat else None,
        "throughput_per_sec": round(throughput, 2), "concurrency": args.concurrency,
        "bulk_seconds": round(bulk_elapsed, 1),
        "sentiment": dict(sentiments), "intent": dict(intents), "urgency": dict(urgencies),
    }, indent=2), encoding="utf-8")
    print("\nwrote live_metrics.json — download it and send it back for the resume bullets.")


if __name__ == "__main__":
    main()
