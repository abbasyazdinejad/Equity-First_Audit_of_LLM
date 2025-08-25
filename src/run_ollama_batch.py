#!/usr/bin/env python3
"""
run_ollama_batch.py

Execute prompts across multiple open-weight LLMs served by Ollama.
- Respects fixed decoding params to ensure comparability.
- Fault-tolerant logging (resume if interrupted).
- Cleans out <think> ... </think> style artifacts if present.

Inputs:
  --prompts CSV with columns: prompt_id, prompt_text, [metadata...]
  --models  Comma-separated list of Ollama model names (e.g., llama3.2:latest,mistral:7b,deepseek-r1:8b)
Outputs:
  CSV per model in outputs/raw/<model_sanitized>.csv with columns:
  [prompt_id, model, timestamp, prompt_text, response_raw, response_clean, ...metadata]
"""

from __future__ import annotations
import argparse, csv, datetime as dt, os, re, sys, time
import pandas as pd
import requests

CLEAN_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL|re.IGNORECASE)
RETRY_CODES = {429, 500, 502, 503, 504}

def _fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def load_prompts(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        _fail(f"prompts CSV not found: {path}")
    df = pd.read_csv(path)
    if "prompt_id" not in df.columns or "prompt_text" not in df.columns:
        _fail("prompts CSV must include prompt_id and prompt_text")
    return df

def ollama_generate(model: str, prompt: str, temperature: float, max_tokens: int, stop: list[str] | None) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if stop:
        payload["stop"] = stop
    # stream=false -> we get one JSON
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code in RETRY_CODES:
        raise RuntimeError(f"HTTP {r.status_code}")
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def clean_response(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    return CLEAN_THINK.sub("", txt).strip()

def append_rows(out_csv: str, rows: list[dict]) -> None:
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--models", required=True, help="comma-separated Ollama models")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--stop", default="", help="comma-separated stop sequences")
    ap.add_argument("--outdir", default="outputs/raw")
    ap.add_argument("--resume", action="store_true", help="skip already-seen prompt_ids for each model")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stop = [s for s in args.stop.split(",") if s] or None
    os.makedirs(args.outdir, exist_ok=True)

    for model in models:
        out_csv = os.path.join(args.outdir, f"{sanitize(model)}.csv")
        seen = set()
        if args.resume and os.path.exists(out_csv):
            prev = pd.read_csv(out_csv, usecols=["prompt_id"])
            seen = set(prev["prompt_id"].astype(str))
            print(f"[{model}] resume enabled: skipping {len(seen)} already logged rows")

        batch = []
        for _, row in prompts.iterrows():
            pid = str(row["prompt_id"])
            if pid in seen:
                continue
            prompt_text = str(row["prompt_text"])
            ts = dt.datetime.utcnow().isoformat()

            # retry loop
            for attempt in range(4):
                try:
                    resp = ollama_generate(model, prompt_text, args.temperature, args.max_tokens, stop)
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[{model}] attempt {attempt+1} failed: {e}; retry in {wait}s")
                    time.sleep(wait)
            else:
                resp = ""

            clean = clean_response(resp)
            out_row = {
                "prompt_id": pid,
                "model": model,
                "timestamp_utc": ts,
                "prompt_text": prompt_text,
                "response_raw": resp,
                "response_clean": clean,
            }
            # include persona metadata if present
            for meta in ("category","persona_id","province","sex","age_group","mental_health","language"):
                if meta in row:
                    out_row[meta] = row[meta]
            batch.append(out_row)

            # flush periodically
            if len(batch) >= 50:
                append_rows(out_csv, batch)
                print(f"[{model}] wrote {len(batch)} rows -> {out_csv}")
                batch = []

        if batch:
            append_rows(out_csv, batch)
            print(f"[{model}] wrote final {len(batch)} rows -> {out_csv}")

    print("[OK] batch completed")

if __name__ == "__main__":
    main()

