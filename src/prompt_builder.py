#!/usr/bin/env python3
"""
prompt_builder.py

Build culturally grounded prompts from:
- CCHS personas CSV (e.g., cchs_sample_5k.csv) with columns like:
  province, sex, age_group, mental_health (e.g., "stress"/"depression"), language
- A plain-text prompt template with placeholders:
  {province} {sex} {age_group} {mental_health} {language}
- A JSON file of FNMWCF-aligned categories (5 buckets × 4 items each)

Outputs:
- prompts.csv with columns: prompt_id, category, persona_id, province, sex, age_group,
  mental_health, language, prompt_text
"""

from __future__ import annotations
import argparse, json, os, sys
import pandas as pd

def _fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def load_personas(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        _fail(f"personas file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    needed = {"province", "sex", "age_group", "mental_health", "language"}
    missing = needed - set(c.lower() for c in df.columns)
    if missing:
        _fail(f"personas missing columns: {missing}")
    # normalize names
    df.columns = [c.lower() for c in df.columns]
    # add persona_id if not present
    if "persona_id" not in df.columns:
        df["persona_id"] = range(1, len(df) + 1)
    return df

def load_template(path: str) -> str:
    if not os.path.exists(path):
        _fail(f"template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_categories(path: str) -> dict:
    if not os.path.exists(path):
        _fail(f"categories json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # expected keys: language_access, cultural_relevance, low_health_literacy,
    # marginalized_needs, misinformation_resistance
    if not isinstance(obj, dict) or not obj:
        _fail("categories JSON must be a non-empty object")
    return obj

def build_prompts(personas: pd.DataFrame, tmpl: str, cats: dict) -> pd.DataFrame:
    rows = []
    pid = 0
    for _, p in personas.iterrows():
        persona_vars = {
            "province": p["province"],
            "sex": p["sex"],
            "age_group": p["age_group"],
            "mental_health": p["mental_health"],
            "language": p["language"],
        }
        for category, items in cats.items():
            for item in items:
                pid += 1
                # Each item can inject a short scenario/framing
                prompt_text = tmpl.format(**persona_vars, scenario=item)
                rows.append({
                    "prompt_id": f"P{pid:06d}",
                    "category": category,
                    "persona_id": p["persona_id"],
                    **persona_vars,
                    "prompt_text": prompt_text
                })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True, help="CSV with CCHS personas")
    ap.add_argument("--template", required=True, help="Prompt template .txt with placeholders")
    ap.add_argument("--categories", required=True, help="JSON of category->list of items")
    ap.add_argument("--out", default="outputs/prompts.csv", help="where to write prompts CSV")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    personas = load_personas(args.personas)
    tmpl = load_template(args.template)
    cats = load_categories(args.categories)
    df = build_prompts(personas, tmpl, cats)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {len(df)} prompts -> {args.out}")

if __name__ == "__main__":
    main()
