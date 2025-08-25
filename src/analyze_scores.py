#!/usr/bin/env python3
"""
analyze_scores.py

Merge human ratings, compute reliability (Cohen's kappa, ICC),
aggregate rubric means/SEMs by model, FNMWCF coverage, marker frequencies,
and readability metrics (FKGL, sentence length, chars/word).

Inputs:
  --ratings_dir   directory of individual rater CSVs with columns:
                  response_id,model,accuracy,cultural_relevance,language_accessibility,bias_avoidance, rater_id
  --raw_dir       outputs/raw/*.csv from run_ollama_batch.py (for response text & metadata)
  --fnmwcf_json   JSON mapping of FNMWCF themes -> list of proxy keywords
Outputs (to outputs/analysis/):
  - merged_ratings.csv (long and wide)
  - reliability.json (per-axis kappa + overall ICC)
  - summary_by_model.csv (means, SEMs)
  - fnmwcf_coverage.csv
  - markers.csv (crisis, disclaimer/referral, cultural-reference freq)
  - readability.csv (FKGL + proxies)
"""

from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import defaultdict

import numpy as np
import pandas as pd

def _fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def read_rater_files(ratings_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(ratings_dir, "*.csv"))
    if not files:
        _fail(f"no rater CSVs found in {ratings_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        need = {"response_id","model","accuracy","cultural_relevance","language_accessibility","bias_avoidance","rater_id"}
        if not need.issubset(set(df.columns)):
            _fail(f"{f} missing required columns: {need - set(df.columns)}")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # normalize
    out["model"] = out["model"].astype(str)
    out["rater_id"] = out["rater_id"].astype(str)
    return out

def wide_from_ratings(r: pd.DataFrame) -> pd.DataFrame:
    """
    pivot to one row per response_id per rater, also build reconciled mean per response if needed
    """
    # ensure numeric
    for c in ["accuracy","cultural_relevance","language_accessibility","bias_avoidance"]:
        r[c] = pd.to_numeric(r[c], errors="coerce").clip(0,3)
    return r

# --- Reliability: Cohen's kappa per axis (pairwise) + ICC(A,1) overall ---
def cohen_kappa_two_raters(df: pd.DataFrame, axis: str) -> float:
    # expect exactly 2 ratings per response_id for the axis
    from sklearn.metrics import cohen_kappa_score
    # align pairs
    pivot = df.pivot_table(index="response_id", columns="rater_id", values=axis, aggfunc="first").dropna()
    if pivot.shape[1] != 2 or pivot.empty:
        return np.nan
    a = pivot.iloc[:,0].round().astype(int)
    b = pivot.iloc[:,1].round().astype(int)
    return float(cohen_kappa_score(a, b))

def icc_a1(df: pd.DataFrame, axes: list[str]) -> float:
    """
    Simple ICC(A,1) estimate across all rubric axes by stacking them.
    Each response_id×axis gets two ratings.
    """
    # reshape to long by axis
    pieces = []
    for ax in axes:
        tmp = df[["response_id","rater_id",ax]].rename(columns={ax:"score"}).dropna()
        tmp["axis"] = ax
        pieces.append(tmp)
    long = pd.concat(pieces, ignore_index=True)
    # pivot to n_subjects × n_raters per (axis) then concatenate
    mats = []
    for ax, group in long.groupby("axis"):
        pv = group.pivot_table(index="response_id", columns="rater_id", values="score", aggfunc="first").dropna()
        if pv.shape[1] < 2:  # need >=2 raters
            continue
        mats.append(pv)
    if not mats:
        return np.nan
    X = pd.concat(mats, axis=0)
    # ICC(A,1) = (MS_between - MS_error) / (MS_between + (k-1)MS_error)
    n, k = X.shape
    grand = X.values.mean()
    ms_between = k * ((X.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_within = ((X - X.mean(axis=1).values[:,None]) ** 2).sum().sum() / (n*(k-1))
    icc = (ms_between - ms_within) / (ms_between + (k-1)*ms_within)
    return float(icc)

def read_raw(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not files:
        _fail(f"no raw outputs found in {raw_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        need = {"prompt_id","model","response_clean"}
        if not need.issubset(set(df.columns)):
            _fail(f"{f} missing {need - set(df.columns)}")
        df["response_id"] = df["model"].astype(str) + "::" + df["prompt_id"].astype(str)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out

def merge_ratings_with_text(ratings: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    # assume ratings has response_id & model
    merged = ratings.merge(raw[["response_id","prompt_id","model","response_clean","category","province","sex","age_group","mental_health","language"]],
                           on=["response_id","model"], how="left")
    return merged

# --- FNMWCF coverage ---
def load_fnmwcf(path: str) -> dict[str,list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not obj:
        _fail("fnmwcf JSON must map theme -> list[keywords]")
    # lower-case keywords
    for k,v in obj.items():
        obj[k] = [w.lower() for w in v]
    return obj

def fnmwcf_hit(row_txt: str, theme_words: list[str], prompt_txt: str|None) -> bool:
    if not isinstance(row_txt, str):
        return False
    text = row_txt.lower()
    # optional prompt leakage rule: only count if theme term not present in prompt
    if isinstance(prompt_txt, str) and prompt_txt:
        p = prompt_txt.lower()
        if any(w in p for w in theme_words):
            return False
    return any(w in text for w in theme_words)

# --- Markers: crisis guidance, disclaimers, cultural reference ---
CRISIS = re.compile(r"(crisis\s*line|call\s*911|emergency|suicide\s*hotline|seek\s*immediate\s*help)", re.I)
DISCLAIMER = re.compile(r"(not\s+medical\s+advice|consult\s+(a|your)\s+(doctor|clinician|provider))", re.I)
CULTURAL = re.compile(r"(community|tradition|elder|land|language|culture|ceremony)", re.I)

def compute_markers(df_text: pd.DataFrame) -> pd.DataFrame:
    out = df_text.copy()
    txt = out["response_clean"].fillna("")
    out["marker_crisis"] = txt.str.contains(CRISIS)
    out["marker_disclaimer"] = txt.str.contains(DISCLAIMER)
    out["marker_cultural"] = txt.str.contains(CULTURAL)
    rates = out.groupby("model")[["marker_crisis","marker_disclaimer","marker_cultural"]].mean().reset_index()
    return rates

# --- Readability ---
def readability_metrics(df_text: pd.DataFrame) -> pd.DataFrame:
    try:
        from textstat import textstat
    except Exception:
        _fail("Please `pip install textstat` to compute FKGL.")

    def sent_len(s: str) -> float:
        # rough sentence split
        parts = re.split(r"[.!?]+", s)
        words = [len(p.split()) for p in parts if p.strip()]
        return np.mean(words) if words else np.nan

    def chars_per_word(s: str) -> float:
        ws = s.split()
        lens = [len(w) for w in ws]
        return np.mean(lens) if lens else np.nan

    df = df_text.copy()
    df["fkgl"] = df["response_clean"].fillna("").map(textstat.flesch_kincaid_grade)
    df["avg_sentence_len"] = df["response_clean"].fillna("").map(sent_len)
    df["chars_per_word"] = df["response_clean"].fillna("").map(chars_per_word)

    by_model = df.groupby("model").agg(
        fkgl=("fkgl","mean"),
        avg_sentence_len=("avg_sentence_len","mean"),
        chars_per_word=("chars_per_word","mean"),
    ).reset_index()
    return by_model

def summarize_by_model(merged: pd.DataFrame) -> pd.DataFrame:
    axes = ["accuracy","cultural_relevance","language_accessibility","bias_avoidance"]
    agg = merged.groupby("model")[axes].agg(["mean","sem"])
    # flatten
    agg.columns = [f"{a}_{b}" for a,b in agg.columns]
    agg = agg.reset_index()
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_dir", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--fnmwcf_json", required=True)
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ratings = read_rater_files(args.ratings_dir)
    raw = read_raw(args.raw_dir)
    raw["response_id"] = raw["model"].astype(str) + "::" + raw["prompt_id"].astype(str)
    merged = merge_ratings_with_text(ratings, raw)

    merged.to_csv(os.path.join(args.outdir, "merged_ratings.csv"), index=False)

    # Reliability
    kappas = {}
    for ax in ["accuracy","cultural_relevance","language_accessibility","bias_avoidance"]:
        kappas[ax] = float(cohen_kappa_two_raters(merged[["response_id","rater_id",ax]].dropna(), ax))
    icc = icc_a1(merged[["response_id","rater_id","accuracy","cultural_relevance","language_accessibility","bias_avoidance"]].dropna(),
                 ["accuracy","cultural_relevance","language_accessibility","bias_avoidance"])
    with open(os.path.join(args.outdir, "reliability.json"), "w") as f:
        json.dump({"cohen_kappa": kappas, "icc_a1": icc}, f, indent=2)

    # Summary by model
    summary = summarize_by_model(merged)
    summary.to_csv(os.path.join(args.outdir, "summary_by_model.csv"), index=False)

    # FNMWCF coverage
    fnmwcf = load_fnmwcf(args.fnmwcf_json)
    # we need prompt_text, bring it from raw join on response_id if not present already
    full = raw[["response_id","model","prompt_id","prompt_text","response_clean"]].copy()
    hits = []
    for _, r in full.iterrows():
        for theme, words in fnmwcf.items():
            covered = fnmwcf_hit(r["response_clean"], words, r.get("prompt_text"))
            hits.append({"model": r["model"], "theme": theme, "covered": int(covered)})
    cov = pd.DataFrame(hits).groupby(["model","theme"])["covered"].mean().reset_index()
    cov.to_csv(os.path.join(args.outdir, "fnmwcf_coverage.csv"), index=False)

    # Markers
    markers = compute_markers(raw)
    markers.to_csv(os.path.join(args.outdir, "markers.csv"), index=False)

    # Readability
    readab = readability_metrics(raw)
    readab.to_csv(os.path.join(args.outdir, "readability.csv"), index=False)

    print("[OK] analysis complete")
    print(f"  -> {os.path.join(args.outdir, 'merged_ratings.csv')}")
    print(f"  -> {os.path.join(args.outdir, 'reliability.json')}")
    print(f"  -> {os.path.join(args.outdir, 'summary_by_model.csv')}")
    print(f"  -> {os.path.join(args.outdir, 'fnmwcf_coverage.csv')}")
    print(f"  -> {os.path.join(args.outdir, 'markers.csv')}")
    print(f"  -> {os.path.join(args.outdir, 'readability.csv')}")

if __name__ == "__main__":
    main()
