#!/usr/bin/env python3
"""
audit_plots.py

Generate publication-ready figures for the Equity-First LLM Audit:
1) Rubric heatmap (Accuracy, Cultural relevance, Language accessibility, Bias avoidance; 0–3)
2) Flesch–Kincaid Grade Level (FKGL) bar chart with recommended Grade 6–8 band

INPUTS
------
--scores : CSV file with model-level rubric scores. Supports:
    Wide format  : model,accuracy,cultural_relevance,language_accessibility,bias_avoidance
    Long format  : model,dimension,score   (dimension ∈ {accuracy,cultural_relevance,language_accessibility,bias_avoidance})
--responses : CSV of cleaned responses used to compute FKGL per model.
    Must contain columns: model, response_text
    If textstat is unavailable, you can pass --fkgl to provide a precomputed CSV.

OPTIONAL
--------
--fkgl : CSV with columns model,fkgl to skip computing grade levels from text.
--outdir : Directory to save figures (default: results/figures)

USAGE EXAMPLES
--------------
# Heatmap from scores (wide) + FKGL computed from responses
python src/audit_plots.py --scores data/mean_scores_by_model.csv --responses data/responses_clean.csv

# Heatmap from scores (long) + FKGL from a small CSV
python src/audit_plots.py --scores data/mean_scores_long.csv --fkgl data/fkgl_by_model.csv

# Custom output directory
python src/audit_plots.py --scores data/mean_scores_by_model.csv --responses data/responses_clean.csv --outdir outputs/figures

Author: (c) 2025 Your Name. MIT License.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/servers
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
RUBRIC_AXES = [
    "accuracy",
    "cultural_relevance",
    "language_accessibility",
    "bias_avoidance",
]

RUBRIC_AXIS_LABELS = [
    "Accuracy",
    "Cultural relevance",
    "Language accessibility",
    "Bias avoidance",
]


def _fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_scores(scores_path: str) -> pd.DataFrame:
    """
    Read rubric scores. Accepts:
      - Wide format: columns = [model] + RUBRIC_AXES
      - Long format: columns = [model, dimension, score]
    Returns a DataFrame indexed by model with columns in RUBRIC_AXES order.
    """
    if not os.path.exists(scores_path):
        _fail(f"Scores file not found: {scores_path}")

    df = pd.read_csv(scores_path)
    cols = set(c.lower() for c in df.columns)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Wide format?
    if {"model", *RUBRIC_AXES}.issubset(cols):
        wide = df.copy()
        wide = wide[["model"] + RUBRIC_AXES]
        # Ensure float and clip to [0,3]
        for ax in RUBRIC_AXES:
            wide[ax] = pd.to_numeric(wide[ax], errors="coerce").clip(0, 3)
        wide = wide.dropna()
        wide = wide.set_index("model")
        return wide

    # Long format?
    elif {"model", "dimension", "score"}.issubset(cols):
        long = df.copy()
        long["dimension"] = long["dimension"].str.strip().str.lower()
        long = long[long["dimension"].isin(RUBRIC_AXES)]
        long["score"] = pd.to_numeric(long["score"], errors="coerce").clip(0, 3)
        wide = long.pivot_table(index="model", columns="dimension", values="score", aggfunc="mean")
        # Reorder columns
        wide = wide.reindex(columns=RUBRIC_AXES)
        return wide

    else:
        _fail(
            "Scores CSV must be either wide (model + four rubric columns) "
            "or long (model,dimension,score)."
        )
    return pd.DataFrame()


def plot_rubric_heatmap(wide: pd.DataFrame, outpath: str) -> None:
    """
    wide: index = model, columns = RUBRIC_AXES with values 0–3
    """
    if wide.empty:
        _fail("No scores to plot in heatmap.")
    models = list(wide.index)
    data = wide.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=3)

    # Ticks/labels
    ax.set_xticks(np.arange(len(RUBRIC_AXES)))
    ax.set_xticklabels(RUBRIC_AXIS_LABELS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)

    # Annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=10, color="black")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rubric score (0–3)", rotation=90)

    ax.set_title("Equity rubric scores by model (means over double independent ratings)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def compute_fkgl_from_text(responses_csv: str) -> pd.DataFrame:
    """
    Compute FKGL per model from a CSV with columns: model, response_text.

    Requires `textstat`. If not available, instruct user to provide --fkgl.
    Returns DataFrame with columns: model, fkgl.
    """
    if not os.path.exists(responses_csv):
        _fail(f"Responses file not found: {responses_csv}")

    try:
        from textstat import textstat
    except Exception:
        _fail(
            "textstat is not installed. Install with `pip install textstat`, "
            "or provide a precomputed CSV via --fkgl (columns: model,fkgl)."
        )

    df = pd.read_csv(responses_csv)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"model", "response_text"}.issubset(set(df.columns)):
        _fail("responses CSV must contain columns: model, response_text")

    # Compute FKGL per row, then average by model
    def safe_fkgl(txt: str) -> float:
        try:
            return float(textstat.flesch_kincaid_grade(str(txt)))
        except Exception:
            return np.nan

    df["fkgl"] = df["response_text"].astype(str).map(safe_fkgl)
    out = df.groupby("model", as_index=False)["fkgl"].mean().dropna()
    return out


def read_fkgl_csv(fkgl_csv: str) -> pd.DataFrame:
    if not os.path.exists(fkgl_csv):
        _fail(f"FKGL CSV not found: {fkgl_csv}")
    df = pd.read_csv(fkgl_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"model", "fkgl"}.issubset(set(df.columns)):
        _fail("FKGL CSV must contain columns: model,fkgl")
    return df[["model", "fkgl"]].dropna()


def plot_fkgl_bar(fkgl_df: pd.DataFrame, outpath: str, band: Tuple[float, float] = (6.0, 8.0)) -> None:
    if fkgl_df.empty:
        _fail("No FKGL data to plot.")
    models = fkgl_df["model"].tolist()
    values = fkgl_df["fkgl"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, values)

    # Recommended band
    ax.axhspan(band[0], band[1], color="tab:blue", alpha=0.12, label="Recommended (Grade 6–8)")
    ax.axhline(band[0], color="gray", linestyle="--", linewidth=1)
    ax.axhline(band[1], color="gray", linestyle="--", linewidth=1)

    # Annotate bars
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Flesch–Kincaid Grade Level")
    ax.set_title("Readability of Model Outputs (FKGL)")
    ax.legend(loc="upper right", frameon=True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate audit figures (rubric heatmap + FKGL).")
    p.add_argument("--scores", required=True, help="CSV for rubric scores (wide or long format).")
    p.add_argument("--responses", default=None, help="CSV with columns: model,response_text (to compute FKGL).")
    p.add_argument("--fkgl", default=None, help="Precomputed FKGL CSV with columns: model,fkgl (skips textstat).")
    p.add_argument("--outdir", default="results/figures", help="Output directory for figures.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    # 1) Rubric heatmap
    scores_wide = read_scores(args.scores)
    heatmap_path = os.path.join(args.outdir, "rubric_heatmap.png")
    plot_rubric_heatmap(scores_wide, heatmap_path)
    print(f"[OK] Saved heatmap: {heatmap_path}")

    # 2) FKGL bars
    if args.fkgl:
        fkgl_df = read_fkgl_csv(args.fkgl)
    elif args.responses:
        fkgl_df = compute_fkgl_from_text(args.responses)
    else:
        _fail("To plot FKGL, provide either --fkgl (precomputed) or --responses (to compute from text).")

    fkgl_path = os.path.join(args.outdir, "fkgl_band.png")
    plot_fkgl_bar(fkgl_df, fkgl_path, band=(6.0, 8.0))
    print(f"[OK] Saved FKGL plot: {fkgl_path}")


if __name__ == "__main__":
    main()
