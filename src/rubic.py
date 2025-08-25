import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------
# 1) INPUT YOUR DATA HERE
# --------------------
models = ["LLaMA 3.2 (latest)", "Mistral 7B", "DeepSeek-R1 8B"]
axes = ["Accuracy", "Cultural relevance", "Language accessibility", "Bias avoidance"]

# Example placeholder means (0–3). Replace with your computed means.
scores = np.array([
    [2.6, 2.1, 2.4, 2.0],  # LLaMA 3.2
    [2.8, 2.3, 2.6, 2.2],  # Mistral 7B
    [2.4, 2.0, 2.2, 1.9],  # DeepSeek-R1 8B
], dtype=float)

# --------------------
# 2) HEATMAP PLOT
# --------------------
fig, ax = plt.subplots(figsize=(7.5, 3.8))  # one chart, journal friendly

im = ax.imshow(scores, aspect="auto", vmin=0, vmax=3)

# Ticks and labels
ax.set_xticks(np.arange(len(axes)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(axes, rotation=30, ha="right")
ax.set_yticklabels(models)

# Gridlines for matrix look
ax.set_xticks(np.arange(-.5, len(axes), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotate cells with values
for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        ax.text(j, i, f"{scores[i, j]:.1f}", ha="center", va="center", fontsize=9)

# Colorbar with 0–3 scale
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Rubric score (0–3)")

ax.set_title("Equity rubric scores by model (means over double independent ratings)")
plt.tight_layout()

# --------------------
# 3) SAVE FOR LATEX
# --------------------
plt.savefig("rubric_heatmap.png", dpi=400, bbox_inches="tight")
plt.show()
