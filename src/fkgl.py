import numpy as np
import matplotlib.pyplot as plt

# Example FKGL values (replace with actual results)
models = ["DeepSeek-R1 8B", "LLaMA 3.2 (latest)", "Mistral 7B"]
fkgl_means = np.array([11.8, 11.2, 12.5], dtype=float)

# Plot
fig, ax = plt.subplots(figsize=(7.0, 4.0))
x = np.arange(len(models))

# Bars
bars = ax.bar(x, fkgl_means, label="Model FKGL")

# Recommended readability band (grades 6–8)
band = ax.axhspan(6, 8, color="lightgrey", alpha=0.4, label="Recommended range (Grade 6–8)")
ax.axhline(6, linestyle="--", color="grey", linewidth=1)
ax.axhline(8, linestyle="--", color="grey", linewidth=1)

# Annotate bar values
for xi, yi in zip(x, fkgl_means):
    ax.text(xi, yi + 0.2, f"{yi:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right")
ax.set_ylabel("Flesch–Kincaid Grade Level")
ax.set_title("Readability of Model Outputs (FKGL)")

ax.set_ylim(0, max(14, fkgl_means.max() + 2))

# Legend
ax.legend(loc="lower right", frameon=True)
# Legend outside the plot (right-hand side)


#
#plt.tight_layout()

# Save
out_path = "fkgl2.png"
plt.savefig(out_path, dpi=400, bbox_inches="tight")
out_path
