import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Define color palette
colors = ["#88CCAA", "#FFCC66", "#99CCFF"]

# ------------------------------------------------------------------
# 1. Frequency axis and irregular spectrum
# ------------------------------------------------------------------
frequencies = np.linspace(0, 20, 3000) 

def irregular_spectrum(f):
    np.random.seed(42)
    peak_centers = np.linspace(1.5, 21, 12)
    amps = np.linspace(0.9, 0.3, len(peak_centers))
    widths = np.random.uniform(2.5, 4, len(peak_centers))
    spectrum = np.zeros_like(f)
    for mu, amp, w in zip(peak_centers, amps, widths):
        spectrum += amp * np.exp(-((f - mu) ** 2) / (2 * (w ** 2)))
    noise = sum((0.05 / i) * np.sin(i * f + np.random.uniform(0, 2 * np.pi)) for i in range(3, 15))
    spectrum += noise
    return spectrum

s_full = irregular_spectrum(frequencies)

# ------------------------------------------------------------------
# 2. Hierarchy cut-offs
# ------------------------------------------------------------------
cutoffs = [4, 8, 16]
labels = ["Hierarchy 1 coverage", "Hierarchy 2 incremental", "Hierarchy 3 incremental"]

# ------------------------------------------------------------------
# 3. Plot using constrained_layout
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

# Spectrum curve
ax.plot(frequencies, s_full, color="#4B0082", linewidth=3, label="Full spectrum curve")

# Cutoff bands and labels
prev_cut = 0
for idx, cut in enumerate(cutoffs):
    mask = (frequencies >= prev_cut) & (frequencies <= cut)
    ax.fill_between(frequencies, 0, s_full, where=mask, step="mid", color=colors[idx], alpha=0.5)
    ax.axvline(x=cut, color="gray", linestyle="--", linewidth=2)
    ax.text(cut + 0.3, 0.35 * s_full.max(), rf"Hierarchy ${idx+1}$ cutoff", rotation=90,
            va="center", color="gray", fontsize=18)  # Explicit large font here
    prev_cut = cut

# Un-modeled area after h3
mask_unmodelled = frequencies >= cutoffs[-1]
ax.fill_between(frequencies, 0, s_full, where=mask_unmodelled, step="mid",
                color="lightgray", alpha=0.4, hatch="///", edgecolor="gray",
                label="Un-modeled region")

# Decorations with explicit fonts
ax.set_xlabel("Spatial Frequency of Hypothesis Space Performance Landscape", fontsize=18)
ax.set_ylabel("Spectral Amplitude of Landscape", fontsize=18)
# ax.set_title("Conceptual Spectrum: Hierarchies as Successive Low-pass Filters", fontsize=20, weight='bold')

# Ensure ticks are also larger
ax.tick_params(axis='both', labelsize=16)

# Legend with larger fonts and markers
area_patches = [Patch(facecolor=colors[i], edgecolor="none", alpha=0.5, label=labels[i]) for i in range(3)]
line_full = Line2D([0], [0], color="#4B0082", linewidth=3, label="Full spectrum curve")
patch_unmodelled = Patch(facecolor="lightgray", edgecolor="gray", alpha=0.4, hatch="///", label="Un-modeled region")

ax.legend(handles=area_patches + [patch_unmodelled, line_full], loc="upper right",
          fontsize=13.5, frameon=False, handlelength=2, markerscale=1.5)

# Grid
ax.grid(alpha=0.4, linestyle=':', linewidth=1.2)

# Save BEFORE show
fig.savefig("spectrum_hierarchy_cutoffs.pdf", dpi=600, bbox_inches="tight")
plt.show()
