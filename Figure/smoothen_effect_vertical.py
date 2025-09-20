import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. Set global font settings (larger, academic style)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "serif"  # Optional: Use serif for academic look
})

# ----------------------------------------------------------------------
# 1.  Landscape definitions
# ----------------------------------------------------------------------
def level_1(x):
    return np.exp(-((x - 2) ** 2) / 2) + np.exp(-((x + 2) ** 2) / 2)

def level_2(x):
    return level_1(x) + 0.3 * np.sin(3 * x)

def level_3(x):
    return level_2(x) + 0.15 * np.sin(10 * x)

# ----------------------------------------------------------------------
# 2.  Characteristic points
# ----------------------------------------------------------------------
x = np.linspace(-6, 6, 12_000)
region = (x > 1.5) & (x < 3.5)

x_init = 0.0
y_init = level_1(x_init)

x1_opt = x[region][np.argmax(level_1(x[region]))]
x2_opt = x[region][np.argmax(level_2(x[region]))]
x3_opt = x[region][np.argmax(level_3(x[region]))]

# Values on Level‑3 landscape (ensuring correct ordering)
y1_on_L3 = level_3(x1_opt)
y2_on_L3 = level_3(x2_opt)
y3_on_L3 = level_3(x3_opt)

assert y3_on_L3 > y2_on_L3 > y1_on_L3, "Hierarchy ordering violated"

# Sub‑optimal local optimum (no‑hierarchy)
dy3 = np.gradient(level_3(x), x)
candidate_idx = [i for i in range(1, len(dy3)) if dy3[i - 1] > 0 and dy3[i] <= 0]
idx_flat = min(candidate_idx, key=lambda i: abs(x[i]))
x_flat, y_flat = x[idx_flat], level_3(x[idx_flat])


# ----------------------------------------------------------------------
# 3. Plot
# ----------------------------------------------------------------------
pair1_colour = "green"
pair2_colour = "orange"
opt3_colour  = "purple"
init_colour  = "red"

dot_size = 70  # Larger dots
x_marker_size = dot_size * 1.5  # Bigger for 'x' markers

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(x, level_1(x), label="Level 1 landscape")
axes[0].scatter(x_init, level_1(x_init), marker="D", color=init_colour, s=dot_size, zorder=6, label="Initial point")
axes[0].scatter(x1_opt, level_1(x1_opt), marker="o", color=pair1_colour, s=dot_size, zorder=6, label="Level 1 optimum")
axes[0].set_title("Level 1: High-level abstract landscape")
axes[0].set_ylabel("Performance")
axes[0].legend(loc="upper left")

axes[1].plot(x, level_2(x), color="grey", label="Level 2 landscape")
axes[1].scatter(x_init, level_2(x_init), marker="D", color=init_colour, s=dot_size, zorder=6, label="Initial point")
axes[1].scatter(x1_opt, level_2(x1_opt), marker="o", color=pair1_colour, s=dot_size, zorder=6, label="Level 1 optimum (projected)")
axes[1].scatter(x1_opt, level_2(x1_opt), marker="x", s=x_marker_size, color=pair1_colour, zorder=6, label="Start (L1 optimum)")
axes[1].scatter(x2_opt, level_2(x2_opt), marker="o", color=pair2_colour, s=dot_size, zorder=7, label="Level 2 optimum")
axes[1].set_title("Level 2: Intermediate landscape")
axes[1].set_ylabel("Performance")
axes[1].legend(loc="upper left")

axes[2].plot(x, level_3(x), color="slateblue", label="Level 3 landscape")
axes[2].scatter(x_init, level_3(x_init), marker="D", color=init_colour, s=dot_size, zorder=6, label="Initial point")
axes[2].scatter(x1_opt, level_3(x1_opt), marker="o", color=pair1_colour, s=dot_size, zorder=6, label="Level 1 optimum (projected)")
axes[2].scatter(x2_opt, level_3(x2_opt), marker="o", color=pair2_colour, s=dot_size, zorder=7, label="Level 2 optimum (projected)")
axes[2].scatter(x2_opt, level_3(x2_opt), marker="x", s=x_marker_size, color=pair2_colour, zorder=7, label="Start (L2 optimum)")
axes[2].scatter(x3_opt, level_3(x3_opt), marker="o", color=opt3_colour, s=dot_size, zorder=8, label="Level 3 optimum")
axes[2].scatter(x_flat, y_flat, marker="P", color="black", s=dot_size*1.2, zorder=8, label="No-hierarchy local optimum")
axes[2].set_title("Level 3: Detailed rugged landscape")
axes[2].set_xlabel("Hypothesis space")
axes[2].set_ylabel("Performance")
axes[2].legend(loc="upper left")

plt.tight_layout()
plt.show()
fig.savefig("hhs_smoothing_vertical.pdf", dpi=600, bbox_inches="tight")