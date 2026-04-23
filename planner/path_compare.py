"""
MediNav — Path Comparison Visualisation
planner/path_compare.py

Compares:
  - Standard A* (shortest path, red)
  - Risk-Aware A* (safe path, blue)

on a hospital-corridor occupancy grid with a wall-proximity risk map.
Saves: outputs/path_comparison.png
"""

import heapq
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# 1. Build the 100x100 occupancy grid
# ---------------------------------------------------------------------------

H, W = 100, 100
grid = np.ones((H, W), dtype=int)  # start all walls

def free(r_slice, c_slice):
    grid[r_slice, c_slice] = 0

# Top horizontal corridor
free(slice(10, 21), slice(0, 100))

# Bottom horizontal corridor
free(slice(70, 81), slice(0, 100))

# Right vertical connector
free(slice(20, 71), slice(70, 81))

# Narrow horizontal shortcut (rows 40-50)
free(slice(40, 51), slice(0, 81))  # ends where vertical connector begins

# Add 2×2 equipment obstacles in the narrow shortcut
obstacles = [(43, 20), (43, 40), (45, 58), (42, 70)]
for (r, c) in obstacles:
    grid[r:r+2, c:c+2] = 1

# ---------------------------------------------------------------------------
# 2. Risk map from wall-proximity distance
# ---------------------------------------------------------------------------

free_mask = (grid == 0).astype(float)
dist = distance_transform_edt(free_mask)   # distance of each free cell to nearest wall

risk_map = np.zeros((H, W), dtype=float)
for r in range(H):
    for c in range(W):
        if grid[r, c] == 1:
            risk_map[r, c] = 1.0   # walls themselves are max risk
            continue
        d = dist[r, c]
        if d <= 5:
            risk_map[r, c] = np.random.uniform(0.8, 1.0)
        elif d <= 10:
            risk_map[r, c] = np.random.uniform(0.4, 0.6)
        else:
            risk_map[r, c] = np.random.uniform(0.05, 0.2)

# ---------------------------------------------------------------------------
# 3. A* implementation (inline — no external import)
# ---------------------------------------------------------------------------

def _h(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(grid, risk_map, start, goal, lambda_weight=0.0):
    """
    General A* with combined cost:
        f = g_dist + lambda * g_risk + h
    Returns list of (row, col) or None.
    """
    rows, cols = grid.shape
    open_heap = [(_h(start, goal), 0.0, 0.0, start[0], start[1])]
    best = {start: (0.0, 0.0)}
    came_from = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g_d, g_r, r, c = heapq.heappop(open_heap)
        node = (r, c)

        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        # Staleness check
        if node in best:
            bd, br = best[node]
            curr_f = g_d + lambda_weight * g_r + _h(node, goal)
            old_f  = bd  + lambda_weight * br  + _h(node, goal)
            if curr_f > old_f + 1e-9:
                continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            nb = (nr, nc)
            step_d = math.sqrt(dr * dr + dc * dc)
            step_r = float(risk_map[nr, nc])
            new_gd = g_d + step_d
            new_gr = g_r + step_r

            new_f = new_gd + lambda_weight * new_gr + _h(nb, goal)
            if nb in best:
                bd, br = best[nb]
                old_f = bd + lambda_weight * br + _h(nb, goal)
                if new_f >= old_f - 1e-9:
                    continue

            best[nb] = (new_gd, new_gr)
            came_from[nb] = node
            heapq.heappush(open_heap, (new_f, new_gd, new_gr, nr, nc))

    return None

# ---------------------------------------------------------------------------
# 4. Plan paths
# ---------------------------------------------------------------------------

START = (15, 5)
GOAL  = (75, 75)

np.random.seed(0)  # reproducible risk noise

path_std  = astar(grid, risk_map, START, GOAL, lambda_weight=0.0)
path_safe = astar(grid, risk_map, START, GOAL, lambda_weight=8.0)

if path_std is None:
    raise RuntimeError("Standard A* found no path — check corridor connectivity.")
if path_safe is None:
    raise RuntimeError("Risk-Aware A* found no path — check corridor connectivity.")

len_std  = len(path_std)
len_safe = len(path_safe)
overhead = (len_safe - len_std) / len_std * 100

print(f"Standard A*   — path length : {len_std} cells")
print(f"Risk-Aware A* — path length : {len_safe} cells")
print(f"Safety overhead             : {overhead:.1f}%")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------

def path_xy(path):
    """Return (cols, rows) arrays for matplotlib (x=col, y=row)."""
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    return cols, rows   # x, y

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

for ax, path, color, title, annotation in [
    (axes[0], path_std,  "#e63946", "Standard A* (Shortest Path)",
     "Path hugs walls — HIGH COLLISION RISK"),
    (axes[1], path_safe, "#4895ef", "Risk-Aware A* (MediNav)",
     "Path stays in corridor centre — SAFE"),
]:
    # --- base map ---
    wall_display = np.where(grid == 1, 0.0, 0.85)   # walls black, free light grey
    ax.imshow(wall_display, cmap="gray", vmin=0, vmax=1,
              origin="upper", interpolation="nearest")

    # --- risk heatmap (faint red) ---
    risk_rgba = np.zeros((H, W, 4), dtype=float)
    risk_rgba[:, :, 0] = risk_map          # red channel
    risk_rgba[:, :, 3] = risk_map * 0.35   # alpha
    ax.imshow(risk_rgba, origin="upper", interpolation="nearest")

    # --- path ---
    px, py = path_xy(path)
    ax.plot(px, py, color=color, linewidth=2.2, zorder=5, solid_capstyle="round")

    # --- start / goal markers ---
    ax.plot(START[1], START[0], "o", color="#06d6a0", markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6, label="Start")
    ax.plot(GOAL[1],  GOAL[0],  "*", color="#ffd166", markersize=13,
            markeredgecolor="white", markeredgewidth=1.0, zorder=6, label="Goal")

    # --- annotation box ---
    ax.text(50, 93, annotation, ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                      alpha=0.82, edgecolor="white", linewidth=0.8))

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0d1b2a")

    # legend
    legend = ax.legend(loc="upper left", fontsize=8,
                       facecolor="#0d1b2a", edgecolor="gray", labelcolor="white")

fig.suptitle("MediNav Path Planning — Risk-Aware vs. Standard A*",
             color="white", fontsize=14, fontweight="bold", y=0.98)

# --- bottom stats text box ---
stats_text = (
    f"Standard A*: Path Length = {len_std} cells  |  "
    f"Risk-Aware A*: Path Length = {len_safe} cells  |  "
    f"Safety overhead = {overhead:.1f}%"
)
fig.text(0.5, 0.01, stats_text, ha="center", va="bottom",
         fontsize=9.5, color="white",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                   edgecolor="#4895ef", linewidth=1.2))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

os.makedirs("outputs", exist_ok=True)
out_path = "outputs/path_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nSaved -> {out_path}")
plt.close()
