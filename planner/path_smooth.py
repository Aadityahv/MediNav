import heapq
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import CubicSpline

os.makedirs("outputs", exist_ok=True)


def _h(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(grid, risk_map, start, goal, lambda_weight=0.0):
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

        if node in best:
            bd, br = best[node]
            if g_d > bd + 1e-9 or g_r > br + 1e-9:
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


def build_risk_map(grid, alpha=1.5):
    free_mask = (grid == 0).astype(float)
    dist = distance_transform_edt(free_mask)
    risk = np.exp(-alpha * dist)
    return np.clip(risk, 0, 1)


def build_standard_grid():
    g = np.ones((100, 100), dtype=int)
    g[10:21, 0:100] = 0
    g[70:81, 0:100] = 0
    g[20:71, 70:81] = 0
    g[40:51, 0:81]  = 0
    g[0, :]  = 1;  g[99, :] = 1
    g[:, 0]  = 1;  g[:, 99] = 1
    g[43:45, 20:22] = 1
    g[43:45, 40:42] = 1
    g[45:47, 58:60] = 1
    g[42:44, 70:72] = 1
    return g

np.random.seed(0)
grid_std = build_standard_grid()
risk_std  = build_risk_map(grid_std, alpha=1.5)

START = (15, 5)
GOAL  = (75, 75)

raw_path = astar(grid_std, risk_std, START, GOAL, lambda_weight=8.0)
if raw_path is None:
    raise RuntimeError("A* found no path on standard grid.")

raw_rows = np.array([p[0] for p in raw_path], dtype=float)
raw_cols = np.array([p[1] for p in raw_path], dtype=float)
t = np.linspace(0, 1, len(raw_path))

cs_r = CubicSpline(t, raw_rows)
cs_c = CubicSpline(t, raw_cols)

t_fine = np.linspace(0, 1, 300)
smooth_rows = cs_r(t_fine)
smooth_cols = cs_c(t_fine)

smooth_rows = np.clip(smooth_rows, 0, 99)
smooth_cols = np.clip(smooth_cols, 0, 99)

valid = [
    (int(round(r)), int(round(c)))
    for r, c in zip(smooth_rows, smooth_cols)
    if grid_std[int(round(r)), int(round(c))] == 0
]
smooth_path_r = [p[0] for p in valid]
smooth_path_c = [p[1] for p in valid]

print(f"[Part 1] Raw path    : {len(raw_path)} cells")
print(f"[Part 1] Smoothed    : {len(valid)} points sampled")

H, W = 100, 100
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 7))
fig1.patch.set_facecolor("#1a1a2e")

wall_disp = np.where(grid_std == 1, 0.0, 0.2)

risk_rgba = np.zeros((H, W, 4), dtype=float)
risk_rgba[:, :, 0] = risk_std
risk_rgba[:, :, 3] = risk_std * 0.3

for ax, title, path_r, path_c, color, lw, annotation in [
    (axes1[0], "Raw A* Path",
     raw_rows, raw_cols, "#e63946", 1.5, None),
    (axes1[1], "Smoothed Path (Cubic Spline)",
     smooth_path_r, smooth_path_c, "#00f5d4", 2.0,
     "Physically realisable trajectory"),
]:
    ax.imshow(wall_disp, cmap="gray", vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.imshow(risk_rgba, origin="upper", interpolation="nearest")
    ax.plot(path_c, path_r, color=color, linewidth=lw, zorder=5,
            solid_capstyle="round")
    ax.plot(START[1], START[0], "o", color="#06d6a0", markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6, label="Start")
    ax.plot(GOAL[1],  GOAL[0],  "*", color="#ffd166", markersize=13,
            markeredgecolor="white", markeredgewidth=1.0, zorder=6, label="Goal")
    if annotation:
        ax.text(50, 93, annotation, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#4895ef",
                          alpha=0.85, edgecolor="white", linewidth=0.8))
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0d1b2a")
    ax.legend(loc="upper left", fontsize=8,
              facecolor="#0d1b2a", edgecolor="gray", labelcolor="white")

fig1.suptitle("MediNav Path Smoothing — Cubic Spline Post-Processing",
              color="white", fontsize=13, fontweight="bold", y=0.98)
fig1.text(0.5, 0.01,
          f"Raw path: {len(raw_path)} cells  |  Smoothed path: {len(valid)} points sampled",
          ha="center", va="bottom", fontsize=9.5, color="white",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                    edgecolor="#4895ef", linewidth=1.2))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig1.savefig("outputs/path_smoothed.png", dpi=150, bbox_inches="tight",
             facecolor=fig1.get_facecolor())
plt.close(fig1)
print("Saved → outputs/path_smoothed.png")


def add_border(g):
    g[0, :] = 1;  g[99, :] = 1
    g[:, 0] = 1;  g[:, 99] = 1
    return g

def layout_standard():
    g = np.ones((100, 100), dtype=int)
    g[10:21, 0:100] = 0
    g[70:81, 0:100] = 0
    g[20:71, 70:81] = 0
    g[40:51, 0:81]  = 0
    g[43:45, 20:22] = 1
    g[43:45, 40:42] = 1
    g[45:47, 58:60] = 1
    g[42:44, 70:72] = 1
    return add_border(g)

def layout_tjunction():
    g = np.ones((100, 100), dtype=int)
    g[45:56, 0:100] = 0
    g[0:46,  45:56] = 0
    g[45:100,45:56] = 0
    g[10:21, 0:46]  = 0
    g[75:86, 0:46]  = 0
    g[10:21, 55:100]= 0
    g[75:86, 55:100]= 0
    return add_border(g)

def layout_lshape():
    g = np.ones((100, 100), dtype=int)
    g[5:96,  5:20]  = 0
    g[80:96, 5:96]  = 0
    g[5:21,  5:96]  = 0
    return add_border(g)

layouts = [
    ("Standard Ward",   layout_standard(),  (15, 5),  (75, 75)),
    ("T-Junction Ward", layout_tjunction(), (12, 5),  (78, 5)),
    ("L-Shape Ward",    layout_lshape(),    (10, 10), (85, 85)),
]

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor("#1a1a2e")

for ax, (name, g, s, goal) in zip(axes2, layouts):
    risk = build_risk_map(g, alpha=1.5)

    p_std  = astar(g, risk, s, goal, lambda_weight=0.0)
    p_safe = astar(g, risk, s, goal, lambda_weight=8.0)

    if p_std is None:
        print(f"WARNING: Standard A* found no path for '{name}'")
    if p_safe is None:
        print(f"WARNING: Risk-Aware A* found no path for '{name}'")

    def violations(path, risk_map):
        if path is None:
            return 0
        return sum(1 for r, c in path if risk_map[r, c] > 0.05)

    viol_std  = violations(p_std,  risk)
    viol_safe = violations(p_safe, risk)
    viol_reduction = (
        (viol_std - viol_safe) / viol_std * 100
        if viol_std > 0 else 0.0
    )
    len_std  = len(p_std)  if p_std  else 0
    len_safe = len(p_safe) if p_safe else 0

    print(f"[{name}] Std: {len_std} cells, {viol_std} violations | "
          f"Safe: {len_safe} cells, {viol_safe} violations | "
          f"Violations reduced: {viol_reduction:.1f}%")

    wall_disp = np.where(g == 1, 0.0, 0.2)
    ax.imshow(wall_disp, cmap="gray", vmin=0, vmax=1,
              origin="upper", interpolation="nearest")

    r_rgba = np.zeros((100, 100, 4), dtype=float)
    r_rgba[:, :, 0] = risk
    r_rgba[:, :, 3] = risk * 0.25
    ax.imshow(r_rgba, origin="upper", interpolation="nearest")

    if p_std:
        ax.plot([c for _, c in p_std], [r for r, _ in p_std],
                color="#e63946", linewidth=1.8, linestyle="--",
                zorder=5, label=f"Std A* ({len_std})")
    if p_safe:
        ax.plot([c for _, c in p_safe], [r for r, _ in p_safe],
                color="#4895ef", linewidth=2.0, linestyle="-",
                zorder=6, label=f"Risk-Aware ({len_safe})")

    ax.plot(s[1],    s[0],    "o", color="#06d6a0", markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=7)
    ax.plot(goal[1], goal[0], "*", color="#ffd166", markersize=13,
            markeredgecolor="white", markeredgewidth=1.0, zorder=7)

    stats = (f"Std: {len_std} cells  |  Safe: {len_safe} cells\n"
             f"Violations reduced {viol_reduction:.0f}%")
    ax.text(50, 94, stats, ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e",
                      alpha=0.88, edgecolor="#4895ef", linewidth=0.8))

    ax.set_title(f"{name}", color="white", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(-0.5, 99.5); ax.set_ylim(99.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0d1b2a")
    ax.legend(loc="upper left", fontsize=7,
              facecolor="#0d1b2a", edgecolor="gray", labelcolor="white")

fig2.suptitle(
    "MediNav Generalisation — Risk-Aware A* Across 3 Hospital Layouts",
    color="white", fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
fig2.savefig("outputs/multi_layout.png", dpi=150, bbox_inches="tight",
             facecolor=fig2.get_facecolor())
plt.close(fig2)
print("Saved → outputs/multi_layout.png")
