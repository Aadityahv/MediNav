"""
Parameter Sensitivity Analysis — α, w1, w2
MediNav Project — planner/param_sensitivity.py

Validates the risk map parameters by sweeping across values and measuring
path length and safety violations. Produces a dual-subplot figure saved
to outputs/param_sensitivity.png.
"""

import heapq
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# 1. Hospital grid (100x100)
# ---------------------------------------------------------------------------
def build_grid():
    grid = np.ones((100, 100), dtype=int)
    grid[10:21, 0:100] = 0   # top corridor
    grid[70:81, 0:100] = 0   # bottom corridor
    grid[20:71, 70:81] = 0   # right vertical connector
    grid[40:51, 0:81]  = 0   # narrow shortcut

    # Border walls
    grid[0, :]  = 1
    grid[99, :] = 1
    grid[:, 0]  = 1
    grid[:, 99] = 1

    # Obstacles inside the narrow shortcut
    grid[43:45, 20:22] = 1
    grid[43:45, 40:42] = 1
    grid[45:47, 58:60] = 1
    grid[42:44, 70:72] = 1

    return grid


# ---------------------------------------------------------------------------
# 2. Risk map builder
# ---------------------------------------------------------------------------
def build_risk_map(grid, alpha, w1, w2, uncertainty=0.3):
    free_mask = (grid == 0).astype(float)
    dist = distance_transform_edt(free_mask)
    risk_map = np.exp(-alpha * dist)
    risk_map = np.clip(risk_map, 0, 1)

    uniform_uncertainty = np.full_like(risk_map, uncertainty)
    risk_map = w1 * risk_map + w2 * uniform_uncertainty

    return risk_map, dist


# ---------------------------------------------------------------------------
# 3. Risk-aware A* (inline, 8-connected)
# ---------------------------------------------------------------------------
def astar_risk(grid, risk_map, start, goal, lambda_weight=8.0):
    rows, cols = grid.shape

    def heuristic(node):
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    start_f = heuristic(start)
    open_heap = [(start_f, 0.0, start[0], start[1])]

    g_cost = {start: 0.0}
    came_from = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        node = (r, c)

        if g > g_cost.get(node, float('inf')) + 1e-9:
            continue

        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            step_dist = math.sqrt(dr * dr + dc * dc)
            step_cost = step_dist + lambda_weight * risk_map[nr, nc]

            new_g = g + step_cost
            neighbour = (nr, nc)

            if new_g < g_cost.get(neighbour, float('inf')) - 1e-9:
                g_cost[neighbour] = new_g
                came_from[neighbour] = node
                new_f = new_g + heuristic(neighbour)
                heapq.heappush(open_heap, (new_f, new_g, nr, nc))

    return None


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(path, dist_field):
    """Return (path_length_euclidean, safety_violations)."""
    if path is None:
        return float('inf'), float('inf')

    length = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        length += math.sqrt(dr * dr + dc * dc)

    violations = sum(1 for (r, c) in path if dist_field[r, c] < 3)
    return length, violations


# ---------------------------------------------------------------------------
# 5. Test 1 — Vary alpha
# ---------------------------------------------------------------------------
def test_alpha(grid, start, goal):
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    w1_fixed, w2_fixed = 0.7, 0.3
    results = []

    for alpha in alphas:
        risk_map, dist_field = build_risk_map(grid, alpha, w1_fixed, w2_fixed)
        path = astar_risk(grid, risk_map, start, goal, lambda_weight=8.0)
        length, violations = compute_metrics(path, dist_field)
        results.append((alpha, length, violations))

    return results


# ---------------------------------------------------------------------------
# 6. Test 2 — Vary w1/w2
# ---------------------------------------------------------------------------
def test_weights(grid, start, goal):
    weight_pairs = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3),
                    (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]
    alpha_fixed = 1.5
    results = []

    for w1, w2 in weight_pairs:
        risk_map, dist_field = build_risk_map(grid, alpha_fixed, w1, w2)
        path = astar_risk(grid, risk_map, start, goal, lambda_weight=8.0)
        length, violations = compute_metrics(path, dist_field)
        results.append((w1, w2, length, violations))

    return results


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------
def plot_results(alpha_results, weight_results, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "MediNav Parameter Sensitivity Analysis\n"
        "Justification of α=1.5, w1=0.7, w2=0.3",
        fontsize=13, fontweight='bold'
    )

    # ---- Left subplot: Effect of α ----
    alphas = [r[0] for r in alpha_results]
    lengths_a = [r[1] for r in alpha_results]
    violations_a = [r[2] for r in alpha_results]

    color_l = '#1f77b4'
    color_v = '#d62728'

    ax1.set_xlabel('α (decay constant)', fontsize=10)
    ax1.set_ylabel('Path Length', color=color_l, fontsize=10)
    ax1.plot(alphas, lengths_a, 'o-', color=color_l, linewidth=2,
             markersize=7, label='Path Length')
    ax1.tick_params(axis='y', labelcolor=color_l)

    ax1r = ax1.twinx()
    ax1r.set_ylabel('Safety Violations', color=color_v, fontsize=10)
    ax1r.plot(alphas, violations_a, 's-', color=color_v, linewidth=2,
              markersize=7, label='Safety Violations')
    ax1r.tick_params(axis='y', labelcolor=color_v)

    ax1.axvline(x=1.5, color='gray', linestyle='--', linewidth=1.2)
    # Place annotation at top of axes using transform
    ax1.text(1.5, 0.95, 'chosen α', transform=ax1.get_xaxis_transform(),
             fontsize=9, ha='center', va='top', color='gray',
             fontstyle='italic')

    ax1.set_title("Effect of α on Path Safety vs Length", fontsize=11, pad=10)
    ax1.grid(True, alpha=0.3)

    # Pad the y-axis limits slightly
    len_margin_a = (max(lengths_a) - min(lengths_a)) * 0.15 or 1.0
    ax1.set_ylim(min(lengths_a) - len_margin_a, max(lengths_a) + len_margin_a)
    viol_margin_a = (max(violations_a) - min(violations_a)) * 0.15 or 1.0
    ax1r.set_ylim(min(violations_a) - viol_margin_a, max(violations_a) + viol_margin_a)

    # ---- Right subplot: Effect of w1/w2 ----
    w1_vals = [r[0] for r in weight_results]
    lengths_w = [r[2] for r in weight_results]
    violations_w = [r[3] for r in weight_results]

    ax2.set_xlabel('w1 (risk weight)', fontsize=10)
    ax2.set_ylabel('Path Length', color=color_l, fontsize=10)
    ax2.plot(w1_vals, lengths_w, 'o-', color=color_l, linewidth=2,
             markersize=7, label='Path Length')
    ax2.tick_params(axis='y', labelcolor=color_l)

    ax2r = ax2.twinx()
    ax2r.set_ylabel('Safety Violations', color=color_v, fontsize=10)
    ax2r.plot(w1_vals, violations_w, 's-', color=color_v, linewidth=2,
              markersize=7, label='Safety Violations')
    ax2r.tick_params(axis='y', labelcolor=color_v)

    ax2.axvline(x=0.7, color='gray', linestyle='--', linewidth=1.2)
    ax2.text(0.7, 0.95, 'chosen w1', transform=ax2.get_xaxis_transform(),
             fontsize=9, ha='center', va='top', color='gray',
             fontstyle='italic')

    ax2.set_title("Effect of Risk Weight w1 on Path Safety vs Length",
                   fontsize=11, pad=10)
    ax2.grid(True, alpha=0.3)

    # Pad the y-axis limits slightly
    len_margin_w = (max(lengths_w) - min(lengths_w)) * 0.15 or 1.0
    ax2.set_ylim(min(lengths_w) - len_margin_w, max(lengths_w) + len_margin_w)
    viol_margin_w = (max(violations_w) - min(violations_w)) * 0.15 or 1.0
    ax2r.set_ylim(min(violations_w) - viol_margin_w, max(violations_w) + viol_margin_w)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] Figure saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Terminal summary tables
# ---------------------------------------------------------------------------
def print_tables(alpha_results, weight_results):
    print("\n" + "=" * 55)
    print("TEST 1 — Effect of Decay Constant α")
    print("     (w1=0.7, w2=0.3 fixed)")
    print("=" * 55)
    print(f"{'α':>6}  {'Path Length':>12}  {'Safety Violations':>18}")
    print("-" * 40)
    for alpha, length, violations in alpha_results:
        print(f"{alpha:>6.1f}  {length:>12.1f}  {violations:>18d}")

    print("\n" + "=" * 55)
    print("TEST 2 — Effect of Risk Weights w1 / w2")
    print("     (α=1.5 fixed)")
    print("=" * 55)
    print(f"{'w1':>6}  {'w2':>6}  {'Path Length':>12}  {'Safety Violations':>18}")
    print("-" * 48)
    for w1, w2, length, violations in weight_results:
        print(f"{w1:>6.1f}  {w2:>6.1f}  {length:>12.1f}  {violations:>18d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    grid = build_grid()
    start = (15, 5)
    goal = (75, 75)

    # Sanity checks
    assert grid[start[0], start[1]] == 0, f"Start {start} is a wall!"
    assert grid[goal[0], goal[1]] == 0, f"Goal {goal} is a wall!"

    print("MediNav — Parameter Sensitivity Analysis")
    print(f"Grid: {grid.shape}, Start: {start}, Goal: {goal}")
    print(f"Free cells: {np.sum(grid == 0)}, Wall cells: {np.sum(grid == 1)}")

    # Run tests
    print("\nRunning TEST 1 — varying α ...")
    alpha_results = test_alpha(grid, start, goal)

    print("Running TEST 2 — varying w1/w2 ...")
    weight_results = test_weights(grid, start, goal)

    # Print tables
    print_tables(alpha_results, weight_results)

    # Plot & save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "outputs", "param_sensitivity.png")
    plot_results(alpha_results, weight_results, save_path)

    print("\nDone.")
