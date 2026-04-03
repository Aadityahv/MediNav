"""
Extended Evaluation Script — Dijkstra vs Standard A* vs Risk-Aware A*
Runs 30 Monte Carlo trials on the synthetic hospital grid and prints a
three-planner comparison statistics table.  Saves a grouped bar chart to
outputs/evaluation_extended.png and a log to outputs/eval_log_extended.txt.
"""

import sys
import io
import os

# Force UTF-8 output on Windows to support box-drawing characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Hospital grid generator (inlined for self-containment — matches evaluate.py)
# ---------------------------------------------------------------------------

def _generate_hospital_grid(size=100):
    grid = np.ones((size, size), dtype=np.int32)

    # Top horizontal corridor: rows 15-30
    grid[15:31, 1:size - 1] = 0
    # Bottom horizontal corridor: rows 65-80
    grid[65:81, 1:size - 1] = 0
    # Right vertical corridor: cols 70-85, rows 30-65
    grid[30:66, 70:86] = 0
    # Central service corridor: rows 45-55, cols 5-70
    grid[45:56, 5:71] = 0

    # Outer walls
    grid[0, :] = 1
    grid[size - 1, :] = 1
    grid[:, 0] = 1
    grid[:, size - 1] = 1

    # Risk map
    free_mask = (grid == 0).astype(np.float64)
    dist_from_walls = distance_transform_edt(free_mask)
    risk_map = np.exp(-1.5 * dist_from_walls)
    risk_map = np.clip(risk_map, 0.0, 1.0)

    return grid, risk_map, dist_from_walls


def _get_random_free_cell(grid, dist_map, min_wall_dist=5):
    candidates = np.argwhere(dist_map >= min_wall_dist)
    if len(candidates) == 0:
        raise ValueError("No valid free cell.")
    idx = np.random.randint(len(candidates))
    return tuple(candidates[idx])


# ---------------------------------------------------------------------------
# Shared helpers (matches evaluate.py exactly)
# ---------------------------------------------------------------------------

_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _heuristic(a, b):
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Planners
# ---------------------------------------------------------------------------

def dijkstra(grid, start, goal):
    """Dijkstra — priority = cumulative Euclidean distance only, no heuristic."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        cost, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in closed:
            continue
        closed.add(current)

        for dr, dc in _NEIGHBORS:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and neighbor not in closed:
                move_cost = np.sqrt(dr ** 2 + dc ** 2)
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g, neighbor))
    return None  # no path


def standard_astar(grid, start, goal):
    """Standard A* — movement cost is Euclidean distance only."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in closed:
            continue
        closed.add(current)

        for dr, dc in _NEIGHBORS:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and neighbor not in closed:
                move_cost = np.sqrt(dr ** 2 + dc ** 2)
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + _heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None  # no path


def risk_astar(grid, risk_map, start, goal, lambda_weight=8.0):
    """Risk-aware A* — movement cost incorporates risk penalty."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in closed:
            continue
        closed.add(current)

        for dr, dc in _NEIGHBORS:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and neighbor not in closed:
                move_cost = np.sqrt(dr ** 2 + dc ** 2)
                risk_penalty = lambda_weight * risk_map[nr, nc]
                tentative_g = g_score[current] + move_cost + risk_penalty
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + _heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(n_trials=30, seed=42):
    np.random.seed(seed)
    grid, risk_map, dist_map = _generate_hospital_grid()

    # Per-planner accumulators
    dij_lengths, dij_min_clr, dij_violations = [], [], []
    std_lengths, std_min_clr, std_violations = [], [], []
    ra_lengths,  ra_min_clr,  ra_violations  = [], [], []

    completed = 0
    attempts = 0
    max_attempts = n_trials * 5  # avoid infinite loop

    while completed < n_trials and attempts < max_attempts:
        attempts += 1
        try:
            start = _get_random_free_cell(grid, dist_map, min_wall_dist=5)
            # Goal must be at least 20 cells from start
            for _ in range(200):
                goal = _get_random_free_cell(grid, dist_map, min_wall_dist=5)
                if _heuristic(start, goal) >= 20:
                    break
            else:
                continue  # could not find far-enough goal
        except ValueError:
            continue

        path_dij = dijkstra(grid, start, goal)
        path_std = standard_astar(grid, start, goal)
        path_ra  = risk_astar(grid, risk_map, start, goal, lambda_weight=8.0)

        # If ANY planner fails → skip trial
        if path_dij is None or path_std is None or path_ra is None:
            continue

        # --- Metrics for Dijkstra ---
        clr_dij = [dist_map[r, c] for r, c in path_dij]
        dij_lengths.append(len(path_dij))
        dij_min_clr.append(min(clr_dij))
        dij_violations.append(sum(1 for d in clr_dij if d < 3))

        # --- Metrics for Standard A* ---
        clr_std = [dist_map[r, c] for r, c in path_std]
        std_lengths.append(len(path_std))
        std_min_clr.append(min(clr_std))
        std_violations.append(sum(1 for d in clr_std if d < 3))

        # --- Metrics for Risk-Aware A* ---
        clr_ra = [dist_map[r, c] for r, c in path_ra]
        ra_lengths.append(len(path_ra))
        ra_min_clr.append(min(clr_ra))
        ra_violations.append(sum(1 for d in clr_ra if d < 3))

        completed += 1
        print(f"  Trial {completed}/{n_trials} done  (start={start}, goal={goal})")

    if completed == 0:
        print("ERROR: No trials completed — grid may be too disconnected.")
        return

    # --- Compute summary statistics ---
    mean_len_dij = np.mean(dij_lengths)
    mean_len_std = np.mean(std_lengths)
    mean_len_ra  = np.mean(ra_lengths)

    overhead_dij = 0.0  # Dijkstra is the baseline
    overhead_std = (mean_len_std - mean_len_dij) / mean_len_dij * 100
    overhead_ra  = (mean_len_ra  - mean_len_dij) / mean_len_dij * 100

    mean_clr_dij = np.mean(dij_min_clr)
    mean_clr_std = np.mean(std_min_clr)
    mean_clr_ra  = np.mean(ra_min_clr)

    mean_viol_dij = np.mean(dij_violations)
    mean_viol_std = np.mean(std_violations)
    mean_viol_ra  = np.mean(ra_violations)

    total_cells_dij = sum(dij_lengths)
    total_cells_std = sum(std_lengths)
    total_cells_ra  = sum(ra_lengths)

    viol_rate_dij = sum(dij_violations) / total_cells_dij * 100
    viol_rate_std = sum(std_violations) / total_cells_std * 100
    viol_rate_ra  = sum(ra_violations)  / total_cells_ra  * 100

    # --- Build table string ---
    table_lines = []
    table_lines.append("┌──────────────────────────────┬──────────────┬──────────────┬──────────────────┐")
    table_lines.append("│ Metric                       │ Dijkstra     │ Standard A*  │ Risk-Aware A*    │")
    table_lines.append("├──────────────────────────────┼──────────────┼──────────────┼──────────────────┤")
    table_lines.append(f"│ Mean Path Length (cells)     │ {mean_len_dij:>11.1f}  │ {mean_len_std:>11.1f}  │ {mean_len_ra:>15.1f}  │")
    table_lines.append(f"│ Path Length vs Dijkstra (%)  │ {'0%':>11s}  │ {overhead_std:>10.1f}%  │ {overhead_ra:>14.1f}%  │")
    table_lines.append(f"│ Mean Min Clearance (cells)   │ {mean_clr_dij:>11.2f}  │ {mean_clr_std:>11.2f}  │ {mean_clr_ra:>15.2f}  │")
    table_lines.append(f"│ Mean Safety Violations       │ {mean_viol_dij:>11.1f}  │ {mean_viol_std:>11.1f}  │ {mean_viol_ra:>15.1f}  │")
    table_lines.append(f"│ Safety Violation Rate (%)    │ {viol_rate_dij:>10.1f}%  │ {viol_rate_std:>10.1f}%  │ {viol_rate_ra:>14.1f}%  │")
    table_lines.append("└──────────────────────────────┴──────────────┴──────────────┴──────────────────┘")

    table_str = "\n".join(table_lines)

    # --- Print table ---
    print()
    print(table_str)
    print(f"\n  Completed {completed} / {n_trials} trials  (seed={seed})")

    # --- Save outputs ---
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Save log
    log_path = os.path.join(out_dir, "eval_log_extended.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Extended Evaluation — Dijkstra vs Standard A* vs Risk-Aware A*\n")
        f.write(f"Trials: {completed}/{n_trials}  Seed: {seed}\n\n")
        f.write(table_str + "\n")
    print(f"  Log saved → {log_path}")

    # --- Grouped bar chart ---
    labels = ["Mean Min\nClearance", "Safety Violations\n(mean)", "Path Length\nOverhead (%)"]
    dij_vals = [mean_clr_dij, mean_viol_dij, overhead_dij]
    std_vals = [mean_clr_std, mean_viol_std, overhead_std]
    ra_vals  = [mean_clr_ra,  mean_viol_ra,  overhead_ra]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, dij_vals, width, label="Dijkstra",
                   color="#FF9800", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x,         std_vals, width, label="Standard A*",
                   color="#4285F4", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, ra_vals,  width, label="Risk-Aware A*",
                   color="#34A853", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Value")
    ax.set_title(
        "MediNav Planner Comparison — Dijkstra vs Standard A* vs Risk-Aware A* (30 trials)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Custom legend (explicit, not relying on default positioning)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF9800", edgecolor="white", label="Dijkstra"),
        Patch(facecolor="#4285F4", edgecolor="white", label="Standard A*"),
        Patch(facecolor="#34A853", edgecolor="white", label="Risk-Aware A*"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars3, fmt="%.1f", padding=3, fontsize=8)

    fig.tight_layout()

    chart_path = os.path.join(out_dir, "evaluation_extended.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"\n  Completed {completed}/30 trials. Chart saved to outputs/evaluation_extended.png")


if __name__ == "__main__":
    evaluate()
