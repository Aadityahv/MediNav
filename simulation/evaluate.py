"""
Evaluation Script — Standard A* vs Risk-Aware A*
Runs 20 Monte Carlo trials on the synthetic hospital grid and prints a
comparison statistics table. Saves a grouped bar chart to outputs/.
"""

import sys
import io

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
# Hospital grid generator (inlined for self-containment)
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
# A* Planners
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

    std_lengths = []
    std_min_clearances = []
    std_violations = []

    ra_lengths = []
    ra_min_clearances = []
    ra_violations = []

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

        path_std = standard_astar(grid, start, goal)
        path_ra = risk_astar(grid, risk_map, start, goal, lambda_weight=8.0)

        if path_std is None or path_ra is None:
            continue

        # --- Metrics for standard A* ---
        clearances_std = [dist_map[r, c] for r, c in path_std]
        std_lengths.append(len(path_std))
        std_min_clearances.append(min(clearances_std))
        std_violations.append(sum(1 for d in clearances_std if d < 3))

        # --- Metrics for risk-aware A* ---
        clearances_ra = [dist_map[r, c] for r, c in path_ra]
        ra_lengths.append(len(path_ra))
        ra_min_clearances.append(min(clearances_ra))
        ra_violations.append(sum(1 for d in clearances_ra if d < 3))

        completed += 1
        print(f"  Trial {completed}/{n_trials} done  (start={start}, goal={goal})")

    if completed == 0:
        print("ERROR: No trials completed — grid may be too disconnected.")
        return

    # --- Compute summary statistics ---
    mean_len_std = np.mean(std_lengths)
    mean_len_ra = np.mean(ra_lengths)
    overhead_pct = (mean_len_ra - mean_len_std) / mean_len_std * 100

    mean_clr_std = np.mean(std_min_clearances)
    mean_clr_ra = np.mean(ra_min_clearances)

    mean_viol_std = np.mean(std_violations)
    mean_viol_ra = np.mean(ra_violations)

    total_cells_std = sum(std_lengths)
    total_cells_ra = sum(ra_lengths)
    viol_rate_std = sum(std_violations) / total_cells_std * 100
    viol_rate_ra = sum(ra_violations) / total_cells_ra * 100

    # --- Print table ---
    print()
    print("┌─────────────────────────────┬─────────────────┬─────────────────┐")
    print("│ Metric                      │ Standard A*     │ Risk-Aware A*   │")
    print("├─────────────────────────────┼─────────────────┼─────────────────┤")
    print(f"│ Mean Path Length (cells)    │ {mean_len_std:>14.1f}  │ {mean_len_ra:>14.1f}  │")
    print(f"│ Path Length Overhead (%)    │ {'0%':>14s}  │ {overhead_pct:>13.1f}%  │")
    print(f"│ Mean Min Clearance (cells)  │ {mean_clr_std:>14.2f}  │ {mean_clr_ra:>14.2f}  │")
    print(f"│ Mean Safety Violations      │ {mean_viol_std:>14.1f}  │ {mean_viol_ra:>14.1f}  │")
    print(f"│ Safety Violation Rate (%)   │ {viol_rate_std:>13.1f}%  │ {viol_rate_ra:>13.1f}%  │")
    print("└─────────────────────────────┴─────────────────┴─────────────────┘")
    print(f"\n  Completed {completed} / {n_trials} trials  (seed={seed})")

    # --- Bar chart ---
    labels = ["Mean Min\nClearance", "Safety Violations\n(mean)", "Path Length\nOverhead (%)"]
    std_vals = [mean_clr_std, mean_viol_std, 0.0]
    ra_vals = [mean_clr_ra, mean_viol_ra, overhead_pct]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, std_vals, width, label="Standard A*",
                   color="#4285F4", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ra_vals, width, label="Risk-Aware A*",
                   color="#34A853", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Value")
    ax.set_title("Standard A* vs Risk-Aware A* — Evaluation Metrics (30 trials)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)
    fig.tight_layout()

    import os
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evaluation_results.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved → {out_path}")


if __name__ == "__main__":
    evaluate()
