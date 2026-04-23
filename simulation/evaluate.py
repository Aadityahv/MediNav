"""
Evaluation Script — Standard A* vs Risk-Aware A*
Runs 30 Monte Carlo trials on the synthetic hospital grid and prints a
comparison statistics table. Saves a grouped bar chart to outputs/.

Extended with:
  - Dynamic simulation (static vs real-time replanning)
  - Rule-based vs NN risk comparison
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
import os

# ---------------------------------------------------------------------------
# GLOBAL SAFETY FLAGS
# ---------------------------------------------------------------------------
USE_DYNAMIC_RISK = True
USE_HUMANS = True
USE_REALTIME = True
USE_NN = True
SAFE_MODE = True

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
# A* Planners (ORIGINAL — PRESERVED EXACTLY)
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
# ORIGINAL Evaluation (PRESERVED EXACTLY)
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

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evaluation_results.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved -> {out_path}")


# ---------------------------------------------------------------------------
# NEW — Dynamic evaluation (static vs real-time replanning)
# NOTE: This is a SUPPLEMENTARY evaluation (10 trials). The authoritative
#       results are the 30-trial Monte Carlo in evaluate() above.
# ---------------------------------------------------------------------------

def evaluate_dynamic(n_trials=10, seed=42):
    """
    Compare static planner vs real-time replanner with moving humans.

    NOTE: Supplementary evaluation — uses different sample size (10 trials)
    than the main 30-trial evaluation. Present separately in reports.
    """
    print("\n" + "=" * 60)
    print("  SUPPLEMENTARY: Dynamic Evaluation (10 trials)")
    print("  Static vs Real-Time Replanning with Moving Humans")
    print("=" * 60)

    np.random.seed(seed)
    grid, base_risk_map, dist_map = _generate_hospital_grid()
    H, W = grid.shape

    static_lengths = []
    static_violations = []
    rt_lengths = []
    rt_violations = []
    rt_replans = []

    completed = 0
    attempts = 0

    while completed < n_trials and attempts < n_trials * 5:
        attempts += 1
        try:
            start = _get_random_free_cell(grid, dist_map, min_wall_dist=5)
            for _ in range(200):
                goal = _get_random_free_cell(grid, dist_map, min_wall_dist=5)
                if _heuristic(start, goal) >= 20:
                    break
            else:
                continue
        except ValueError:
            continue

        # Generate humans
        rng = np.random.RandomState(seed + attempts)
        free_cells = np.argwhere(dist_map >= 3)
        num_h = 5
        if len(free_cells) < num_h:
            continue
        h_idxs = rng.choice(len(free_cells), size=num_h, replace=False)
        humans = [
            {'pos': tuple(free_cells[i]), 'vel': (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))}
            for i in h_idxs
        ]

        # Compute human-aware risk
        human_positions = [h['pos'] for h in humans]
        human_mask = np.ones((H, W), dtype=float)
        for (hr, hc) in human_positions:
            r, c = int(np.clip(hr, 0, H-1)), int(np.clip(hc, 0, W-1))
            human_mask[r, c] = 0
        human_dist = distance_transform_edt(human_mask)
        human_risk = np.exp(-human_dist * 0.3)
        human_risk = (human_risk - human_risk.min()) / (human_risk.max() - human_risk.min() + 1e-8)
        dynamic_risk = 0.7 * base_risk_map + 0.3 * human_risk
        dynamic_risk = np.clip(dynamic_risk, 0, 1)

        # Static planner (single-shot on initial dynamic risk)
        path_static = risk_astar(grid, dynamic_risk, start, goal, lambda_weight=8.0)
        if path_static is None:
            continue

        # Real-time replanner
        def risk_fn(h_list):
            if h_list:
                h_pos = [h['pos'] for h in h_list]
                hm = np.ones((H, W), dtype=float)
                for (hr, hc) in h_pos:
                    r, c = int(np.clip(hr, 0, H-1)), int(np.clip(hc, 0, W-1))
                    hm[r, c] = 0
                hd = distance_transform_edt(hm)
                hr_risk = np.exp(-hd * 0.3)
                hr_risk = (hr_risk - hr_risk.min()) / (hr_risk.max() - hr_risk.min() + 1e-8)
                return np.clip(0.7 * base_risk_map + 0.3 * hr_risk, 0, 1)
            return base_risk_map

        try:
            from planner.astar_risk import astar_realtime
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'planner'))
            from astar_risk import astar_realtime
        trajectory, replans = astar_realtime(
            grid, risk_fn, start, goal,
            humans=humans, max_steps=500, replan_interval=10,
            lambda_weight=8.0
        )

        if len(trajectory) < 2:
            continue

        # Metrics
        clr_static = [dist_map[r, c] for r, c in path_static]
        static_lengths.append(len(path_static))
        static_violations.append(sum(1 for d in clr_static if d < 3))

        clr_rt = [dist_map[r, c] for r, c in trajectory]
        rt_lengths.append(len(trajectory))
        rt_violations.append(sum(1 for d in clr_rt if d < 3))
        rt_replans.append(len(replans))

        completed += 1
        print(f"  Trial {completed}/{n_trials}")

    if completed == 0:
        print("  No dynamic trials completed.")
        return

    print()
    print(f"  {'Metric':<30s} {'Static':>12s} {'Real-Time':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Mean Path Length':<30s} {np.mean(static_lengths):>12.1f} {np.mean(rt_lengths):>12.1f}")
    print(f"  {'Mean Safety Violations':<30s} {np.mean(static_violations):>12.1f} {np.mean(rt_violations):>12.1f}")
    print(f"  {'Mean Replanning Events':<30s} {'N/A':>12s} {np.mean(rt_replans):>12.1f}")
    print(f"\n  Completed {completed}/{n_trials} supplementary dynamic trials")
    print(f"  NOTE: These are supplementary results (n={n_trials}), not comparable")
    print(f"        to the main 30-trial Monte Carlo evaluation.")


# ---------------------------------------------------------------------------
# NEW — NN evaluation (rule-based vs NN-blended risk)
# NOTE: Proof-of-concept demonstration. The NN is trained on rule-based
#       labels, so this shows the architecture can learn spatial risk
#       patterns. In future work, the NN could generalise to learned
#       features from real hospital sensor data.
# ---------------------------------------------------------------------------

def evaluate_nn(n_trials=10, seed=42):
    """
    Compare rule-based risk planner vs NN-blended risk planner.

    NOTE: Proof-of-concept — the NN is trained on rule-based labels and
    blended 50/50 back with the rule-based risk. This demonstrates the
    NN architecture can learn spatial risk patterns; it is not a claimed
    improvement over rule-based risk.
    """
    print("\n" + "=" * 60)
    print("  SUPPLEMENTARY: NN Proof-of-Concept (10 trials)")
    print("  Rule-Based vs NN-Blended Risk")
    print("=" * 60)

    np.random.seed(seed)
    grid, risk_rule, dist_map = _generate_hospital_grid()

    # Build NN risk map
    try:
        try:
            from risk.risk_nn import RiskNN
            from risk.risk_map import extract_features
        except ImportError:
            import sys as _sys
            _risk_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'risk')
            _sys.path.insert(0, _risk_dir)
            from risk_nn import RiskNN
            from risk_map import extract_features

        features = extract_features(dist_map)
        labels = risk_rule.flatten()

        model = RiskNN(input_dim=2, hidden_dim=16)
        model.train_model(features, labels, epochs=300, lr=0.01, verbose=False)
        nn_preds = model.predict(features).reshape(grid.shape)
        risk_nn = 0.5 * risk_rule + 0.5 * nn_preds
        risk_nn = np.clip(risk_nn, 0, 1)
    except Exception as e:
        if SAFE_MODE:
            print(f"  WARNING: NN training failed ({e}), using rule-based only")
            risk_nn = risk_rule
        else:
            raise

    rule_lengths = []
    rule_violations = []
    nn_lengths = []
    nn_violations = []

    completed = 0
    attempts = 0

    while completed < n_trials and attempts < n_trials * 5:
        attempts += 1
        try:
            start = _get_random_free_cell(grid, dist_map, min_wall_dist=5)
            for _ in range(200):
                goal = _get_random_free_cell(grid, dist_map, min_wall_dist=5)
                if _heuristic(start, goal) >= 20:
                    break
            else:
                continue
        except ValueError:
            continue

        path_rule = risk_astar(grid, risk_rule, start, goal, lambda_weight=8.0)
        path_nn = risk_astar(grid, risk_nn, start, goal, lambda_weight=8.0)

        if path_rule is None or path_nn is None:
            continue

        clr_rule = [dist_map[r, c] for r, c in path_rule]
        rule_lengths.append(len(path_rule))
        rule_violations.append(sum(1 for d in clr_rule if d < 3))

        clr_nn = [dist_map[r, c] for r, c in path_nn]
        nn_lengths.append(len(path_nn))
        nn_violations.append(sum(1 for d in clr_nn if d < 3))

        completed += 1
        print(f"  Trial {completed}/{n_trials}")

    if completed == 0:
        print("  No NN trials completed.")
        return

    print()
    print(f"  {'Metric':<30s} {'Rule-Based':>12s} {'NN-Blended':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Mean Path Length':<30s} {np.mean(rule_lengths):>12.1f} {np.mean(nn_lengths):>12.1f}")
    print(f"  {'Mean Safety Violations':<30s} {np.mean(rule_violations):>12.1f} {np.mean(nn_violations):>12.1f}")
    print(f"\n  Completed {completed}/{n_trials} supplementary NN trials")
    print(f"  NOTE: Proof-of-concept — NN trained on rule-based labels.")
    print(f"        Demonstrates NN architecture viability, not a claimed")
    print(f"        improvement over rule-based risk computation.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Always run original evaluation
    evaluate()

    # Dynamic evaluation (optional)
    if USE_DYNAMIC_RISK and USE_REALTIME:
        try:
            evaluate_dynamic(n_trials=10, seed=42)
        except Exception as e:
            if SAFE_MODE:
                print(f"\n  WARNING: Dynamic evaluation failed ({e}), skipping")
            else:
                raise

    # NN evaluation (optional)
    if USE_NN:
        try:
            evaluate_nn(n_trials=10, seed=42)
        except Exception as e:
            if SAFE_MODE:
                print(f"\n  WARNING: NN evaluation failed ({e}), skipping")
            else:
                raise
