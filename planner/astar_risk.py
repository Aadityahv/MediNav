"""
Risk-Aware A* Pathfinding Algorithm
MediNav Project — planner/astar_risk.py

Cost function:
    f(n) = g_dist(n) + lambda * g_risk(n) + h(n)

    g_dist(n)  = cumulative Euclidean distance from start -> n
    g_risk(n)  = cumulative risk scores along the path to n
    lambda     = safety weight (higher = more risk-averse)
    h(n)       = Euclidean distance from n -> goal (admissible heuristic)

Extended with:
    - Multi-factor cost function (+ turning penalty)
    - Real-time replanning with dynamic risk updates
"""

import heapq
import math
import numpy as np

# ---------------------------------------------------------------------------
# GLOBAL SAFETY FLAGS
# ---------------------------------------------------------------------------
USE_DYNAMIC_RISK = True
USE_HUMANS = True
USE_REALTIME = True
USE_NN = True
SAFE_MODE = True


# ---------------------------------------------------------------------------
# 1. ORIGINAL — astar_risk (UNTOUCHED)
# ---------------------------------------------------------------------------

def astar_risk(grid, risk_map, start, goal, lambda_weight=5.0):
    """
    Risk-Aware A* on a 2D grid.

    Parameters
    ----------
    grid         : np.ndarray (H x W), 0 = free, 1 = wall
    risk_map     : np.ndarray (H x W), float in [0.0, 1.0]
    start        : (row, col) tuple
    goal         : (row, col) tuple
    lambda_weight: float — scales the risk term vs. distance

    Returns
    -------
    List of (row, col) tuples from start -> goal, or None if unreachable.
    """

    rows, cols = grid.shape

    def h(node):
        """Euclidean heuristic — admissible because it never overestimates."""
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    # -----------------------------------------------------------------------
    # Priority queue entries: (f_cost, g_dist, g_risk, row, col)
    #   f_cost = g_dist + lambda*g_risk + h
    # We store g_dist and g_risk separately to reconstruct stats later.
    # -----------------------------------------------------------------------
    start_f = h(start)
    # heap: (f, g_dist, g_risk, row, col)
    open_heap = [(start_f, 0.0, 0.0, start[0], start[1])]

    # Best known (g_dist, g_risk) for each cell
    best = {}  # (row, col) -> (g_dist, g_risk)
    best[start] = (0.0, 0.0)

    # For path reconstruction
    came_from = {start: None}

    # 8-connected neighbours
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g_d, g_r, r, c = heapq.heappop(open_heap)
        node = (r, c)

        # Skip if we've already found a better path to this cell
        if node in best:
            bd, br = best[node]
            if g_d > bd + 1e-9 or g_r > br + 1e-9:
                # Only skip if both components are worse (or equal-worse)
                if g_d >= bd - 1e-9 and g_r >= br - 1e-9:
                    continue

        if node == goal:
            # Reconstruct path
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
                continue  # wall

            neighbour = (nr, nc)

            # Step distance (diagonal = sqrt(2))
            step_dist = math.sqrt(dr * dr + dc * dc)
            # Step risk = risk at the neighbour cell
            step_risk = float(risk_map[nr, nc])

            new_g_d = g_d + step_dist
            new_g_r = g_r + step_risk

            if neighbour in best:
                bd, br = best[neighbour]
                # Domination check — skip if not strictly better in combined cost
                new_f = new_g_d + lambda_weight * new_g_r + h(neighbour)
                old_f = bd + lambda_weight * br + h(neighbour)
                if new_f >= old_f - 1e-9:
                    continue

            best[neighbour] = (new_g_d, new_g_r)
            came_from[neighbour] = node
            new_f = new_g_d + lambda_weight * new_g_r + h(neighbour)
            heapq.heappush(open_heap, (new_f, new_g_d, new_g_r, nr, nc))

    return None  # No path found


# ---------------------------------------------------------------------------
# 2. ORIGINAL — astar_standard (UNTOUCHED)
# ---------------------------------------------------------------------------

def astar_standard(grid, start, goal):
    """
    Standard A* — minimises Euclidean distance only (lambda = 0).
    Equivalent to astar_risk with a zero risk_map and lambda_weight=0.
    """
    zero_risk = np.zeros_like(grid, dtype=float)
    return astar_risk(grid, zero_risk, start, goal, lambda_weight=0.0)


# ---------------------------------------------------------------------------
# 3. NEW — Multi-factor cost A* (with turning penalty)
# ---------------------------------------------------------------------------

def astar_multifactor(grid, risk_map, start, goal,
                      lambda_weight=5.0, beta=0.5):
    """
    Multi-factor A* with turning penalty.

    Cost = distance + alpha * risk + beta * turning_cost

    Parameters
    ----------
    grid          : np.ndarray (H x W), 0=free, 1=wall
    risk_map      : np.ndarray (H x W), float in [0, 1]
    start         : (row, col)
    goal          : (row, col)
    lambda_weight : float — risk weight
    beta          : float — turning penalty weight

    Returns
    -------
    List of (row, col) tuples from start -> goal, or None.
    """
    rows, cols = grid.shape

    def h(node):
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    # heap: (f, g_combined, prev_dr, prev_dc, row, col)
    open_heap = [(h(start), 0.0, 0, 0, start[0], start[1])]

    # best[node] = best_g_combined
    best = {start: 0.0}
    came_from = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g, prev_dr, prev_dc, r, c = heapq.heappop(open_heap)
        node = (r, c)

        if g > best.get(node, float('inf')) + 1e-9:
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

            neighbour = (nr, nc)

            # Distance cost
            step_dist = math.sqrt(dr * dr + dc * dc)
            # Risk cost
            step_risk = lambda_weight * float(risk_map[nr, nc])

            # Turning cost: angle between previous direction and current
            turning_cost = 0.0
            if prev_dr != 0 or prev_dc != 0:
                # Dot product / magnitudes for angle
                mag_prev = math.sqrt(prev_dr ** 2 + prev_dc ** 2)
                mag_curr = math.sqrt(dr ** 2 + dc ** 2)
                dot = prev_dr * dr + prev_dc * dc
                cos_angle = dot / (mag_prev * mag_curr + 1e-8)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = math.acos(cos_angle)  # radians [0, pi]
                turning_cost = beta * angle / math.pi  # normalize to [0, beta]

            new_g = g + step_dist + step_risk + turning_cost

            if new_g < best.get(neighbour, float('inf')) - 1e-9:
                best[neighbour] = new_g
                came_from[neighbour] = node
                new_f = new_g + h(neighbour)
                heapq.heappush(open_heap, (new_f, new_g, dr, dc, nr, nc))

    return None


# ---------------------------------------------------------------------------
# 4. NEW — Real-time replanning A*
# ---------------------------------------------------------------------------

def astar_realtime(grid, risk_map_fn, start, goal, humans=None,
                   max_steps=500, replan_interval=5, lambda_weight=5.0):
    """
    Real-time replanning A*. Steps through the path one cell at a time,
    replanning periodically as the risk map changes (due to moving humans).

    Parameters
    ----------
    grid            : np.ndarray (H x W), 0=free, 1=wall
    risk_map_fn     : callable(humans) -> np.ndarray (H, W)
                      Function that returns the current risk map
    start           : (row, col)
    goal            : (row, col)
    humans          : list of dicts with 'pos' and 'vel' keys (or None)
    max_steps       : int — maximum steps before giving up
    replan_interval : int — replan every N steps
    lambda_weight   : float

    Returns
    -------
    trajectory      : list of (row, col) — actual path taken
    replan_events   : list of int — step numbers where replanning occurred
    """
    if not USE_REALTIME:
        # Fallback: single-shot planning
        risk_map = risk_map_fn(humans)
        path = astar_risk(grid, risk_map, start, goal, lambda_weight)
        return path if path else [start], []

    trajectory = [start]
    replan_events = []
    current = start
    current_path = None
    path_index = 0

    for step in range(max_steps):
        if current == goal:
            break

        # Replan periodically or if no path
        if current_path is None or step % replan_interval == 0:
            # Update humans
            if humans:
                _update_humans_inline(humans, grid)

            # Get current risk map
            risk_map = risk_map_fn(humans)

            # Plan from current position
            current_path = astar_risk(grid, risk_map, current, goal, lambda_weight)
            if current_path is None or len(current_path) < 2:
                break  # stuck
            path_index = 1  # skip current position
            replan_events.append(step)

        # Move one step along planned path
        if path_index < len(current_path):
            current = current_path[path_index]
            path_index += 1
        else:
            current_path = None  # force replan
            continue

        trajectory.append(current)

    return trajectory, replan_events


def _update_humans_inline(humans, grid):
    """
    Update human positions in-place (used by real-time planner).
    Moves humans by their velocity, bouncing off walls.
    """
    H, W = grid.shape
    for human in humans:
        if 'pos' not in human or 'vel' not in human:
            continue
        pos = np.array(human['pos'], dtype=float)
        vel = np.array(human['vel'], dtype=float)
        new_pos = pos + vel

        # Bounds check
        new_pos[0] = np.clip(new_pos[0], 1, H - 2)
        new_pos[1] = np.clip(new_pos[1], 1, W - 2)

        # Wall collision -> bounce
        r, c = int(new_pos[0]), int(new_pos[1])
        if grid[r, c] == 1:
            human['vel'] = [-vel[0], -vel[1]]
        else:
            human['pos'] = (r, c)


# ---------------------------------------------------------------------------
# Quick self-test on a 20x20 grid
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    SIZE = 20
    grid = np.zeros((SIZE, SIZE), dtype=int)

    # Add a horizontal wall with a gap
    grid[10, :15] = 1
    grid[10, 16:] = 1  # gap at col 15

    # Risk map: high near the wall row
    risk = np.zeros((SIZE, SIZE), dtype=float)
    for r in range(SIZE):
        for c in range(SIZE):
            dist_to_wall = abs(r - 10)
            risk[r, c] = max(0.0, 1.0 - dist_to_wall / 5.0)

    start = (2, 2)
    goal = (18, 18)

    # --- Original tests (unchanged) ---
    path_std = astar_standard(grid, start, goal)
    path_risk = astar_risk(grid, risk, start, goal, lambda_weight=5.0)

    if path_std:
        print(f"Standard A*      — path length: {len(path_std)} cells")
    else:
        print("Standard A* — no path found")

    if path_risk:
        print(f"Risk-Aware A*    — path length: {len(path_risk)} cells")
    else:
        print("Risk-Aware A* — no path found")

    if path_std and path_risk:
        overhead = (len(path_risk) - len(path_std)) / len(path_std) * 100
        print(f"Safety overhead  : {overhead:.1f}%")

    # --- New tests (optional) ---
    print("\n--- Multi-Factor A* Test ---")
    try:
        path_mf = astar_multifactor(grid, risk, start, goal,
                                     lambda_weight=5.0, beta=0.5)
        if path_mf:
            print(f"Multi-Factor A*  — path length: {len(path_mf)} cells")
        else:
            print("Multi-Factor A* — no path found")
    except Exception as e:
        if SAFE_MODE:
            print(f"  Multi-Factor A* failed ({e}), skipping")

    if USE_REALTIME:
        print("\n--- Real-Time Replanning A* Test ---")
        try:
            def risk_fn(humans=None):
                return risk

            trajectory, replans = astar_realtime(
                grid, risk_fn, start, goal,
                humans=None, max_steps=200, replan_interval=5
            )
            print(f"Real-Time A*     — trajectory: {len(trajectory)} cells, replans: {len(replans)}")
        except Exception as e:
            if SAFE_MODE:
                print(f"  Real-Time A* failed ({e}), skipping")
