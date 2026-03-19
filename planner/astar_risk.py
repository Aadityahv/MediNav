"""
Risk-Aware A* Pathfinding Algorithm
MediNav Project — planner/astar_risk.py

Cost function:
    f(n) = g_dist(n) + lambda * g_risk(n) + h(n)

    g_dist(n)  = cumulative Euclidean distance from start → n
    g_risk(n)  = cumulative risk scores along the path to n
    lambda     = safety weight (higher = more risk-averse)
    h(n)       = Euclidean distance from n → goal (admissible heuristic)
"""

import heapq
import math
import numpy as np


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
    List of (row, col) tuples from start → goal, or None if unreachable.
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


def astar_standard(grid, start, goal):
    """
    Standard A* — minimises Euclidean distance only (lambda = 0).
    Equivalent to astar_risk with a zero risk_map and lambda_weight=0.
    """
    zero_risk = np.zeros_like(grid, dtype=float)
    return astar_risk(grid, zero_risk, start, goal, lambda_weight=0.0)


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
