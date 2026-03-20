"""
Synthetic Hospital Ward Environment Generator
Generates a 2D grid for robot navigation testing.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def generate_hospital_grid(size=100):
    """
    Generate a synthetic hospital ward environment.

    Returns:
        grid: 100x100 numpy array, 0=free, 1=wall
        risk_map: 100x100 numpy array, 0.0 to 1.0
        landmarks: list of (row, col) tuples at corridor junctions
    """
    # Start with all walls
    grid = np.ones((size, size), dtype=np.int32)

    # --- Carve corridors ---

    # 1. Top horizontal corridor: rows 15-30, full width minus outer walls
    grid[15:31, 1:size - 1] = 0

    # 2. Bottom horizontal corridor: rows 65-80, full width minus outer walls
    grid[65:81, 1:size - 1] = 0

    # 3. Right vertical corridor connecting them: cols 70-85, rows 30-65
    grid[30:66, 70:86] = 0

    # 4. Central (narrower) horizontal service corridor: rows 45-55, cols 5-70
    grid[45:56, 5:71] = 0

    # --- Outer walls (border) ---
    grid[0, :] = 1
    grid[size - 1, :] = 1
    grid[:, 0] = 1
    grid[:, size - 1] = 1

    # --- Risk map ---
    # Distance from walls: where wall==1, distance==0
    free_mask = (grid == 0).astype(np.float64)
    dist_from_walls = distance_transform_edt(free_mask)
    risk_map = np.exp(-1.5 * dist_from_walls)
    risk_map = np.clip(risk_map, 0.0, 1.0)

    # --- Landmarks at corridor junction centres ---
    landmarks = [
        (22, 70),   # Top corridor meets right vertical corridor (top-right T)
        (22, 38),   # Mid-point of top corridor
        (50, 70),   # Central corridor meets right vertical corridor
        (50, 38),   # Mid-point of central corridor
        (72, 70),   # Bottom corridor meets right vertical corridor (bottom-right T)
        (72, 38),   # Mid-point of bottom corridor
    ]

    return grid, risk_map, landmarks


def get_random_free_cell(grid, min_wall_dist=5):
    """
    Return a random (row, col) that is free and at least `min_wall_dist` cells
    from any wall.
    """
    free_mask = (grid == 0).astype(np.float64)
    dist = distance_transform_edt(free_mask)
    candidates = np.argwhere(dist >= min_wall_dist)
    if len(candidates) == 0:
        raise ValueError("No free cell found with the required wall distance.")
    idx = np.random.randint(len(candidates))
    return tuple(candidates[idx])


if __name__ == "__main__":
    grid, risk_map, landmarks = generate_hospital_grid()
    free_cells = int(np.sum(grid == 0))
    print(f"Grid shape      : {grid.shape}")
    print(f"Free cells      : {free_cells}")
    print(f"Wall cells      : {grid.size - free_cells}")
    print(f"Landmark points : {len(landmarks)}")
    print(f"Landmarks       : {landmarks}")
    print(f"Risk map range  : [{risk_map.min():.4f}, {risk_map.max():.4f}]")
