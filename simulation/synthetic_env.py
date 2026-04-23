"""
Synthetic Hospital Ward Environment Generator
Generates a 2D grid for robot navigation testing.

Extended with:
  - Human agent generation and simulation
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# GLOBAL SAFETY FLAGS
# ---------------------------------------------------------------------------
USE_DYNAMIC_RISK = True
USE_HUMANS = True
USE_REALTIME = True
USE_NN = True
SAFE_MODE = True


# ---------------------------------------------------------------------------
# 1. ORIGINAL — generate_hospital_grid (PRESERVED EXACTLY)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. ORIGINAL — get_random_free_cell (PRESERVED EXACTLY)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. NEW — Human agent generation
# ---------------------------------------------------------------------------

def generate_humans(grid, num_humans=5, seed=42):
    """
    Generate simulated human agents in the hospital grid.

    Each human has a position and velocity. Humans stay in free cells
    and have small random velocities to simulate walking.

    Parameters
    ----------
    grid       : np.ndarray (H, W), 0=free, 1=wall
    num_humans : int — number of humans to generate
    seed       : int — random seed for reproducibility

    Returns
    -------
    humans : list of dicts, each with:
        'pos' : (row, col) — current position
        'vel' : (vr, vc)   — velocity (cells per timestep)
    """
    rng = np.random.RandomState(seed)

    free_mask = (grid == 0).astype(np.float64)
    dist = distance_transform_edt(free_mask)
    # Place humans at least 3 cells from walls
    candidates = np.argwhere(dist >= 3)

    if len(candidates) == 0:
        print("  WARNING: No valid positions for humans")
        return []

    num_humans = min(num_humans, len(candidates))
    idxs = rng.choice(len(candidates), size=num_humans, replace=False)

    humans = []
    for idx in idxs:
        pos = tuple(candidates[idx])
        # Small random velocity: -0.5 to 0.5 cells per step
        vel = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
        humans.append({
            'pos': pos,
            'vel': vel
        })

    return humans


# ---------------------------------------------------------------------------
# 4. NEW — Human update per timestep
# ---------------------------------------------------------------------------

def update_humans(humans, grid, dt=1.0):
    """
    Update human positions by one timestep.

    Movement logic:
    - Move each human by velocity * dt
    - Bounce off walls (reflect velocity component)
    - Keep humans in free cells

    Parameters
    ----------
    humans : list of dicts (modified in-place)
    grid   : np.ndarray (H, W), 0=free, 1=wall
    dt     : float — time step multiplier

    Returns
    -------
    humans : same list, updated in-place
    """
    H, W = grid.shape

    for human in humans:
        pos = np.array(human['pos'], dtype=float)
        vel = np.array(human['vel'], dtype=float)

        new_pos = pos + vel * dt

        # Clamp to grid bounds
        new_pos[0] = np.clip(new_pos[0], 1, H - 2)
        new_pos[1] = np.clip(new_pos[1], 1, W - 2)

        r, c = int(new_pos[0]), int(new_pos[1])

        # Wall collision -> bounce
        if grid[r, c] == 1:
            # Reflect velocity
            human['vel'] = (-vel[0], -vel[1])
            # Stay in place
        else:
            human['pos'] = (r, c)

    return humans


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid, risk_map, landmarks = generate_hospital_grid()
    free_cells = int(np.sum(grid == 0))
    print(f"Grid shape      : {grid.shape}")
    print(f"Free cells      : {free_cells}")
    print(f"Wall cells      : {grid.size - free_cells}")
    print(f"Landmark points : {len(landmarks)}")
    print(f"Landmarks       : {landmarks}")
    print(f"Risk map range  : [{risk_map.min():.4f}, {risk_map.max():.4f}]")

    # Human simulation (optional)
    if USE_HUMANS:
        print("\n--- Human Simulation ---")
        humans = generate_humans(grid, num_humans=5)
        print(f"Generated {len(humans)} humans:")
        for i, h in enumerate(humans):
            print(f"  Human {i}: pos={h['pos']}, vel=({h['vel'][0]:.2f}, {h['vel'][1]:.2f})")

        # Simulate 10 timesteps
        print("\nSimulating 10 timesteps...")
        for t in range(10):
            update_humans(humans, grid, dt=1.0)

        print("After 10 steps:")
        for i, h in enumerate(humans):
            print(f"  Human {i}: pos={h['pos']}")
