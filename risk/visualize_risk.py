"""
MediNav — Risk Visualization
risk/visualize_risk.py

Static 4-panel visualization (original) + optional dynamic animation
showing moving humans, evolving risk field, and robot path.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
# 1. ORIGINAL — static risk visualization (preserved exactly)
# ---------------------------------------------------------------------------

def plot_static_risk(p_map, risk_map):
    """
    Create the original 4-panel static risk visualization.

    Parameters
    ----------
    p_map    : np.ndarray (H, W) — occupancy probability map
    risk_map : np.ndarray (H, W) — risk map
    """
    if p_map.shape != risk_map.shape:
        raise ValueError("Mismatch between occupancy map and risk map dimensions")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].imshow(p_map, cmap='gray')
    axs[0, 0].set_title("Occupancy Probability Map")

    im1 = axs[0, 1].imshow(risk_map, cmap='hot')
    axs[0, 1].set_title("Risk Heatmap")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    axs[1, 0].imshow(p_map, cmap='gray')
    axs[1, 0].imshow(risk_map, cmap='hot', alpha=0.6)
    axs[1, 0].set_title("Overlay (Risk on Map)")

    axs[1, 1].hist(risk_map.flatten(), bins=50)
    axs[1, 1].set_title("Risk Distribution")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/risk_visualization.png", dpi=150)
    print("Saved outputs/risk_visualization.png")
    print("\nVisualization saved to outputs/risk_visualization.png")
    plt.close()


# ---------------------------------------------------------------------------
# 2. NEW — dynamic risk animation
# ---------------------------------------------------------------------------

def animate_dynamic_risk(grid, risk_maps, humans_history, robot_path,
                         save_path="outputs/risk_animation.gif"):
    """
    Create an animated GIF showing:
      - The hospital grid (walls)
      - Evolving risk heatmap (from risk_maps sequence)
      - Moving humans (from humans_history)
      - Robot traversing its path

    Parameters
    ----------
    grid            : np.ndarray (H, W) — 0=free, 1=wall
    risk_maps       : list of np.ndarray (H, W) — risk map per frame
    humans_history  : list of list of (x, y) — human positions per frame
    robot_path      : list of (row, col) — robot trajectory
    save_path       : str — output GIF path
    """
    try:
        # Ensure Pillow is available for GIF saving
        import PIL  # noqa: F401
    except ImportError:
        print("  WARNING: Pillow not installed, attempting to install...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])

    H, W = grid.shape
    n_frames = min(len(risk_maps), len(robot_path))

    if n_frames == 0:
        print("  WARNING: No frames to animate, skipping animation")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Static wall layer
    wall_display = np.where(grid == 1, 0.0, 0.2)
    ax.imshow(wall_display, cmap='gray', vmin=0, vmax=1, origin='upper')

    # Dynamic elements
    risk_img = ax.imshow(np.zeros((H, W, 4)), origin='upper')
    human_scatter = ax.scatter([], [], c='cyan', s=80, marker='o',
                               edgecolors='white', linewidths=1.0, zorder=10,
                               label='Humans')
    robot_trail, = ax.plot([], [], color='tab:blue', linewidth=2.5, zorder=8)
    robot_dot, = ax.plot([], [], 'wo', markersize=8, zorder=9)

    # Start/goal markers
    if len(robot_path) > 0:
        s = robot_path[0]
        g = robot_path[-1]
        ax.plot(s[1], s[0], 'go', markersize=10, label='Start', zorder=11)
        ax.plot(g[1], g[0], 'y*', markersize=15, label='Goal', zorder=11)

    ax.set_title("MediNav — Dynamic Risk-Aware Navigation", color='white')
    ax.legend(loc='upper left', fontsize=8, facecolor='#0d1b2a',
              edgecolor='gray', labelcolor='white')

    def init():
        risk_img.set_data(np.zeros((H, W, 4)))
        human_scatter.set_offsets(np.empty((0, 2)))
        robot_trail.set_data([], [])
        robot_dot.set_data([], [])
        return risk_img, human_scatter, robot_trail, robot_dot

    def update(frame):
        # Risk heatmap
        rm = risk_maps[min(frame, len(risk_maps) - 1)]
        risk_rgba = np.zeros((H, W, 4), dtype=float)
        risk_rgba[:, :, 0] = rm
        risk_rgba[:, :, 3] = rm * 0.4
        risk_img.set_data(risk_rgba)

        # Humans
        if frame < len(humans_history) and humans_history[frame]:
            h_positions = np.array(humans_history[frame])
            human_scatter.set_offsets(h_positions[:, ::-1])  # (row, col) -> (col, row) for matplotlib
        else:
            human_scatter.set_offsets(np.empty((0, 2)))

        # Robot trail
        if frame > 0 and frame <= len(robot_path):
            trail = robot_path[:frame]
            xs = [c for r, c in trail]
            ys = [r for r, c in trail]
            robot_trail.set_data(xs, ys)
            robot_dot.set_data([xs[-1]], [ys[-1]])

        return risk_img, human_scatter, robot_trail, robot_dot

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   init_func=init, interval=80, blit=True)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    writer = animation.PillowWriter(fps=12)
    anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"  Animation saved -> {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists("outputs"):
        raise FileNotFoundError("outputs folder not found")

    try:
        p_map = np.load("outputs/occupancy_grid.npy")
        risk_map = np.load("outputs/risk_map.npy")
    except Exception as e:
        raise FileNotFoundError("Required .npy files not found. Run SLAM and risk_map first.") from e

    # Always run original static visualization
    plot_static_risk(p_map, risk_map)

    # Optionally run dynamic animation
    if USE_DYNAMIC_RISK:
        try:
            from scipy.ndimage import distance_transform_edt
            import heapq
            import math as _math

            # Import compute_human_risk (handle both module and script execution)
            try:
                from risk.risk_map import compute_human_risk
            except ImportError:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(__file__))
                from risk_map import compute_human_risk

            print("\n--- Generating Dynamic Risk Animation ---")

            # ----------------------------------------------------------
            # Synthetic hospital grid — reliable corridors for clear
            # visual demonstration of collision-avoidance replanning
            # ----------------------------------------------------------
            HA, WA = 100, 100
            anim_grid = np.ones((HA, WA), dtype=int)

            # Carve corridors
            anim_grid[10:21, 0:100] = 0   # Top horizontal corridor
            anim_grid[70:81, 0:100] = 0   # Bottom horizontal corridor
            anim_grid[20:71, 70:81] = 0   # Right vertical connector
            anim_grid[40:51, 0:81]  = 0   # Narrow horizontal shortcut

            # Outer walls
            anim_grid[0, :]  = 1
            anim_grid[99, :] = 1
            anim_grid[:, 0]  = 1
            anim_grid[:, 99] = 1

            # Equipment obstacles in the shortcut
            anim_grid[43:45, 20:22] = 1
            anim_grid[43:45, 40:42] = 1
            anim_grid[45:47, 58:60] = 1
            anim_grid[42:44, 70:72] = 1

            # Wall-proximity risk (exponential decay from walls)
            _free_mask = (anim_grid == 0).astype(float)
            _wall_dist = distance_transform_edt(_free_mask)
            _wall_risk = np.clip(np.exp(-1.5 * _wall_dist), 0, 1)

            # ---- Inline A* for self-contained animation ----
            def _astar_anim(grid, rmap, start, goal, lam=8.0):
                rows, cols = grid.shape
                def _h(a, b):
                    return _math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
                heap = [(_h(start, goal), 0.0, start[0], start[1])]
                best = {start: 0.0}
                parent = {start: None}
                dirs = [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(-1,1),(1,-1),(1,1)]
                while heap:
                    f, gc, r, c = heapq.heappop(heap)
                    nd = (r, c)
                    if nd == goal:
                        path = []
                        cur = goal
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return path
                    if gc > best.get(nd, float('inf')):
                        continue
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if grid[nr, nc] == 1:
                            continue
                        nb = (nr, nc)
                        step = _math.sqrt(dr*dr + dc*dc)
                        ng = gc + step + lam * rmap[nr, nc]
                        if ng < best.get(nb, float('inf')):
                            best[nb] = ng
                            parent[nb] = nd
                            heapq.heappush(heap,
                                           (ng + _h(nb, goal), ng, nr, nc))
                return []

            s_pos = (15, 5)
            g_pos = (75, 75)

            # Plan initial path (wall risk only — no humans yet)
            init_path = _astar_anim(anim_grid, _wall_risk, s_pos, g_pos)
            if not init_path or len(init_path) < 2:
                raise ValueError("A* found no path on animation grid")
            print(f"  Initial A* path: {len(init_path)} cells")

            # ---- Strategic human placement for near-collision visuals ----
            np.random.seed(42)

            # Human 0 — walks UP through the right vertical corridor,
            #            head-on collision course with robot going DOWN
            h_positions = [np.array([55.0, 75.0])]
            h_velocities = [np.array([-0.35, 0.0])]

            # Human 1 — walks LEFT through the top corridor,
            #            approaching robot head-on in the early frames
            h_positions.append(np.array([15.0, 40.0]))
            h_velocities.append(np.array([0.0, -0.25]))

            # Humans 2-4 — ambient corridor traffic (visual background)
            h_positions.append(np.array([75.0, 30.0]))
            h_velocities.append(np.array([0.0, 0.35]))

            h_positions.append(np.array([45.0, 25.0]))
            h_velocities.append(np.array([0.0, 0.30]))

            h_positions.append(np.array([75.0, 65.0]))
            h_velocities.append(np.array([0.0, -0.20]))

            # ---- Frame-by-frame simulation with real-time replanning ----
            max_frames = min(len(init_path) + 30, 160)

            robot_traj  = []
            risk_seq    = []
            humans_seq  = []

            cur_pos  = s_pos
            path_q   = list(init_path[1:])   # remaining waypoints

            for frame in range(max_frames):
                # Record robot position at this frame
                robot_traj.append(cur_pos)

                # Update all human positions (bounce off walls)
                cur_humans = []
                for i in range(len(h_positions)):
                    pos = h_positions[i]
                    vel = h_velocities[i]
                    new_pos = pos + vel
                    new_pos[0] = np.clip(new_pos[0], 1, HA - 2)
                    new_pos[1] = np.clip(new_pos[1], 1, WA - 2)
                    ri, ci = int(new_pos[0]), int(new_pos[1])
                    if anim_grid[ri, ci] == 1:
                        h_velocities[i] = -vel   # bounce
                        new_pos = pos.copy()
                    h_positions[i] = new_pos
                    cur_humans.append((int(new_pos[0]), int(new_pos[1])))
                humans_seq.append(cur_humans)

                # Compute dynamic risk: wall proximity + human proximity
                h_risk, _ = compute_human_risk((HA, WA), cur_humans)
                f_risk = 0.7 * _wall_risk + 0.3 * h_risk
                f_risk = (f_risk - f_risk.min()) / \
                         (f_risk.max() - f_risk.min() + 1e-8)
                risk_seq.append(f_risk)

                # If already at goal, just keep recording
                if cur_pos == g_pos:
                    continue

                # Nearest human distance from robot
                min_hd = float('inf')
                for hp in cur_humans:
                    d = _math.sqrt((cur_pos[0] - hp[0])**2 +
                                   (cur_pos[1] - hp[1])**2)
                    min_hd = min(min_hd, d)

                # Replan when: human nearby, path exhausted, or periodic
                if min_hd < 12 or not path_q or frame % 20 == 0:
                    new_path = _astar_anim(anim_grid, f_risk,
                                           cur_pos, g_pos)
                    if new_path and len(new_path) > 1:
                        path_q = list(new_path[1:])

                # Advance robot one step along planned path
                if path_q:
                    cur_pos = path_q.pop(0)

            # Trim all sequences to same length
            n = min(len(risk_seq), len(robot_traj), len(humans_seq))
            risk_seq   = risk_seq[:n]
            robot_traj = robot_traj[:n]
            humans_seq = humans_seq[:n]

            print(f"  Animation: {n} frames, {len(h_positions)} humans")
            print(f"  Robot trajectory: {len(robot_traj)} steps")

            animate_dynamic_risk(anim_grid, risk_seq, humans_seq,
                                 robot_traj)

        except Exception as e:
            if SAFE_MODE:
                print(f"  WARNING: Animation failed ({e}), skipping")
            else:
                raise
