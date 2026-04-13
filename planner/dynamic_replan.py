"""
MediNav dynamic obstacle replanning demo.

Creates:
  - outputs/dynamic_replan.gif
  - outputs/dynamic_replan_comparison.png
"""

import heapq
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import distance_transform_edt


OUTPUT_DIR = Path("outputs")
GIF_PATH = OUTPUT_DIR / "dynamic_replan.gif"
PNG_PATH = OUTPUT_DIR / "dynamic_replan_comparison.png"

START = (15, 5)
GOAL = (75, 75)
MAX_STEPS = 500
LOOKAHEAD_STEPS = 5
RISK_ALPHA = 1.5


@dataclass
class Gurney:
    row: int
    col: int
    min_col: int
    max_col: int
    direction: int = 1

    def move(self):
        next_col = self.col + self.direction
        if next_col > self.max_col or next_col < self.min_col:
            self.direction *= -1
            next_col = self.col + self.direction
        self.col = next_col

    def cells(self):
        block = []
        for rr in range(self.row - 1, self.row + 2):
            for cc in range(self.col - 1, self.col + 2):
                block.append((rr, cc))
        return block

    def copy(self):
        return Gurney(
            row=self.row,
            col=self.col,
            min_col=self.min_col,
            max_col=self.max_col,
            direction=self.direction,
        )


def ensure_pillow_installed():
    try:
        import PIL  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])


def create_static_grid():
    grid_static = np.ones((100, 100), dtype=int)
    grid_static[10:21, 0:100] = 0
    grid_static[70:81, 0:100] = 0
    grid_static[20:71, 70:81] = 0
    grid_static[40:51, 0:81] = 0

    grid_static[0, :] = 1
    grid_static[99, :] = 1
    grid_static[:, 0] = 1
    grid_static[:, 99] = 1

    grid_static[43:45, 20:22] = 1
    grid_static[43:45, 40:42] = 1
    grid_static[45:47, 58:60] = 1
    grid_static[42:44, 70:72] = 1
    return grid_static


def initial_gurneys():
    return [
        Gurney(row=15, col=30, min_col=10, max_col=60, direction=1),
        Gurney(row=75, col=20, min_col=5, max_col=50, direction=1),
        Gurney(row=45, col=15, min_col=5, max_col=40, direction=1),
    ]


def apply_gurneys_to_grid(grid_static, gurneys):
    dynamic_grid = grid_static.copy()
    rows, cols = dynamic_grid.shape
    for gurney in gurneys:
        for rr, cc in gurney.cells():
            if 0 <= rr < rows and 0 <= cc < cols:
                dynamic_grid[rr, cc] = 1
    return dynamic_grid


def rebuild_risk_map(dynamic_grid, alpha=1.5):
    free_space = 1 - dynamic_grid
    distance = distance_transform_edt(free_space)
    risk_map = np.exp(-distance / alpha)
    risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-8)
    return risk_map


def reconstruct_path(came_from, goal):
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def risk_astar(grid, risk_map, start, goal, lambda_weight=8.0):
    rows, cols = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return None

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    def h(node):
        return math.hypot(node[0] - goal[0], node[1] - goal[1])

    open_heap = [(h(start), 0.0, start)]
    came_from = {start: None}
    best_cost = {start: 0.0}

    while open_heap:
        _, g_cost, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct_path(came_from, goal)

        if g_cost > best_cost.get(current, float("inf")) + 1e-9:
            continue

        cr, cc = current
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            step_dist = math.hypot(dr, dc)
            new_cost = g_cost + step_dist + lambda_weight * float(risk_map[nr, nc])
            neighbour = (nr, nc)

            if new_cost + 1e-9 < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                came_from[neighbour] = current
                f_cost = new_cost + h(neighbour)
                heapq.heappush(open_heap, (f_cost, new_cost, neighbour))

    return None


def snapshot_gurneys(gurneys):
    return [g.copy() for g in gurneys]


def simulate_replanning_run(
    grid_static,
    start,
    goal,
    lambda_weight=8.0,
    max_steps=500,
    lookahead_steps=5,
):
    gurneys = initial_gurneys()
    robot = start
    trail = [start]
    replans = 0

    dynamic_grid = apply_gurneys_to_grid(grid_static, gurneys)
    risk_map = rebuild_risk_map(dynamic_grid, alpha=RISK_ALPHA)
    planned_path = risk_astar(dynamic_grid, risk_map, robot, goal, lambda_weight=lambda_weight)
    if planned_path is None:
        planned_path = [robot]

    history = []

    for step in range(max_steps + 1):
        history.append(
            {
                "step": step,
                "robot": robot,
                "trail": trail.copy(),
                "planned_path": planned_path.copy(),
                "risk_map": risk_map.copy(),
                "dynamic_grid": dynamic_grid.copy(),
                "gurneys": snapshot_gurneys(gurneys),
                "replans": replans,
            }
        )

        if robot == goal or step == max_steps:
            break

        if len(planned_path) >= 2 and dynamic_grid[planned_path[1]] == 0:
            robot = planned_path[1]
            planned_path = planned_path[1:]

        trail.append(robot)

        for gurney in gurneys:
            gurney.move()

        dynamic_grid = apply_gurneys_to_grid(grid_static, gurneys)
        risk_map = rebuild_risk_map(dynamic_grid, alpha=RISK_ALPHA)

        blocked_ahead = False
        if len(planned_path) >= 2:
            ahead = planned_path[1 : 1 + lookahead_steps]
            blocked_ahead = any(dynamic_grid[r, c] == 1 for r, c in ahead)

        needs_replan = len(planned_path) < 2 or blocked_ahead
        if robot != goal and needs_replan:
            new_path = risk_astar(
                dynamic_grid,
                risk_map,
                robot,
                goal,
                lambda_weight=lambda_weight,
            )
            replans += 1
            if new_path is None:
                planned_path = [robot]
            else:
                planned_path = new_path

    return {
        "history": history,
        "number_of_replans": replans,
        "total_steps": max(0, len(trail) - 1),
        "final_path_length": len(trail),
        "reached_goal": robot == goal,
        "trail": trail,
        "final_robot": robot,
    }


def simulate_standard_run(grid_static, start, goal, max_steps=500):
    gurneys = initial_gurneys()
    robot = start
    trail = [start]
    collisions = []

    zero_risk = np.zeros_like(grid_static, dtype=float)
    fixed_path = risk_astar(grid_static, zero_risk, start, goal, lambda_weight=0.0)
    if fixed_path is None:
        fixed_path = [start]

    path_index = 0
    history = []

    for step in range(max_steps + 1):
        dynamic_grid = apply_gurneys_to_grid(grid_static, gurneys)
        if dynamic_grid[robot] == 1:
            collisions.append(robot)

        history.append(
            {
                "step": step,
                "robot": robot,
                "trail": trail.copy(),
                "planned_path": fixed_path,
                "dynamic_grid": dynamic_grid,
                "gurneys": snapshot_gurneys(gurneys),
                "collisions": collisions.copy(),
            }
        )

        if robot == goal or step == max_steps:
            break

        if path_index + 1 < len(fixed_path):
            path_index += 1
            robot = fixed_path[path_index]

        trail.append(robot)

        for gurney in gurneys:
            gurney.move()

    return {
        "history": history,
        "collisions": collisions,
        "collision_count": len(collisions),
        "fixed_path": fixed_path,
        "trail": trail,
        "final_robot": robot,
    }


def plot_base_grid(ax, grid_static):
    cmap = colors.ListedColormap(["#2f2f3e", "#000000"])
    ax.imshow(grid_static, cmap=cmap, vmin=0, vmax=1, origin="upper")
    ax.set_xlim(-0.5, 99.5)
    ax.set_ylim(99.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_gurneys(ax, gurneys):
    for gurney in gurneys:
        rect = plt.Rectangle(
            (gurney.col - 1.5, gurney.row - 1.5),
            3,
            3,
            linewidth=0,
            facecolor="#ff9f1c",
            alpha=0.95,
        )
        ax.add_patch(rect)


def draw_path(ax, path, color, linestyle="-", linewidth=1.2, alpha=1.0):
    if len(path) < 2:
        return
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)


def create_animation(grid_static, replanning_result, gif_path):
    ensure_pillow_installed()
    history = replanning_result["history"]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()

    def update(frame_idx):
        frame = history[frame_idx]
        ax.clear()
        ax.set_facecolor("#1a1a2e")

        plot_base_grid(ax, grid_static)
        ax.imshow(
            frame["risk_map"],
            cmap="Reds",
            alpha=0.25,
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )

        draw_path(ax, frame["planned_path"], color="#4ea8ff", linestyle=":", linewidth=1.2)
        draw_path(ax, frame["trail"], color="white", linestyle="-", linewidth=1.0, alpha=0.9)

        draw_gurneys(ax, frame["gurneys"])

        ax.scatter([START[1]], [START[0]], s=60, c="#2ecc71", marker="o", zorder=5)
        ax.scatter([GOAL[1]], [GOAL[0]], s=110, c="#f1c40f", marker="*", zorder=5)
        ax.scatter(
            [frame["robot"][1]],
            [frame["robot"][0]],
            s=70,
            c="white",
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )

        ax.set_title(
            f"MediNav Dynamic Replanning - Step {frame['step']} | Replans: {frame['replans']}",
            color="white",
            fontsize=12,
            pad=10,
        )

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000 / 12, repeat=False)
    writer = PillowWriter(fps=12)
    anim.save(gif_path, writer=writer)
    plt.close(fig)


def make_comparison_figure(grid_static, standard_result, replanning_result, png_path):
    idx_std = min(100, len(standard_result["history"]) - 1)
    idx_rep = min(100, len(replanning_result["history"]) - 1)
    frame_std = standard_result["history"][idx_std]
    frame_rep = replanning_result["history"][idx_rep]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    left = axes[0]
    plot_base_grid(left, grid_static)
    draw_path(left, standard_result["fixed_path"], color="#ff4d4d", linewidth=1.4)
    draw_path(left, frame_std["trail"], color="white", linewidth=1.0, alpha=0.8)
    draw_gurneys(left, frame_std["gurneys"])
    left.scatter([frame_std["robot"][1]], [frame_std["robot"][0]], s=65, c="white", marker="o", zorder=5)

    if standard_result["collisions"]:
        cx = [p[1] for p in standard_result["collisions"]]
        cy = [p[0] for p in standard_result["collisions"]]
        left.scatter(cx, cy, c="#ff2d2d", marker="x", s=50, linewidths=1.2, zorder=6)

    left.set_title("Standard A* - No Replanning", color="white", fontsize=13)
    left.text(
        0.02,
        0.03,
        f"Collisions: {standard_result['collision_count']} (robot cannot adapt)",
        transform=left.transAxes,
        color="white",
        fontsize=10,
        bbox={"facecolor": "#000000", "alpha": 0.45, "pad": 6, "edgecolor": "none"},
    )

    right = axes[1]
    plot_base_grid(right, grid_static)
    draw_path(right, frame_rep["planned_path"], color="#4ea8ff", linestyle=":", linewidth=1.5)
    draw_path(right, frame_rep["trail"], color="white", linewidth=1.0, alpha=0.9)
    draw_gurneys(right, frame_rep["gurneys"])
    right.scatter([frame_rep["robot"][1]], [frame_rep["robot"][0]], s=65, c="white", marker="o", zorder=5)
    right.set_title("Risk-Aware MediNav - Active Replanning", color="white", fontsize=13)
    right.text(
        0.02,
        0.03,
        f"Replans: {replanning_result['number_of_replans']} | Goal reached: {'Yes' if replanning_result['reached_goal'] else 'No'}",
        transform=right.transAxes,
        color="white",
        fontsize=10,
        bbox={"facecolor": "#000000", "alpha": 0.45, "pad": 6, "edgecolor": "none"},
    )

    fig.suptitle(
        "MediNav Dynamic Obstacle Avoidance - Replanning vs Fixed Path",
        color="white",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_static = create_static_grid()

    standard_result = simulate_standard_run(
        grid_static=grid_static,
        start=START,
        goal=GOAL,
        max_steps=MAX_STEPS,
    )

    replanning_result = simulate_replanning_run(
        grid_static=grid_static,
        start=START,
        goal=GOAL,
        lambda_weight=8.0,
        max_steps=MAX_STEPS,
        lookahead_steps=LOOKAHEAD_STEPS,
    )

    create_animation(grid_static, replanning_result, GIF_PATH)
    make_comparison_figure(grid_static, standard_result, replanning_result, PNG_PATH)

    print("=== Dynamic Replanning Results ===")
    print(f"Standard A*: Collisions = {standard_result['collision_count']}")
    print(
        "Risk-Aware A* with replanning: "
        f"Replans = {replanning_result['number_of_replans']}, "
        f"Goal reached = {replanning_result['reached_goal']}"
    )
    print(f"Saved: {GIF_PATH.as_posix()}")
    print(f"Saved: {PNG_PATH.as_posix()}")


if __name__ == "__main__":
    main()
