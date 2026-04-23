"""
MediNav — Risk Metrics
risk/metrics.py

Computes basic and advanced risk statistics.
Original metrics preserved. New path-based metrics added optionally.
"""

import numpy as np
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
# 1. ORIGINAL — basic metrics (preserved exactly)
# ---------------------------------------------------------------------------

def compute_basic_metrics(risk_map):
    """
    Compute and print basic risk map statistics.
    This is the original metrics logic.
    """
    mean_risk = np.mean(risk_map)
    max_risk = np.max(risk_map)
    min_risk = np.min(risk_map)
    std_risk = np.std(risk_map)

    high_risk_ratio = np.sum(risk_map > 0.7) / risk_map.size
    medium_risk_ratio = np.sum((risk_map > 0.3) & (risk_map <= 0.7)) / risk_map.size
    low_risk_ratio = np.sum(risk_map <= 0.3) / risk_map.size

    print("\n--- Risk Metrics ---")
    print(f"Min Risk: {min_risk:.6f}")
    print(f"Max Risk: {max_risk:.6f}")
    print(f"Mean Risk: {mean_risk:.6f}")
    print(f"Std Dev: {std_risk:.6f}")

    print("\n--- Risk Distribution ---")
    print(f"Low Risk (<=0.3): {low_risk_ratio:.4f}")
    print(f"Medium Risk (0.3–0.7): {medium_risk_ratio:.4f}")
    print(f"High Risk (>0.7): {high_risk_ratio:.4f}")

    return {
        "min": min_risk, "max": max_risk, "mean": mean_risk, "std": std_risk,
        "high_ratio": high_risk_ratio, "medium_ratio": medium_risk_ratio,
        "low_ratio": low_risk_ratio
    }


# ---------------------------------------------------------------------------
# 2. NEW — advanced path-based metrics
# ---------------------------------------------------------------------------

def compute_advanced_metrics(risk_map, path=None):
    """
    Compute advanced risk metrics, optionally along a specific path.

    Parameters
    ----------
    risk_map : np.ndarray (H, W) — risk values in [0, 1]
    path     : list of (row, col) tuples or None

    Returns
    -------
    dict with advanced metrics
    """
    metrics = {}

    if path is not None and len(path) > 0:
        # Risk values along the path
        path_risks = np.array([risk_map[r, c] for r, c in path])

        metrics["path_avg_risk"] = float(np.mean(path_risks))
        metrics["path_max_risk"] = float(np.max(path_risks))
        metrics["path_min_risk"] = float(np.min(path_risks))

        # Time (cells) in high-risk zones
        high_risk_cells = int(np.sum(path_risks > 0.7))
        medium_risk_cells = int(np.sum((path_risks > 0.3) & (path_risks <= 0.7)))
        low_risk_cells = int(np.sum(path_risks <= 0.3))

        metrics["high_risk_cells"] = high_risk_cells
        metrics["medium_risk_cells"] = medium_risk_cells
        metrics["low_risk_cells"] = low_risk_cells
        metrics["high_risk_pct"] = high_risk_cells / len(path) * 100
        metrics["medium_risk_pct"] = medium_risk_cells / len(path) * 100
        metrics["low_risk_pct"] = low_risk_cells / len(path) * 100

        print("\n--- Advanced Path Metrics ---")
        print(f"Path length         : {len(path)} cells")
        print(f"Average risk        : {metrics['path_avg_risk']:.4f}")
        print(f"Max risk on path    : {metrics['path_max_risk']:.4f}")
        print(f"Min risk on path    : {metrics['path_min_risk']:.4f}")
        print(f"High-risk cells     : {high_risk_cells} ({metrics['high_risk_pct']:.1f}%)")
        print(f"Medium-risk cells   : {medium_risk_cells} ({metrics['medium_risk_pct']:.1f}%)")
        print(f"Low-risk cells      : {low_risk_cells} ({metrics['low_risk_pct']:.1f}%)")
    else:
        print("\n--- Advanced Metrics: No path provided, skipping path-based analysis ---")

    # Global advanced: percentile analysis
    p90 = float(np.percentile(risk_map, 90))
    p95 = float(np.percentile(risk_map, 95))
    p99 = float(np.percentile(risk_map, 99))
    metrics["p90"] = p90
    metrics["p95"] = p95
    metrics["p99"] = p99

    print(f"\n--- Risk Percentiles (global) ---")
    print(f"90th percentile     : {p90:.4f}")
    print(f"95th percentile     : {p95:.4f}")
    print(f"99th percentile     : {p99:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    risk_map = np.load("outputs/risk_map.npy")

    # Original metrics (always run)
    compute_basic_metrics(risk_map)

    # Advanced metrics
    # Try to load a saved path if available
    path = None
    try:
        path_data = np.load("outputs/latest_path.npy")
        path = [tuple(p) for p in path_data]
        print(f"\n  Loaded path from outputs/latest_path.npy ({len(path)} cells)")
    except FileNotFoundError:
        print("\n  No saved path found (outputs/latest_path.npy), running global-only analysis")

    compute_advanced_metrics(risk_map, path=path)
