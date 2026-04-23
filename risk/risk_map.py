"""
MediNav — Risk Map Computation
risk/risk_map.py

Computes a safety-aware risk map from the occupancy grid.
Supports:
  - Static wall-proximity risk (original)
  - Dynamic human-aware risk (new, optional)
  - Neural-network blended risk (new, optional)
  - Per-cell feature extraction for NN training
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import os
import sys

# ---------------------------------------------------------------------------
# GLOBAL SAFETY FLAGS
# ---------------------------------------------------------------------------
USE_DYNAMIC_RISK = True
USE_HUMANS = True
USE_REALTIME = True
USE_NN = True
SAFE_MODE = True   # if True, fallback to original logic on failure

# ---------------------------------------------------------------------------
# Constants (original)
# ---------------------------------------------------------------------------
OCC_THRESHOLD = 0.6
EPS = 1e-8

# --- STATIC risk weights (used in compute_static_risk) ---
# w1 = wall-distance risk,  w2 = SLAM pose uncertainty
# These are the weights from the main report (Section 3.x)
DISTANCE_WEIGHT = 0.7     # w1: wall-distance risk
DENSITY_WEIGHT  = 0.3     # w2: SLAM uncertainty
SMOOTHING_SIGMA = 1.5

# --- DYNAMIC risk weights (used in compute_dynamic_risk) ---
# w1 = wall-distance risk,  w2 = human proximity risk
# NOTE: These are DIFFERENT from the static weights above.
#       Same variable names (w1, w2) but different semantics:
#       Static mode:  R = 0.7 * R_wall + 0.3 * R_slam_uncertainty
#       Dynamic mode: R = 0.7 * R_wall + 0.3 * R_human_proximity
WALL_RISK_WEIGHT  = 0.7   # w1: wall-distance risk (same)
HUMAN_RISK_WEIGHT = 0.3   # w2: human proximity risk (replaces SLAM unc.)


# ---------------------------------------------------------------------------
# 1. ORIGINAL — compute_static_risk (preserves exact original behavior)
# ---------------------------------------------------------------------------

def compute_static_risk(p_map, slam_uncertainty=None):
    """
    Compute the original STATIC risk map from an occupancy probability map.

    Risk formula (static mode):
        R = w1 * R_wall_distance + w2 * R_slam_uncertainty
        w1 = 0.7 (DISTANCE_WEIGHT), w2 = 0.3 (DENSITY_WEIGHT)

    Parameters
    ----------
    p_map             : np.ndarray (H, W) — occupancy probability grid
    slam_uncertainty  : np.ndarray (H, W) or None — SLAM pose uncertainty

    Returns
    -------
    risk_map : np.ndarray (H, W) — normalized risk values in [0, 1]
    distance : np.ndarray (H, W) — EDT distance from obstacles
    """
    if len(p_map.shape) != 2:
        raise ValueError("Occupancy grid must be 2D")

    if slam_uncertainty is None:
        slam_uncertainty = np.ones_like(p_map) * 0.3

    if slam_uncertainty.shape != p_map.shape:
        slam_uncertainty = np.ones_like(p_map) * 0.3

    occupancy = (p_map > OCC_THRESHOLD).astype(int)
    free_space = 1 - occupancy
    distance = distance_transform_edt(free_space)

    distance_risk = np.exp(-distance)
    distance_risk = (distance_risk - distance_risk.min()) / (distance_risk.max() - distance_risk.min() + EPS)

    # R_unc comes from actual SLAM pose covariance (exported by graphslam.py)
    uncertainty_risk = slam_uncertainty.copy()
    uncertainty_risk = (uncertainty_risk - uncertainty_risk.min()) / (
        uncertainty_risk.max() - uncertainty_risk.min() + EPS
    )

    risk_map = DISTANCE_WEIGHT * distance_risk + DENSITY_WEIGHT * uncertainty_risk
    print("R_unc now sourced from SLAM pose covariance map")

    risk_map = gaussian_filter(risk_map, sigma=SMOOTHING_SIGMA)
    risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + EPS)

    return risk_map, distance


# ---------------------------------------------------------------------------
# 2. NEW — compute_human_risk
# ---------------------------------------------------------------------------

def compute_human_risk(grid_shape, humans):
    """
    Compute a risk field based on proximity to humans.

    Parameters
    ----------
    grid_shape : tuple (H, W)
    humans     : list of (x, y) tuples — human positions in grid coords

    Returns
    -------
    human_risk : np.ndarray (H, W) — normalized human-proximity risk in [0, 1]
    human_dist : np.ndarray (H, W) — minimum distance to any human
    """
    H, W = grid_shape
    human_risk = np.zeros((H, W), dtype=float)

    if not humans or len(humans) == 0:
        return human_risk, np.full((H, W), np.inf)

    # Create a binary map with 1 at human positions
    human_mask = np.ones((H, W), dtype=float)
    for (hx, hy) in humans:
        r, c = int(np.clip(hx, 0, H - 1)), int(np.clip(hy, 0, W - 1))
        human_mask[r, c] = 0

    # Distance from each cell to nearest human
    human_dist = distance_transform_edt(human_mask)

    # Risk: exponential decay from humans
    human_risk = np.exp(-human_dist * 0.3)
    human_risk = (human_risk - human_risk.min()) / (human_risk.max() - human_risk.min() + EPS)

    return human_risk, human_dist


# ---------------------------------------------------------------------------
# 3. NEW — compute_dynamic_risk
# ---------------------------------------------------------------------------

def compute_dynamic_risk(p_map, slam_uncertainty=None, humans=None, t=0):
    """
    Compute a combined DYNAMIC risk map: wall risk + human proximity risk.

    Risk formula (dynamic mode):
        R = w1 * R_wall + w2 * R_human_proximity
        w1 = 0.7 (WALL_RISK_WEIGHT), w2 = 0.3 (HUMAN_RISK_WEIGHT)

    NOTE: w1/w2 have DIFFERENT semantics than in compute_static_risk():
        Static:  w2 = SLAM uncertainty
        Dynamic: w2 = human proximity risk

    Parameters
    ----------
    p_map             : np.ndarray (H, W) — occupancy probability grid
    slam_uncertainty  : np.ndarray (H, W) or None
    humans            : list of (x, y) tuples
    t                 : float — current time step (for future time-varying effects)

    Returns
    -------
    risk_map   : np.ndarray (H, W) — combined dynamic risk in [0, 1]
    wall_dist  : np.ndarray (H, W)
    human_dist : np.ndarray (H, W)
    """
    # Wall-based risk (original)
    wall_risk, wall_dist = compute_static_risk(p_map, slam_uncertainty)

    if humans and len(humans) > 0:
        human_risk, human_dist = compute_human_risk(p_map.shape, humans)
        risk_map = WALL_RISK_WEIGHT * wall_risk + HUMAN_RISK_WEIGHT * human_risk
    else:
        risk_map = wall_risk
        human_dist = np.full_like(wall_dist, np.inf)

    # Normalize
    risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + EPS)

    return risk_map, wall_dist, human_dist


# ---------------------------------------------------------------------------
# 4. NEW — extract_features (for NN training)
# ---------------------------------------------------------------------------

def extract_features(distance_map, human_dist_map=None):
    """
    Extract per-cell features for the neural network.

    Parameters
    ----------
    distance_map   : np.ndarray (H, W) — distance to nearest wall
    human_dist_map : np.ndarray (H, W) or None — distance to nearest human

    Returns
    -------
    features : np.ndarray (H*W, 2) — [distance_to_wall, distance_to_human]
    """
    H, W = distance_map.shape
    wall_feat = distance_map.flatten()

    if human_dist_map is not None:
        human_feat = human_dist_map.flatten()
    else:
        human_feat = np.full(H * W, 15.0)  # default "far from humans"

    features = np.column_stack([wall_feat, human_feat])
    return features


# ---------------------------------------------------------------------------
# 5. NEW — NN-blended risk computation
# ---------------------------------------------------------------------------

def compute_nn_risk(features, risk_rule, grid_shape):
    """
    Blend rule-based risk with neural network predictions.

    Parameters
    ----------
    features   : np.ndarray (N, 2) — per-cell features
    risk_rule  : np.ndarray (H, W) — rule-based risk map
    grid_shape : tuple (H, W)

    Returns
    -------
    risk_blended : np.ndarray (H, W) — 50/50 blend of rule + NN
    risk_nn_map  : np.ndarray (H, W) — pure NN predictions (for analysis)
    """
    try:
        if USE_NN:
            # Handle import whether running as module or script
            try:
                from risk.risk_nn import RiskNN
            except ImportError:
                sys.path.insert(0, os.path.dirname(__file__))
                from risk_nn import RiskNN

            model = RiskNN(input_dim=2, hidden_dim=16)

            # Train on rule-based labels
            labels = risk_rule.flatten()
            print("\n--- Training Neural Network Risk Model ---")
            model.train_model(features, labels, epochs=500, lr=0.01, verbose=True)

            # Predict
            nn_preds = model.predict(features)
            risk_nn_map = nn_preds.reshape(grid_shape)

            # Save model
            model.save_model("outputs/risk_nn_weights.npz")

            # Blend: 50% rule + 50% NN
            risk_blended = 0.5 * risk_rule + 0.5 * risk_nn_map
            risk_blended = (risk_blended - risk_blended.min()) / (
                risk_blended.max() - risk_blended.min() + EPS
            )

            print("  NN risk blended with rule-based risk (50/50)")
            return risk_blended, risk_nn_map
        else:
            return risk_rule, risk_rule
    except Exception as e:
        if SAFE_MODE:
            print(f"  WARNING: NN failed ({e}), falling back to rule-based risk")
            return risk_rule, risk_rule
        else:
            raise


# ---------------------------------------------------------------------------
# MAIN — script entry point (preserves original behavior + new features)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Load occupancy grid (original)
    try:
        p_map = np.load("outputs/occupancy_grid.npy")
    except Exception as e:
        raise FileNotFoundError("occupancy_grid.npy not found. Run SLAM first.") from e

    # Load SLAM uncertainty (original)
    try:
        slam_uncertainty = np.load("outputs/slam_uncertainty.npy")
        if slam_uncertainty.shape != p_map.shape:
            slam_uncertainty = np.ones_like(p_map) * 0.3
        print("Loaded SLAM uncertainty map from outputs/slam_uncertainty.npy")
    except FileNotFoundError:
        print("WARNING: slam_uncertainty.npy not found, using uniform uncertainty=0.3")
        slam_uncertainty = np.ones_like(p_map) * 0.3

    # ------ Compute risk (static or dynamic) ------

    if USE_DYNAMIC_RISK and USE_HUMANS:
        # Generate sample humans for standalone run
        np.random.seed(42)
        H, W = p_map.shape
        occupancy = (p_map > OCC_THRESHOLD).astype(int)
        free_cells = np.argwhere(occupancy == 0)
        if len(free_cells) > 5:
            idxs = np.random.choice(len(free_cells), size=5, replace=False)
            humans = [tuple(free_cells[i]) for i in idxs]
        else:
            humans = []

        print(f"\n--- Dynamic Risk Mode (humans={len(humans)}) ---")
        risk_map, wall_dist, human_dist = compute_dynamic_risk(
            p_map, slam_uncertainty, humans
        )

        # Extract features for NN
        features = extract_features(wall_dist, human_dist)

    else:
        print("\n--- Static Risk Mode (original) ---")
        risk_map, wall_dist = compute_static_risk(p_map, slam_uncertainty)
        human_dist = None
        features = extract_features(wall_dist)

    # Save rule-based risk for comparison
    risk_rule = risk_map.copy()
    np.save("outputs/risk_map_rule.npy", risk_rule)

    # ------ NN blending (optional) ------

    if USE_NN:
        risk_map, risk_nn_map = compute_nn_risk(features, risk_map, p_map.shape)
        np.save("outputs/risk_map_nn.npy", risk_nn_map)
        print("  Saved NN risk map -> outputs/risk_map_nn.npy")
    else:
        risk_nn_map = risk_map

    # ------ Save final risk map (same filename as original) ------

    np.save("outputs/risk_map.npy", risk_map)

    print("\n--- Risk Map Statistics ---")
    print(f"Shape: {risk_map.shape}")
    print(f"Min Risk: {risk_map.min():.6f}")
    print(f"Max Risk: {risk_map.max():.6f}")
    print(f"Mean Risk: {risk_map.mean():.6f}")
    print(f"Std Dev: {risk_map.std():.6f}")

    high_risk_ratio = np.sum(risk_map > 0.7) / risk_map.size
    print(f"High Risk Area Ratio (>0.7): {high_risk_ratio:.4f}")

    print("\nRisk map saved to outputs/risk_map.npy")
