import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import os

OCC_THRESHOLD = 0.6
DISTANCE_WEIGHT = 0.7
DENSITY_WEIGHT = 0.3
SMOOTHING_SIGMA = 1.5
EPS = 1e-8

if not os.path.exists("outputs"):
    os.makedirs("outputs")

try:
    p_map = np.load("outputs/occupancy_grid.npy")
except Exception as e:
    raise FileNotFoundError("occupancy_grid.npy not found. Run SLAM first.") from e

try:
    slam_uncertainty = np.load("outputs/slam_uncertainty.npy")
    if slam_uncertainty.shape != p_map.shape:
        slam_uncertainty = np.ones_like(p_map) * 0.3  # fallback if shape mismatch
    print("Loaded SLAM uncertainty map from outputs/slam_uncertainty.npy")
except FileNotFoundError:
    print("WARNING: slam_uncertainty.npy not found, using uniform uncertainty=0.3")
    slam_uncertainty = np.ones_like(p_map) * 0.3

if len(p_map.shape) != 2:
    raise ValueError("Occupancy grid must be 2D")

occupancy = (p_map > OCC_THRESHOLD).astype(int)

free_space = 1 - occupancy
distance = distance_transform_edt(free_space)

distance_risk = np.exp(-distance)
distance_risk = (distance_risk - distance_risk.min()) / (distance_risk.max() - distance_risk.min() + EPS)

# R_unc comes from actual SLAM pose covariance (exported by graphslam.py)
uncertainty_risk = slam_uncertainty.copy()
uncertainty_risk = (uncertainty_risk - uncertainty_risk.min()) / (
    uncertainty_risk.max() - uncertainty_risk.min() + 1e-8
)

risk_map = DISTANCE_WEIGHT * distance_risk + DENSITY_WEIGHT * uncertainty_risk
print("R_unc now sourced from SLAM pose covariance map")

risk_map = gaussian_filter(risk_map, sigma=SMOOTHING_SIGMA)

risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + EPS)

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
