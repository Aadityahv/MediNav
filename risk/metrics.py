import numpy as np

risk_map = np.load("outputs/risk_map.npy")

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
