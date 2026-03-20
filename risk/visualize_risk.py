import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists("outputs"):
    raise FileNotFoundError("outputs folder not found")

try:
    p_map = np.load("outputs/occupancy_grid.npy")
    risk_map = np.load("outputs/risk_map.npy")
except Exception as e:
    raise FileNotFoundError("Required .npy files not found. Run SLAM and risk_map first.") from e

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
plt.savefig('outputs/risk_visualization.png', dpi=150)
print("Saved outputs/risk_visualization.png")

print("\nVisualization saved to outputs/risk_visualization.png")
