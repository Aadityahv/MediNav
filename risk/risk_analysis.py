import numpy as np
import matplotlib.pyplot as plt

risk_map = np.load("outputs/risk_map.npy")

try:
    occupancy = np.load("outputs/occupancy_grid.npy")
except:
    occupancy = None

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(risk_map, cmap='hot')
axs[0].set_title("Risk Heatmap")

axs[1].hist(risk_map.flatten(), bins=50)
axs[1].set_title("Risk Distribution")

if occupancy is not None:
    axs[2].imshow(occupancy, cmap='gray')
    axs[2].imshow(risk_map, cmap='hot', alpha=0.6)
    axs[2].set_title("Overlay")
else:
    axs[2].axis('off')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.savefig("outputs/risk_analysis.png", dpi=150)
plt.show()
