"""
MediNav — Risk Analysis
risk/risk_analysis.py

Original 3-panel analysis + optional NN vs rule-based comparison plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# 1. ORIGINAL — basic analysis plot (preserved exactly)
# ---------------------------------------------------------------------------

def plot_basic_analysis(risk_map, occupancy=None):
    """
    Create the original 3-panel risk analysis plot.
    """
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
    print("Saved outputs/risk_analysis.png")
    plt.close()


# ---------------------------------------------------------------------------
# 2. NEW — NN vs rule-based comparison
# ---------------------------------------------------------------------------

def plot_nn_comparison(risk_rule, risk_nn):
    """
    Create a 4-panel comparison of rule-based vs NN risk predictions.

    Parameters
    ----------
    risk_rule : np.ndarray (H, W) — rule-based risk map
    risk_nn   : np.ndarray (H, W) — NN-predicted risk map
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("MediNav — Rule-Based vs Neural Network Risk Comparison",
                 fontsize=14, fontweight='bold')

    # Panel 1: Rule-based risk
    im0 = axs[0, 0].imshow(risk_rule, cmap='hot', vmin=0, vmax=1)
    axs[0, 0].set_title("Rule-Based Risk")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    # Panel 2: NN-predicted risk
    im1 = axs[0, 1].imshow(risk_nn, cmap='hot', vmin=0, vmax=1)
    axs[0, 1].set_title("Neural Network Risk")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: Difference heatmap
    diff = np.abs(risk_rule - risk_nn)
    im2 = axs[1, 0].imshow(diff, cmap='coolwarm', vmin=0, vmax=0.5)
    axs[1, 0].set_title(f"Absolute Difference (mean={diff.mean():.4f})")
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Histogram overlay
    axs[1, 1].hist(risk_rule.flatten(), bins=50, alpha=0.6, label='Rule-Based',
                   color='tab:red', edgecolor='none')
    axs[1, 1].hist(risk_nn.flatten(), bins=50, alpha=0.6, label='Neural Network',
                   color='tab:blue', edgecolor='none')
    axs[1, 1].set_title("Risk Distribution Comparison")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("Risk Value")
    axs[1, 1].set_ylabel("Frequency")

    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/risk_analysis_nn.png", dpi=150)
    print("Saved outputs/risk_analysis_nn.png")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    risk_map = np.load("outputs/risk_map.npy")

    try:
        occupancy = np.load("outputs/occupancy_grid.npy")
    except Exception:
        occupancy = None

    # Always run original analysis
    plot_basic_analysis(risk_map, occupancy)

    # NN comparison (optional)
    if USE_NN:
        try:
            risk_rule = np.load("outputs/risk_map_rule.npy")
            risk_nn = np.load("outputs/risk_map_nn.npy")
            print("\n--- Generating NN vs Rule-Based Comparison ---")
            plot_nn_comparison(risk_rule, risk_nn)
        except FileNotFoundError:
            if SAFE_MODE:
                print("  WARNING: NN risk maps not found, skipping NN comparison")
                print("  (Run risk/risk_map.py with USE_NN=True first)")
            else:
                raise
        except Exception as e:
            if SAFE_MODE:
                print(f"  WARNING: NN comparison failed ({e}), skipping")
            else:
                raise
