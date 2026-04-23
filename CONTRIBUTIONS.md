# Project Contributions — Ojas Sahu

**Name:** Ojas Sahu  
**ID:** 2023A3PS0861G  
**Module:** Risk Assessment & Visualization  
**Project:** MediNav — Autonomous Contactless Medical Supply Delivery Robot

---

## 🚀 Overview
In the MediNav project, I was responsible for the **Risk Module**, which bridges the gap between the SLAM-generated occupancy map and the Path Planner. My work focused on quantifying safety by calculating risk scores for the environment, allowing the robot to prioritize safe corridors over shorter but riskier paths (like hugging walls or equipment).

---

## 🛠️ Key Technical Contributions

### 1. Risk Map Computation (`risk/risk_map.py`)
- Developed the core algorithm to compute a safety-aware risk map from raw occupancy data.
- Implemented **Euclidean Distance Transform (EDT)** to calculate distances from obstacles.
- Designed a custom **Gaussian-based risk function** where risk decays exponentially as the robot moves away from obstacles.
- Added functionality to save the risk map as a `.npy` file for seamless integration with the A* Planner.

### 2. Risk Visualization (`risk/visualize_risk.py`)
- Created a comprehensive visualization suite to inspect the safety layers.
- Implemented a 3-panel plotting system showing:
    - **Raw Occupancy Grid:** The base environment visualization.
    - **Risk Heatmap:** A visual gradient representing high-risk (red) and low-risk (blue) zones.
    - **Alpha-blended Overlay:** A combined view for verifying path safety against physical obstacles.

### 3. Dijkstra & Comparative Analysis (`simulation/evaluate_extended.py`)
- Implemented **Dijkstra’s Algorithm** as a baseline pathfinding benchmark to compare against A* and Risk-Aware A*.
- Developed the **Extended Evaluation Framework**, enabling comparative studies across three different planners.
- Automated **Monte Carlo Simulations** (30+ trials) to statistically validate the safety improvements of the risk-aware approach.
- Created **Statistical Benchmarks**:
    - Calculated "Path Length vs. Dijkstra" overhead metrics.
    - Tracked **Safety Violation Rates** (cells within dangerous proximity to walls).
    - Produced grouped bar charts (`evaluation_extended.png`) for multi-planner performance review.

### 4. Quantitative Analysis (`risk/metrics.py` & `risk/risk_analysis.py`)
- Programmed a metrics engine to provide statistical insights into the map's safety profile.
- Implemented histogram analysis to visualize the distribution of risk values across the environment.
- Added terminal outputs for critical metrics: Min/Max/Mean risk, obstacle proximity stats, and cell distribution.

### 5. Integration & Logic (`slam/occupancy_grid.py`)
- Modified the SLAM module's output logic to ensure the occupancy grid is saved in a format compatible with numerical processing.
- Handled file I/O logic for cross-module data sharing.

### 6. Conflict Resolution & Branch Management
- Managed the `risk` and `dijkstra-comparison` branches.
- Resolved complex **binary merge conflicts** in `.npy` and `.png` outputs during branch synchronization.

---

## 📄 Documentation & Workflow
- **README Updates:** Authored the complete technical documentation for the Risk Module in the main `README.md`, including setup instructions, execution steps, and output descriptions.
- **Git Management:** Managed the `risk` branch, handled conflict resolution during merges with the `main` branch, and ensured consistent commit history.
- **Output Management:** Standardized the storage and naming conventions for generated plots and data files in the `outputs/` directory.

---

## 📂 Files Created/Modified
- `risk/risk_map.py` [NEW]
- `risk/visualize_risk.py` [NEW]
- `risk/metrics.py` [NEW]
- `risk/risk_analysis.py` [NEW]
- `simulation/evaluate_extended.py` [NEW]
- `slam/occupancy_grid.py` [MODIFIED]
- `README.md` [MODIFIED]

---

## 📊 Outputs Generated
- `outputs/risk_visualization.png`
- `outputs/risk_analysis.png`
- `outputs/evaluation_extended.png` (Dijkstra Comparison)
- `outputs/risk_map.npy`
- `outputs/occupancy_grid.npy`
- `outputs/eval_log_extended.txt`

---
*Created by Antigravity AI on behalf of Ojas Sahu.*
