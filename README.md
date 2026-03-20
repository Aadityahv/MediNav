# MediNav 🏥🤖
### Autonomous Contactless Medical Supply Delivery Robot Using Risk-Aware SLAM

**Course:** BITS F327 — Artificial Intelligence for Robotics  
**Institution:** BITS Pilani, Goa Campus  
**Supervisor:** Dr. Prasad Vinayak Patil  
**Submission Deadline:** 22 April 2026

---

## Team

| Name | ID | Module | Branch |
|------|----|--------|--------|
| Aaditya H Vernenker | 2023A7PS1027G | SLAM (GraphSLAM + Occupancy Grid) | `slam` |
| Ojas Sahu | 2023A3PS0861G | Risk Map + Visualization | `risk` |
| Ayush Chopra | 2023A7PS1018G | A* Planner + Path Comparison | `planner` |
| Hansel Cisil Sunny | 2023A4PS0271G | Synthetic Environment + Evaluation | `simulation` |

---

## Project Overview

MediNav is a Python-based simulation of a risk-aware autonomous navigation system for hospital delivery robots. Instead of finding the *shortest* path, MediNav finds the *safest* path — staying away from walls, equipment, and areas where the robot might lose track of its position.

The system has four components that work together:

```
Sensor Data → [SLAM] → Map → [Risk Module] → Risk Map → [A* Planner] → Safe Path → Robot Moves
```

1. **SLAM** builds a map of the hospital while tracking the robot's position
2. **Risk Module** assigns danger scores to areas of the map (high near walls, low in open corridors)
3. **A\* Planner** finds a path that minimises both distance *and* risk
4. **Evaluation** runs both planners 20 times and compares safety statistics

---

## Folder Structure

```
MediNav/
├── README.md               ← this file
├── requirements.txt        ← Python dependencies
├── outputs/                ← all generated plots/images go here
├── slam/                   ← Aaditya's module
│   ├── graphslam.py
│   └── occupancy_grid.py
├── risk/                   ← Ojas's module
│   ├── risk_map.py
│   └── visualize_risk.py
├── planner/                ← Hansel's module
│   ├── astar_risk.py
│   └── path_compare.py
└── simulation/             ← Ayush's module
    ├── synthetic_env.py
    └── evaluate.py
```

---

## Setup — Run This Once

```bash
# Install dependencies (run this once on your machine)
pip install numpy matplotlib scipy
```

No ROS. No Gazebo. Everything runs as plain Python.

---

## How to Run

```bash
# Each script is independent — run them in any order
python slam/graphslam.py          # generates outputs/slam_trajectory.png
python slam/occupancy_grid.py     # generates outputs/occupancy_grid.png
python risk/risk_map.py           # generates outputs/risk_map.npy
python risk/visualize_risk.py     # generates outputs/risk_visualization.png
python planner/astar_risk.py      # prints path length comparison
python planner/path_compare.py    # generates outputs/path_comparison.png
python simulation/evaluate.py     # prints stats table + outputs/evaluation_results.png
```

---

---

# 📋 GitHub Guide for Team Members

**Read this entire section carefully before touching any code.**

---

## The Branch System

Think of the repo like a Google Doc with version history, but safer.

- **`main`** = the clean, working version. **You never commit directly to main.**
- **Your branch** = your personal workspace. You do all your work here.
- When your code works, Aaditya reviews it and **merges** it into main.

```
main ──────────────────────────────────── (clean, working code)
       ↑ merge         ↑ merge
slam ──────────        risk ─────────────  (each person's workspace)
```

---

## Step-by-Step: First Time Setup (everyone does this once)

### 1. Install Git
- **Windows:** Download from https://git-scm.com/download/win → install with all defaults → restart your computer after
- **Mac:** Open Terminal, type `git --version` → if not installed, it will prompt you to install
- **Linux:** `sudo apt install git`

---

### 2. Set your identity (one time only)
Open a terminal and run:
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your_github_email@gmail.com"
```
Use the same email you signed up to GitHub with.

---

### 3. Authenticate with GitHub — CRITICAL, do this or push will fail

GitHub no longer accepts your account password when pushing from terminal. You need a Personal Access Token. Without this you will get an "Authentication failed" error.

**Steps to get your token:**

1. Go to **github.com** → sign in → click your **profile picture** (top right) → **Settings**
2. Scroll all the way down the left sidebar → click **Developer settings**
3. Click **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
4. Fill in:
   - Note: `MediNav`
   - Expiration: **90 days**
   - Tick: ✅ **repo** (the very first checkbox — this automatically ticks all sub-options under it)
5. Scroll down → click **Generate token**
6. **Copy the token immediately** — it looks like `ghp_xxxxxxxxxxxxxxxxxxxx` — you will NEVER see it again after closing the page

**When Git asks for your password during `git push` → paste this token instead of your actual password.**

> 💡 **Windows tip:** If Windows silently fails without asking for a password, open **Control Panel → Credential Manager → Windows Credentials → find github.com → click Edit** and paste the token as the password field.

---

### 4. Clone the repository
This downloads the project to your computer. Run this in your terminal:
```bash
git clone https://github.com/Aadityahv/MediNav.git
cd MediNav
```

Verify it worked — you should see these folders:
```bash
ls
# Expected output: slam/  risk/  planner/  simulation/  outputs/  README.md  requirements.txt
```

---

### 5. Create your personal branch

Each person has their own branch. Use exactly the branch name from the table at the top.

```bash
git checkout -b risk        # Ojas runs this
git checkout -b planner     # Ayush runs this
git checkout -b simulation  # Hansel runs this
```

Confirm it worked:
```bash
git branch
# You should see your branch name with a * next to it, e.g.:
#   main
# * risk
```

---

## Day-to-Day Workflow (do this every time you work on the project)

### Before you start coding — pull latest changes
```bash
git checkout risk           # replace 'risk' with your branch name
git pull origin main        # get any updates Aaditya merged into main
```

### After you write/edit code — save your progress
```bash
git status                  # see what files you changed
git add .                   # stage all changed files
git commit -m "describe what you did"
git push origin risk        # replace 'risk' with your branch name
```

**Good commit messages** (be specific):
```bash
git commit -m "Add risk heatmap generation using distance transform"
git commit -m "Fix path comparison plot — paths now visually distinct"
git commit -m "Add evaluation loop with 20 Monte Carlo trials"
```

**Bad commit messages** (useless):
```bash
git commit -m "stuff"
git commit -m "fixed"
git commit -m "update"
```

---

## What Each Person Needs to Do

---

### 🧠 Ojas Sahu — `risk` branch

**Your files:** `risk/risk_map.py` and `risk/visualize_risk.py`

**Your output:** `outputs/risk_visualization.png` — a 3-panel figure showing occupancy grid, risk heatmap, and overlay.

**Steps:**
```bash
# 1. After cloning, create your branch
git checkout -b risk

# 2. Write your two Python scripts and save them into the risk/ folder in VS Code

# 3. Test that they actually run without errors
python risk/risk_map.py
python risk/visualize_risk.py

# 4. Check you are on the right branch before committing
git branch
# Should show: * risk

# 5. Save and upload your work
git add risk/
git commit -m "Add risk map computation and 3-panel visualization"
git push origin risk

# 6. Message Aaditya on WhatsApp: "risk branch ready to merge"
```

**What your scripts must produce:**
- `risk_map.py` → prints min/max/mean risk values to terminal, saves `outputs/risk_map.npy`
- `visualize_risk.py` → saves `outputs/risk_visualization.png` (3 subplots: occupancy grid | risk heatmap | overlay)

---

### 🗺️ Hansel Cisil — `planner` branch

**Your files:** `planner/astar_risk.py` and `planner/path_compare.py`

**Your output:** `outputs/path_comparison.png` — side-by-side showing standard A* (red, hugs walls) vs risk-aware A* (blue, stays in corridor centre).

**Steps:**
```bash
# 1. After cloning, create your branch
git checkout -b planner

# 2. Write your two Python scripts and save them into the planner/ folder in VS Code

# 3. Test that they run
python planner/astar_risk.py
python planner/path_compare.py

# 4. Confirm branch before committing
git branch
# Should show: * planner

# 5. Save and upload
git add planner/
git commit -m "Add risk-aware A* planner and path comparison visualization"
git push origin planner

# 6. Message Aaditya: "planner branch ready to merge"
```

**What your scripts must produce:**
- `astar_risk.py` → prints path lengths for both planners on a test grid
- `path_compare.py` → saves `outputs/path_comparison.png` showing two visually distinct paths

---

### 📊 Ayush choppedra — `simulation` branch

**Your files:** `simulation/synthetic_env.py` and `simulation/evaluate.py`

**Your output:** A printed statistics table in terminal + `outputs/evaluation_results.png`

**Steps:**
```bash
# 1. After cloning, create your branch
git checkout -b simulation

# 2. Write your two Python scripts and save them into the simulation/ folder in VS Code

# 3. Test that they run
python simulation/synthetic_env.py
python simulation/evaluate.py

# 4. Confirm branch before committing
git branch
# Should show: * simulation

# 5. Save and upload
git add simulation/
git commit -m "Add hospital environment generator and Monte Carlo evaluation"
git push origin simulation

# 6. Message Aaditya: "simulation branch ready to merge"
```

**What your scripts must produce:**
- `synthetic_env.py` → prints grid summary (number of free cells, landmark positions)
- `evaluate.py` → prints a formatted stats table comparing both planners, saves `outputs/evaluation_results.png`

---

## For Aaditya — Merging Branches into Main

When a team member messages you that their branch is ready, run these commands:

```bash
# Example: merging Ojas's risk branch
git checkout main
git merge risk
git push origin main

# Then switch back to your own branch
git checkout slam
```

Repeat for each person, replacing `risk` with `planner` or `simulation`.

---

## Deadlines

| Date | Who | What |
|------|-----|------|
| 18 Mar (today) | Aaditya | Push folder structure + slam/ to main ✅ |
| 19 Mar (Wed) | Ojas | `risk` branch pushed and messaged to Aaditya |
| 19 Mar (Wed) | Hansel | `simulation` branch pushed and messaged to Aaditya |
| 20 Mar (Thu) | Ayush | `planner` branch pushed and messaged to Aaditya |
| 20 Mar (Thu) | Aaditya | Merge all branches into main |
| 21 Mar (Fri) | Aaditya | Add output images to report |
| 22 Mar 23:59 | Aaditya | Final report submitted |

---

## Common Problems & Fixes

**"Authentication failed when pushing"**
```bash
# You need a Personal Access Token — see Step 3 above.
# On Windows, clear the old saved password:
# Control Panel → Credential Manager → Windows Credentials → github.com → Edit → paste token
```

**"I accidentally committed to main"**
```bash
git checkout -b yourbranchname   # save your work to a new branch first
git checkout main
git reset --hard HEAD~1          # undo the last commit on main
git push origin main --force
```
Message Aaditya immediately after.

**"My push was rejected"**
```bash
git pull origin risk             # replace 'risk' with your branch name
# resolve any conflicts if shown, then:
git push origin risk
```

**"I don't know which branch I'm on"**
```bash
git branch                       # branch with * is your current one
git status                       # also shows branch name at the top
```

**"I want to see what I changed before committing"**
```bash
git diff                         # shows line-by-line changes
git status                       # shows list of changed files
```

**"I cloned the repo but my branch doesn't exist yet"**
```bash
git checkout -b risk             # creates it fresh — this is expected first time
```

---

## Questions?

Message Aaditya on WhatsApp before doing anything you are unsure about. Do not guess with Git — it is always easier to ask first than to fix a broken repo later.
