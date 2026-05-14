# AlphaTrace Quickstart Guide

Get AlphaTrace up and running in less than 5 minutes.

## 1. Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone https://github.com/Archisman-NC/AlphaTrace.git
cd AlphaTrace

# Install dependencies
pip install -r requirements.txt
```

## 2. Launching the Platform

### Dashboard
The primary research interface.
```bash
python3 run_dashboard.py
```

### Verification
Run the full diagnostic suite to ensure quantitative integrity.
```bash
python3 run_verification.py
```

### RL Research
Execute the isolated PPO training and evaluation pipeline.
```bash
python3 run_research.py
```

## 3. Directory Overview

| Directory | Purpose |
| :--- | :--- |
| `app/` | Production-grade quantitative and reasoning modules. |
| `docs/` | Formal architecture and research methodology. |
| `research/` | Isolated RL sandbox and experimental artifacts. |
| `scratch/` | Deterministic verification and diagnostic scripts. |

## 4. Key Entrypoints

*   `run_dashboard.py`: Interactive Operational Terminal.
*   `run_research.py`: Experimental RL Pipeline.
*   `run_verification.py`: Quantitative Integrity Suite.

---
*For a detailed walkthrough of all features, see [Demo Workflows](docs/demo_workflows.md).*
