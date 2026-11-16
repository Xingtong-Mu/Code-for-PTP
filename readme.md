# AIS Vessel Trajectory Prediction and Maritime Early Warning

This repository accompanies the manuscript **Retrieval-Conditioned Probabilistic Trajectory Prediction for Maritime Early Warning Using AIS Sensor Data** by **Xingtong Mu, Xueling Duan, and Ke Deng** (School of Mathematics, Sichuan University).

The proposed framework uses dynamic trajectory retrieval to adapt a regionally pretrained turning-angle model to the target vessel’s current observations. It models vessel speed separately, generates an ensemble of future trajectories, and uses the ensemble to construct calibrated spatial warning regions.

## Repository Contents

| File or directory | Description |
| --- | --- |
| `Proposed.py` | Implements the proposed method, including vessel-specific selection of the speed blending schedule. |
| `Compared_*.py` | Implement the baseline forecasting methods and their DTR variants where applicable. |
| `Ablation_*.py` | Implement the component ablations. |
| Regional ZIP archives | Contain the evaluation trajectories, historical reference library, evaluation records, and regional model checkpoints for the Gulf of Mexico and the U.S. Mid-Atlantic. |
| `requirements.txt` | Lists the Python dependencies. |
| `results/` | Contains vessel-level metrics, aggregate results, paired statistical tests, and warning-region calibration results for the held-out evaluation sets. |

## Installation

Use Python 3.10 or later. From the repository root, install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The scripts use CUDA when available and run on the CPU otherwise.

## Running the Experiments

1. Place both regional ZIP archives in the same directory as the Python scripts. The scripts extract the selected archive on first use.
2. In the script you want to run, set `REGION` to `"Gulf"` for the Gulf of Mexico or `"MidAtlantic"` for the U.S. Mid-Atlantic.
3. Set `LIMIT = 1` to check execution on one vessel or `LIMIT = None` to evaluate the full cohort.
4. From the repository root, run the selected script. For example:

   ```bash
   python Proposed.py
   ```

   In Jupyter, run the same script with:

   ```python
   %run Proposed.py
   ```

Each script runs one forecasting method. Predictions and run records from new executions are written to `run_results/`.
