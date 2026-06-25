# Predictive Maintenance on NASA C-MAPSS (FD001)

Predicting turbofan engine failure from multivariate sensor telemetry. Given a
stream of run-to-failure sensor readings, the model flags whether an engine will
fail within the next **30 cycles** — a binary classification framing of the
classic remaining-useful-life (RUL) problem.

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation, subset **FD001**:

- 100 engines, each run from healthy until failure (~20,600 cycles total)
- 21 sensor channels (temperatures, pressures, fan/core speeds, fuel flow) + 3
  operational settings, recorded once per flight cycle
- Single operating condition, single fault mode (the cleanest subset)

> Note: C-MAPSS is *simulated* turbofan data — industrial-IoT in spirit
> (machine-mounted sensors → predict failure), but the channels are
> temperatures/pressures/speeds, not a literal vibration sensor. The data has no
> missing values and regular sampling, though sensors are noise-contaminated and
> the official train/test split carries a deliberate distribution gap (test
> series are truncated before failure).

Download: [Kaggle — NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
or the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

## Approach

1. **Labeling** — for each engine the last recorded cycle is the failure point;
   `RUL = max_cycle − current_cycle`, then `label = 1 if RUL ≤ 30`.
2. **EDA** — 9-technique pass: lifetime/class balance, univariate distributions,
   class separation via Cohen's d, sensor-vs-RUL signal ranking, hierarchically
   clustered correlations, operating-condition check, degradation curves, fleet
   trajectories, and noise/heteroscedasticity inspection.
3. **Feature engineering** — per-sensor temporal features (short/long rolling
   means, rolling std, rate-of-change, drift-from-start), computed **per engine**
   to prevent cross-engine and look-ahead leakage. ~90 features from 15 useful
   sensors (6 constant sensors dropped by variance threshold).
4. **Evaluation split** — `GroupShuffleSplit` by **engine ID** (70 train / 30
   validation engines, no overlap), so scores reflect generalization to unseen
   equipment rather than memorized trajectories.
5. **Modelling** — gradient boosting vs. RBF SVM, with calibration and
   permutation-importance analysis.

## Results (engine-level validation split)

| Model | ROC-AUC | PR-AUC | Class-1 precision | Class-1 recall |
|---|---|---|---|---|
| Gradient Boosting | 0.994 | 0.970 | 0.897 | 0.906 |
| SVM (RBF) | 0.988 | 0.945 | 0.851 | 0.898 |

Gradient boosting wins on every metric — expected for tabular data with
correlated, different-scale features and strong interactions. Top features are
rolling means of the high-signal sensors, confirming that smoothed trend
features beat raw readings. Baseline positive rate is ~15%, so PR-AUC (not
accuracy) is the headline metric.

> The ~0.99 AUC is the engine-level *validation* split, which is cleaner than
> C-MAPSS's official test set. For a resume-quotable number, evaluate against
> `test_FD001.txt` + `RUL_FD001.txt`.

## Usage

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
# optional, to match the XGBoost variant:
pip install xgboost
```

Set `DATA_DIR` in the Setup cell to the folder containing the `.txt` files:

```python
DATA_DIR = r"C:\path\to\CMAPSSData"   # Windows: keep the r-prefix
```

Then open `predictive_maintenance_cmapss.ipynb` and **Run All**. The notebook
ships with outputs already embedded; re-running regenerates them against your
local data.

## Project structure

```
predictive_maintenance_cmapss.ipynb   # full pipeline: load → EDA → features → models
CMAPSSData/                            # downloaded dataset (train/test/RUL .txt files)
README.md
```

## Notes & extensions

- **Why engine-level split?** A random row split leaks an engine into both train
  and validation, inflating AUC. Splitting by engine tests true generalization.
- **Why scale for SVM but not GBM?** Trees split on thresholds (scale-invariant);
  SVM is distance-based and requires `StandardScaler`.
- **Calibration** — predicted probabilities are roughly calibrated but mildly
  off (partly due to `class_weight="balanced"` distorting the base rate). Fine
  for ranking/thresholding; apply isotonic calibration if using raw probabilities
  for cost decisions.
- **Next steps:** swap in real XGBoost, evaluate on the official test set, add an
  LSTM on raw sequences, and tune the decision threshold toward recall for a
  screening use case.
