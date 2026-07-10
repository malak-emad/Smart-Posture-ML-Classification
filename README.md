# Smart Posture — ML Classification (Random Forest)

> Feature-engineered Random Forest pipeline for classifying sitting posture from 4-sensor IMU data, developed as part of the **Smart Posture** graduation project (Faculty of Engineering, Cairo University).

**Author:** Malak Emad
**Team:** Ayat Tarek · Biatriss Benyamin · Malak Emad · Nariman Ahmed · Saif Mohamed
**Supervisor:** Dr. Aliaa Rehan

> This repo covers the **Random Forest** classification pipeline only. It is one of two independently developed and validated models in the Smart Posture project — the second is a CNN trained directly on raw IMU windows (see the main [Smart Posture repo](#) for the full system, including hardware, app, and dataset details).

---

## Overview

This pipeline classifies sitting posture into 6 classes from 4 IMU sensors placed along the spine (C7, T4, T12, L5):

**Upright · Forward Bending · Backward Bending · Left Bending · Right Bending · Slouching**

It uses hand-engineered features (not raw signal) fed into a Random Forest classifier, validated with **Leave-One-Subject-Out (LOSO)** cross-validation to ensure the model generalizes to unseen individuals rather than memorizing subject-specific patterns.

📦 **Dataset:** [IMU-Sitting-Postures-Dataset](https://github.com/nariman-ahmed/IMU-Sitting-Postures-Dataset) (46 participants, CC BY 4.0)

---

## Pipeline

```
Raw IMU CSVs (4 sensors × 13 channels)
        │
        ▼
Hampel filtering (outlier removal)
        │
        ▼
Upright calibration (per-subject median reference, angle channels only)
        │
        ▼
Sliding windows (100 samples, 50% overlap)
        │
        ▼
Feature extraction (8 blocks → 109 features)
        │
        ▼
Feature selection (block-level wrapper → B1 + B4 → 76 features)
        │
        ▼
Random Forest Classifier
        │
        ▼
Posture prediction (6 classes)
```

---

## 1. Preprocessing

- **Outlier removal:** Hampel filter (rolling median ± MAD-based threshold) applied per channel.
- **Upright calibration:** each subject's upright trials are used to compute a per-sensor median reference for the Euler angle channels (Roll/Pitch/Yaw), which is subtracted from all windows for that subject. This removes subject-specific sensor mounting offset rather than relying on a single global mean.
- **Windowing:** 100-sample windows with 50% overlap (50-sample step), computed per subject per trial.

## 2. Feature Engineering — 8 blocks, 109 features

| Block | Description | # Features |
|---|---|---|
| B1 | Per-sensor stats: mean, std, RMS (acc/gyro) | 52 |
| B2 | C7 vs L5 angle difference (mean/std) | 6 |
| B3 | Pitch/roll mean, std, slope per sensor | 24 |
| B4 | Acceleration magnitude (mean/std/max/min) | 16 |
| B5 | Spinal curvature — quadratic fit across 4 sensor pitches | 2 |
| B6 | Spine slope (linear fit across sensors) | 2 |
| B7 | Sign agreement (C7 × L5) | 1 |
| B8 | Slouch discriminators (L5/T12/T4 relative angles, gradients) | ~17 |

An extended feature set (SMA, energy, jerk, cross-sensor correlation, FFT-based frequency/spectral entropy features) was also implemented and explored during development but is not part of the final selected pipeline described below.

## 3. Feature Selection

Two selection strategies were evaluated:

- **Block-level wrapper selection:** each of the 8 feature blocks was evaluated for its contribution to LOSO accuracy; blocks that hurt generalization were dropped. **B1 (per-sensor stats) + B4 (acceleration magnitude)** were retained → **76 features**.
- **RF-importance-based top-K selection:** a Random Forest is fit on the training folds, features ranked by importance, and the top-K evaluated on a held-out split to pick the best K before running full LOSO with per-fold reselection (avoiding data leakage).

Feature importance analysis (averaged across all 45 LOSO folds) showed **acceleration (X/Y/Z)** and **Euler angles (Pitch, Roll)** as the most informative channels.

## 4. Model Configuration

Hyperparameters were tuned via grid search; top result:

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 5 |
| `min_samples_leaf` | 5 |
| `max_features` | sqrt |
| `class_weight` | balanced |
| `bootstrap` | True |

(Note: the final LOSO evaluation script uses a deeper configuration — `n_estimators=1000, max_depth=25` — for the production pipeline; the table above reflects the grid-search-selected config from hyperparameter tuning. Keep these consistent or document the discrepancy if publishing final numbers.)

As a sanity check, **LazyPredict** was used to screen 25 classifiers on the same 109 features and train/test split, with no feature selection. Random Forest and Extra Trees came out on top, ahead of linear/SVM baselines — supporting RF as a reasonable model choice for this feature set rather than an arbitrary pick.

## 5. Validation Strategy — LOSO

- **Leave-One-Subject-Out**: for each of 45 subjects, the model trains on the other 44 and tests on the held-out subject, repeated for all subjects.
- Feature selection is refit **inside each fold** on training data only, to avoid leakage.
- 45 independent test runs, ~23,490 total windows across 6 posture classes.

## 6. Results

| Metric | Value |
|---|---|
| Window-Level Accuracy (LOSO) | 84.1% |
| Subject-Level Accuracy (LOSO) | 85.9% |

Per-class F1 (approximate, from LOSO confusion matrix):

| Class | F1 |
|---|---|
| Backward Bending | 0.98 |
| Upright | 0.89 |
| Left Bending | 0.92 |
| Right Bending | 0.88 |
| Forward Bending | lower — see confusion below |
| Slouching | lower — see confusion below |

**Main confusion pair:** Forward Bending ↔ Slouching. All other classes are near-perfect. This is consistent with the underlying kinematics — both postures involve anterior trunk flexion and are genuinely difficult to separate from spinal IMU data alone.

ROC and Precision-Recall curves (one-vs-rest, computed from LOSO-pooled predicted probabilities) are included in `results/` for a per-class view of separability beyond raw accuracy.

## 7. Known Limitations

- Forward Bending and Slouching remain the hardest pair to distinguish, due to genuinely similar trunk kinematics — this is a data/kinematics limitation, not purely a modeling one.
- Dataset limited to 45 subjects (healthy adults, ages 19–24); generalizability beyond this population is untested.
- The extended feature set (FFT/spectral/jerk/correlation features) was implemented but not part of the final selected pipeline — it's left in the codebase for reference/future exploration, not because it was validated to improve results.
- Hyperparameter configuration used in the final LOSO run differs slightly from the grid-search-selected config; if reporting final numbers, confirm which configuration produced them.

---

## Repository Structure

```
├── src/
│   ├── loso_pipeline.py          # Preprocessing, feature extraction, LOSO training + block/RF-importance selection
│   ├── feature_importance.py     # Per-channel and per-block importance analysis across LOSO folds
│   ├── roc_pr_curves.py          # ROC / Precision-Recall curves from LOSO-pooled probabilities
│   └── lazypredict_baseline.py   # 25-model screening baseline (no feature selection)
├── results/                      # Generated plots and CSVs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Update `DATASET_PATH` in each script to point to your local copy of the [dataset](https://github.com/nariman-ahmed/IMU-Sitting-Postures-Dataset).

```bash
python src/loso_pipeline.py
python src/feature_importance.py
python src/roc_pr_curves.py
python src/lazypredict_baseline.py
```

---

## Acknowledgements

Developed as part of the Smart Posture graduation project, Faculty of Engineering, Cairo University, supervised by **Dr. Aliaa Rehan**, with support from the Information Technology Academia Collaboration (ITAC) program.
