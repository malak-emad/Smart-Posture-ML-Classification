import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier

DATASET_PATH = r"C:\Users\DELL\Desktop\GP\Aquisition\Labelled_trials_FIXED"

POSTURE_MAP = {
    "1": "backward_bending",
    "2": "upright",
    "3": "slouching",
    "4": "forward_bending",
    "5": "right_bending",
    "6": "left_bending"
}

FEATURE_COLS = [
    "Acceleration X(g)", "Acceleration Y(g)", "Acceleration Z(g)",
    "Angular velocity X(°/s)", "Angular velocity Y(°/s)", "Angular velocity Z(°/s)",
    "Quaternions 0()", "Quaternions 1()", "Quaternions 2()", "Quaternions 3()",
    "Angle X(°)", "Angle Y(°)", "Angle Z(°)"
]

WINDOW_SIZE = 100
WINDOW_STEP = 50

TRAIN_SUBJECTS = {18,14,44,16,10,17,39,33,28,1,32,31,6,12,35,2,42,22,3,36,24,38,11,23,19,46,21,8,15,30,40}
TEST_SUBJECTS  = {4,25,41,34,43,9,27}


def hampel_filter(df, window_size=10, n_sigmas=3):
    df_filtered = df.copy().astype(float)
    k = 1.4826
    for col in df_filtered.columns:
        series         = df_filtered[col]
        rolling_median = series.rolling(window=2*window_size+1, center=True, min_periods=1).median()
        rolling_mad    = (series - rolling_median).abs().rolling(window=2*window_size+1, center=True, min_periods=1).median()
        threshold      = n_sigmas * k * rolling_mad
        outliers       = (series - rolling_median).abs() > threshold
        df_filtered.loc[outliers, col] = rolling_median[outliers]
    return df_filtered

def get_subject_number(file_name):
    return int(os.path.splitext(file_name)[0].split('_')[1])

def create_raw_windows(df, upright_reference=None):
    """
    Returns raw flattened windows — NO feature extraction.
    Each window is (100 × 4 × 13) flattened to 5200 values.
    """
    n_sensors  = 4
    n_features = len(FEATURE_COLS)
    data        = df.to_numpy()
    usable_rows = (data.shape[0] // n_sensors) * n_sensors
    data        = data[:usable_rows]
    reshaped    = data.reshape(-1, n_sensors, n_features)

    if upright_reference is not None:
        reshaped[:, :, 10:13] -= upright_reference
    else:
        mean_angles = np.mean(reshaped[:, :, 10:13], axis=0)
        reshaped[:, :, 10:13] -= mean_angles

    windows = []
    for start in range(0, reshaped.shape[0] - WINDOW_SIZE + 1, WINDOW_STEP):
        window = reshaped[start:start + WINDOW_SIZE]  # (100, 4, 13)
        windows.append(window.flatten())              # → 5200 values, no extraction
    return windows


# ── Load data ─────────────────────────────────────────────────────────────
X_all, y_all, subject_ids_all = [], [], []
removed_subjects = {29}

print("Loading raw windowed data (no feature extraction)...")

upright_refs_raw = {}
upright_folder   = None
for folder in os.listdir(DATASET_PATH):
    if POSTURE_MAP.get(folder) == 'upright':
        upright_folder = os.path.join(DATASET_PATH, folder)
        break

if upright_folder:
    for file_name in sorted(os.listdir(upright_folder)):
        if not file_name.endswith(".csv"): continue
        subject = get_subject_number(file_name)
        if subject in removed_subjects: continue
        df = pd.read_csv(os.path.join(upright_folder, file_name), encoding='utf-8-sig')
        df = df[FEATURE_COLS].copy()
        df = hampel_filter(df)
        data     = df.to_numpy()
        usable   = (data.shape[0] // 4) * 4
        reshaped = data[:usable].reshape(-1, 4, len(FEATURE_COLS))
        trial_ref = np.mean(reshaped[:, :, 10:13], axis=0)
        if subject not in upright_refs_raw:
            upright_refs_raw[subject] = []
        upright_refs_raw[subject].append(trial_ref)

upright_refs = {
    subj: np.median(np.array(trials), axis=0)
    for subj, trials in upright_refs_raw.items()
}

for folder in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path): continue
    label = POSTURE_MAP.get(folder)
    if label is None: continue
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".csv"): continue
        subject = get_subject_number(file_name)
        if subject in removed_subjects: continue
        df  = pd.read_csv(os.path.join(folder_path, file_name), encoding='utf-8-sig')
        df  = df[FEATURE_COLS].copy()
        df  = hampel_filter(df)
        ref = upright_refs.get(subject, None)
        windows = create_raw_windows(df, upright_reference=ref)
        X_all.extend(windows)
        y_all.extend([label] * len(windows))
        subject_ids_all.extend([subject] * len(windows))

X_all           = np.array(X_all, dtype=np.float32)
y_all           = np.array(y_all)
subject_ids_all = np.array(subject_ids_all)

for col in range(X_all.shape[1]):
    mask = ~np.isfinite(X_all[:, col])
    if mask.any():
        X_all[mask, col] = np.nanmedian(X_all[:, col])

print(f"Total windows : {len(X_all)}")
print(f"Feature vector: {X_all.shape[1]} (= 100 timestamps × 4 sensors × 13 channels, flattened)")

# ── Train/Test split ──────────────────────────────────────────────────────
train_mask = np.isin(subject_ids_all, list(TRAIN_SUBJECTS))
test_mask  = np.isin(subject_ids_all, list(TEST_SUBJECTS))

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"\nRunning LazyPredict on raw windows (5200 features) — this will take longer...\n")

# ── LazyPredict ───────────────────────────────────────────────────────────
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print("\n========================================")
print("LAZYPREDICT — RAW WINDOWS (no feat. extraction)")
print("========================================")
print(models.to_string())

models.to_csv("lazypredict_raw_results.csv")
print("\nSaved: lazypredict_raw_results.csv")

print("\n========================================")
print("TOP 10 MODELS")
print("========================================")
print(models.head(10).to_string())
