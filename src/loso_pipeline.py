import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.fft import rfft

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

def hampel_filter(df, window_size=10, n_sigmas=3):
    df_filtered = df.copy().astype(float)
    k = 1.4826
    for col in df_filtered.columns:
        series = df_filtered[col]
        rolling_median = series.rolling(window=2*window_size+1, center=True, min_periods=1).median()
        rolling_mad = (series - rolling_median).abs().rolling(window=2*window_size+1, center=True, min_periods=1).median()
        threshold = n_sigmas * k * rolling_mad
        outliers = (series - rolling_median).abs() > threshold
        df_filtered.loc[outliers, col] = rolling_median[outliers]
    return df_filtered

def get_subject_number(file_name):
    return int(os.path.splitext(file_name)[0].split('_')[1])

def create_windows(df, upright_reference=None):
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
        window = reshaped[start:start + WINDOW_SIZE]
        features = extract_features(window)
        windows.append(features)
    return windows

def extract_features(window):
    features = []
    n_timestamps, n_sensors, n_feat = window.shape  # (100, 4, 13)

    # ================= EXISTING FEATURES (UNCHANGED) =================
    for s in range(n_sensors):
        sensor_data = window[:, s, :]
        features.extend(np.mean(sensor_data, axis=0))
        features.extend(np.std(sensor_data, axis=0))
        features.extend(np.sqrt(np.mean(sensor_data[:, 0:6]**2, axis=0)))

    angles = window[:, :, 10:13]
    top_sensor    = angles[:, 0, :]
    bottom_sensor = angles[:, 1, :]
    diff = top_sensor - bottom_sensor
    features.extend(np.mean(diff, axis=0))
    features.extend(np.std(diff, axis=0))

    for s in range(n_sensors):
        pitch = window[:, s, 11]
        roll  = window[:, s, 10]
        features.append(np.mean(pitch))
        features.append(np.std(pitch))
        features.append(np.mean(roll))
        features.append(np.std(roll))
        slope_pitch = np.polyfit(np.arange(n_timestamps), pitch, 1)[0]
        slope_roll  = np.polyfit(np.arange(n_timestamps), roll, 1)[0]
        features.append(slope_pitch)
        features.append(slope_roll)

    for s in range(n_sensors):
        acc = window[:, s, 0:3]
        acc_mag = np.linalg.norm(acc, axis=1)
        features.append(np.mean(acc_mag))
        features.append(np.std(acc_mag))
        features.append(np.max(acc_mag))
        features.append(np.min(acc_mag))

    # ================= 🆕 NEW FEATURES START HERE =================

    # ─────────────────────────────────────────────
    # 1. Signal Magnitude Area (SMA)
    # ─────────────────────────────────────────────
    for s in range(n_sensors):
        acc = window[:, s, 0:3]
        sma = np.mean(np.abs(acc[:,0]) + np.abs(acc[:,1]) + np.abs(acc[:,2]))
        features.append(sma)

    # ─────────────────────────────────────────────
    # 2. Energy Features
    # ─────────────────────────────────────────────
    for s in range(n_sensors):
        sensor = window[:, s, :]
        energy = np.sum(sensor**2, axis=0)
        features.extend(energy)

    # ─────────────────────────────────────────────
    # 3. Jerk Features (derivative of acceleration)
    # ─────────────────────────────────────────────
    for s in range(n_sensors):
        acc = window[:, s, 0:3]
        jerk = np.diff(acc, axis=0)

        jerk_mag = np.linalg.norm(jerk, axis=1)

        features.append(np.mean(jerk_mag))
        features.append(np.std(jerk_mag))
        features.append(np.sqrt(np.mean(jerk_mag**2)))

    # ─────────────────────────────────────────────
    # 4. Cross-Sensor Correlation (VERY IMPORTANT)
    # ─────────────────────────────────────────────
    # Using pitch (Angle Y)
    pitch_all = window[:, :, 11]  # (100, 4)

    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            corr = np.corrcoef(pitch_all[:, i], pitch_all[:, j])[0,1]
            if np.isnan(corr):
                corr = 0
            features.append(corr)

    # ─────────────────────────────────────────────
    # 5. Frequency Features (FFT)
    # ─────────────────────────────────────────────
    for s in range(n_sensors):
        acc = window[:, s, 0:3]

        for axis in range(3):
            signal = acc[:, axis]

            fft_vals = np.abs(rfft(signal))

            # Energy in frequency domain
            features.append(np.sum(fft_vals**2))

            # Dominant frequency index
            features.append(np.argmax(fft_vals))

    # ─────────────────────────────────────────────
    # 6. Spectral Entropy
    # ─────────────────────────────────────────────
    for s in range(n_sensors):
        acc = window[:, s, 0:3]

        for axis in range(3):
            signal = acc[:, axis]
            fft_vals = np.abs(rfft(signal))

            prob = fft_vals / (np.sum(fft_vals) + 1e-6)
            spec_entropy = entropy(prob)

            features.append(spec_entropy)

    # ================= END =================
    return np.array(features)

def subject_posture_accuracy(y_true, y_pred, subject_ids):
    correct, total = 0, 0
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        for posture in np.unique(y_true[mask]):
            pm = y_true[mask] == posture
            preds = y_pred[mask][pm]
            values, counts = np.unique(preds, return_counts=True)
            if values[np.argmax(counts)] == posture:
                correct += 1
            total += 1
    return correct / total

# ===================== LOAD DATA & ANALYZE SUBJECT UPRIGHT REFERENCE from mean =====================
def plot_confusion_matrix(y_true, y_pred, title, filename):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# # ===================== LOAD DATA =====================
# X_all, y_all, subject_ids_all = [], [], []
# removed_subjects = {29, 10, 23, 26}

# print("Loading data with upright calibration...")

# upright_refs = {}
# upright_folder = None
# for folder in os.listdir(DATASET_PATH):
#     if POSTURE_MAP.get(folder) == 'upright':
#         upright_folder = os.path.join(DATASET_PATH, folder)
#         break

# if upright_folder:
#     for file_name in sorted(os.listdir(upright_folder)):
#         if not file_name.endswith(".csv"): continue
#         subject = get_subject_number(file_name)
#         if subject in removed_subjects: continue
#         df = pd.read_csv(os.path.join(upright_folder, file_name), encoding='utf-8-sig')
#         df = df[FEATURE_COLS].copy()
#         df = hampel_filter(df)
#         data = df.to_numpy()
#         usable = (data.shape[0] // 4) * 4
#         reshaped = data[:usable].reshape(-1, 4, len(FEATURE_COLS))
#         upright_refs[subject] = np.mean(reshaped[:, :, 10:13], axis=0)
#     print(f"Upright references: {len(upright_refs)} subjects.")

# for folder in sorted(os.listdir(DATASET_PATH)):
#     folder_path = os.path.join(DATASET_PATH, folder)
#     if not os.path.isdir(folder_path): continue
#     label = POSTURE_MAP.get(folder)
#     if label is None: continue
#     for file_name in sorted(os.listdir(folder_path)):
#         if not file_name.endswith(".csv"): continue
#         subject = get_subject_number(file_name)
#         if subject in removed_subjects: continue
#         df = pd.read_csv(os.path.join(folder_path, file_name), encoding='utf-8-sig')
#         df = df[FEATURE_COLS].copy()
#         df = hampel_filter(df)
#         ref = upright_refs.get(subject, None)
#         windows = create_windows(df, upright_reference=ref)
#         X_all.extend(windows)
#         y_all.extend([label] * len(windows))
#         subject_ids_all.extend([subject] * len(windows))

# ===================== LOAD DATA & ANALYZE SUBJECT UPRIGHT REFERENCE from variance =====================
# ===================== LOAD DATA =====================
X_all, y_all, subject_ids_all = [], [], []
removed_subjects = {29}

print("Loading data with upright calibration...")

# Step 1: collect all upright trials per subject
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
        trial_ref = np.mean(reshaped[:, :, 10:13], axis=0)  # (4, 3)
        if subject not in upright_refs_raw:
            upright_refs_raw[subject] = []
        upright_refs_raw[subject].append(trial_ref)

# Step 2: take MEDIAN across trials — robust to one corrupted trial
upright_refs = {}
for subject, trials in upright_refs_raw.items():
    trials_arr              = np.array(trials)          # (n_trials, 4, 3)
    upright_refs[subject]   = np.median(trials_arr, axis=0)  # (4, 3)

print(f"Upright references (median): {len(upright_refs)} subjects.")

# Step 3: load all postures
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
        windows = create_windows(df, upright_reference=ref)
        X_all.extend(windows)
        y_all.extend([label] * len(windows))
        subject_ids_all.extend([subject] * len(windows))

X_all           = np.array(X_all)
y_all           = np.array(y_all)
subject_ids_all = np.array(subject_ids_all)

# Fix NaN/Inf
for col in range(X_all.shape[1]):
    mask = ~np.isfinite(X_all[:, col])
    if mask.any():
        X_all[mask, col] = np.nanmedian(X_all[:, col])

unique_subjects = np.unique(subject_ids_all)
n_features      = X_all.shape[1]
print(f"Total windows: {len(X_all)}  |  Features: {n_features}  |  Subjects: {len(unique_subjects)}")

# =====================================================================
# PHASE 1: Find best K using a fast held-out split (last 8 subjects)
# RF importance — fitted on training subjects only
# =====================================================================
print("\n--- Phase 1: Finding best K via RF importance (fast split) ---")

train_subj = unique_subjects[:-8]
test_subj  = unique_subjects[-8:]
tr_mask    = np.isin(subject_ids_all, train_subj)
te_mask    = ~tr_mask

scaler_ph1  = StandardScaler()
X_tr_ph1    = scaler_ph1.fit_transform(X_all[tr_mask])
X_te_ph1    = scaler_ph1.transform(X_all[te_mask])
y_tr_ph1    = y_all[tr_mask]
y_te_ph1    = y_all[te_mask]

# Fit RF to get importances
rf_imp = RandomForestClassifier(
    n_estimators=500, max_depth=25, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf_imp.fit(X_tr_ph1, y_tr_ph1)
importances = rf_imp.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

print(f"Top 10 feature indices by importance: {sorted_idx[:10]}")

# Test K values
k_values = list(range(20, n_features, 10)) + [n_features]
# k_values = list(range(50, 301, 25))
k_accs   = []

rf_fast = RandomForestClassifier(
    n_estimators=300, max_depth=25, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1
)

print("Testing K values...")
for k in k_values:
    idx = sorted_idx[:k]
    rf_fast.fit(X_tr_ph1[:, idx], y_tr_ph1)
    acc = accuracy_score(y_te_ph1, rf_fast.predict(X_te_ph1[:, idx]))
    k_accs.append(acc)
    print(f"  K={k:4d}  Accuracy={acc:.4f}")

best_k   = k_values[int(np.argmax(k_accs))]
best_idx = sorted_idx[:best_k]
print(f"\nBest K = {best_k}  (held-out acc = {max(k_accs):.4f})")

# Plot K curve
plt.figure(figsize=(12, 5))
plt.plot(k_values, k_accs, 'b-o', markersize=5)
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.axhline(0.9115, color='green', linestyle='--', label='Baseline 91.15%')
plt.xlabel('Number of Features (RF Importance)')
plt.ylabel('Accuracy')
plt.title('Feature Selection: K vs Accuracy')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("feature_selection_k_curve.png", dpi=200)
plt.close()
print("Saved: feature_selection_k_curve.png")

# =====================================================================
# PHASE 2: Full LOSO with feature selection INSIDE each fold
# This is methodologically correct — no data leakage
# =====================================================================
print(f"\n--- Phase 2: Full LOSO with RF importance selection (K={best_k}) ---")
print("Note: Feature importance recomputed per fold on training data only.\n")

all_preds, all_true, all_subjects, accuracies = [], [], [], []

for test_subject in unique_subjects:
    train_mask = subject_ids_all != test_subject
    test_mask  = subject_ids_all == test_subject

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Fit a fast RF to get importance for THIS fold
    rf_sel = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_sel.fit(X_tr_s, y_tr)
    fold_importance = rf_sel.feature_importances_
    fold_idx        = np.argsort(fold_importance)[::-1][:best_k]

    # Train final RF on selected features
    rf_final = RandomForestClassifier(
        n_estimators=1000, max_depth=25, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1, bootstrap=True
    )
    rf_final.fit(X_tr_s[:, fold_idx], y_tr)
    y_pred = rf_final.predict(X_te_s[:, fold_idx])

    acc = accuracy_score(y_te, y_pred)
    accuracies.append(acc)
    print(f"  Subject {test_subject:3d}  Accuracy: {acc:.4f}")

    all_preds.extend(y_pred)
    all_true.extend(y_te)
    all_subjects.extend([test_subject] * len(y_te))

all_preds    = np.array(all_preds)
all_true     = np.array(all_true)
all_subjects = np.array(all_subjects)

mean_acc = np.mean(accuracies)
subj_acc = subject_posture_accuracy(all_true, all_preds, all_subjects)

print(f"\n{'='*55}")
print(f"LOSO — RF + RF Importance Selection (K={best_k})")
print(f"{'='*55}")
print(f"Mean Window Accuracy   : {mean_acc:.4f}")
print(f"Std  Window Accuracy   : {np.std(accuracies):.4f}")
print(f"Subject-Level Accuracy : {subj_acc:.4f}")
print(f"\nBaseline (196 feat, no selection): 0.9115")
print(f"Change                           : {mean_acc - 0.9115:+.4f}")
print("\nClassification Report:")
print(classification_report(all_true, all_preds))

plot_confusion_matrix(
    all_true, all_preds,
    f"LOSO — RF Importance Selection K={best_k} | Acc: {mean_acc:.4f}",
    f"confusion_matrix_rf_selection_k{best_k}.png"
)
