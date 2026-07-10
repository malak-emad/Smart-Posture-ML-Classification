import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
BEST_K      = 150   # from your Phase 1 results

CLASSES = [
    "backward_bending",
    "forward_bending",
    "left_bending",
    "right_bending",
    "slouching",
    "upright"
]

# Colors matching your deck style
COLORS = {
    "backward_bending": "#1B2A4A",   # navy
    "forward_bending" : "#D97706",   # amber
    "left_bending"    : "#0D9488",   # teal
    "right_bending"   : "#0369A1",   # blue
    "slouching"       : "#CC0000",   # red
    "upright"         : "#7C3AED",   # purple
}

LABELS = {
    "backward_bending": "Backward Bending",
    "forward_bending" : "Forward Bending",
    "left_bending"    : "Left Bending",
    "right_bending"   : "Right Bending",
    "slouching"       : "Slouching",
    "upright"         : "Upright",
}


# ── Preprocessing functions ───────────────────────────────────────────────
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
        window   = reshaped[start:start + WINDOW_SIZE]
        features = extract_features(window)
        windows.append(features)
    return windows

def extract_features(window):
    features = []
    n_timestamps, n_sensors, n_feat = window.shape

    for s in range(n_sensors):
        sensor_data = window[:, s, :]
        features.extend(np.mean(sensor_data, axis=0))
        features.extend(np.std(sensor_data, axis=0))
        features.extend(np.sqrt(np.mean(sensor_data[:, 0:6]**2, axis=0)))

    angles        = window[:, :, 10:13]
    top_sensor    = angles[:, 0, :]
    bottom_sensor = angles[:, 1, :]
    diff          = top_sensor - bottom_sensor
    features.extend(np.mean(diff, axis=0))
    features.extend(np.std(diff, axis=0))

    for s in range(n_sensors):
        pitch = window[:, s, 11]
        roll  = window[:, s, 10]
        features.append(np.mean(pitch));  features.append(np.std(pitch))
        features.append(np.mean(roll));   features.append(np.std(roll))
        features.append(np.polyfit(np.arange(n_timestamps), pitch, 1)[0])
        features.append(np.polyfit(np.arange(n_timestamps), roll,  1)[0])

    for s in range(n_sensors):
        acc     = window[:, s, 0:3]
        acc_mag = np.linalg.norm(acc, axis=1)
        features.append(np.mean(acc_mag)); features.append(np.std(acc_mag))
        features.append(np.max(acc_mag));  features.append(np.min(acc_mag))

    sensor_positions = np.array([0, 1, 2, 3])
    poly_coeffs = np.array([
        np.polyfit(sensor_positions, window[t, :, 11], 2)[0]
        for t in range(n_timestamps)
    ])
    features.append(np.mean(poly_coeffs)); features.append(np.std(poly_coeffs))

    mean_pitch_spine = np.mean(window[:, :, 11], axis=0)
    spine_slope      = np.polyfit(sensor_positions, mean_pitch_spine, 1)[0]
    features.append(spine_slope); features.append(abs(spine_slope))
    features.append(np.sign(mean_pitch_spine[0]) * np.sign(mean_pitch_spine[3]))

    pitch_C7  = window[:, 0, 11]; pitch_T4  = window[:, 3, 11]
    pitch_T12 = window[:, 2, 11]; pitch_L5  = window[:, 1, 11]
    mean_p_C7  = np.mean(pitch_C7);  mean_p_T4  = np.mean(pitch_T4)
    mean_p_T12 = np.mean(pitch_T12); mean_p_L5  = np.mean(pitch_L5)

    L5_minus_T12   = mean_p_L5 - mean_p_T12
    L5_T12_diff_ts = pitch_L5  - pitch_T12
    features.append(L5_minus_T12);                      features.append(abs(L5_minus_T12))
    features.append(np.sign(L5_minus_T12));             features.append(np.mean(L5_T12_diff_ts))
    features.append(np.std(L5_T12_diff_ts));            features.append(np.sign(np.mean(L5_T12_diff_ts)))
    features.append(np.mean(L5_T12_diff_ts > 0))
    L5_minus_T4  = mean_p_L5  - mean_p_T4
    T12_minus_T4 = mean_p_T12 - mean_p_T4
    features.append(L5_minus_T4);  features.append(abs(L5_minus_T4))
    features.append(T12_minus_T4); features.append(abs(T12_minus_T4))
    upper_grad = mean_p_T4 - mean_p_C7; lower_grad = mean_p_L5 - mean_p_T12
    features.append(upper_grad);   features.append(lower_grad)
    features.append(lower_grad / (abs(upper_grad) + 1e-6))
    acc_z_C7  = np.mean(window[:, 0, 2]); acc_z_T4  = np.mean(window[:, 3, 2])
    acc_z_T12 = np.mean(window[:, 2, 2]); acc_z_L5  = np.mean(window[:, 1, 2])
    features.append(acc_z_C7 - acc_z_L5); features.append(acc_z_T4 - acc_z_T12)
    features.append(np.var([acc_z_C7, acc_z_T4, acc_z_T12, acc_z_L5]))

    return np.array(features)


# ── Load data ─────────────────────────────────────────────────────────────
X_all, y_all, subject_ids_all = [], [], []
removed_subjects = {29}

print("Loading data...")

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
        windows = create_windows(df, upright_reference=ref)
        X_all.extend(windows)
        y_all.extend([label] * len(windows))
        subject_ids_all.extend([subject] * len(windows))

X_all           = np.array(X_all)
y_all           = np.array(y_all)
subject_ids_all = np.array(subject_ids_all)

for col in range(X_all.shape[1]):
    mask = ~np.isfinite(X_all[:, col])
    if mask.any():
        X_all[mask, col] = np.nanmedian(X_all[:, col])

unique_subjects = np.unique(subject_ids_all)
print(f"Total windows: {len(X_all)}  |  Features: {X_all.shape[1]}  |  Subjects: {len(unique_subjects)}")

# ── LOSO — collect probabilities ──────────────────────────────────────────
print("\nRunning LOSO to collect class probabilities...")

all_true   = []
all_proba  = []   # probability vectors shape (n, 6)

for test_subject in unique_subjects:
    train_mask = subject_ids_all != test_subject
    test_mask  = subject_ids_all == test_subject

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # Feature selection — same as your pipeline
    rf_sel = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_sel.fit(X_tr_s, y_tr)
    fold_idx = np.argsort(rf_sel.feature_importances_)[::-1][:BEST_K]

    rf_final = RandomForestClassifier(
        n_estimators=1000, max_depth=25, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1, bootstrap=True
    )
    rf_final.fit(X_tr_s[:, fold_idx], y_tr)

    proba = rf_final.predict_proba(X_te_s[:, fold_idx])  # (n, 6)

    all_true.extend(y_te)
    all_proba.append(proba)

    print(f"  Subject {test_subject:3d} done")

all_true  = np.array(all_true)
all_proba = np.vstack(all_proba)   # (23490, 6)

# ── Binarize labels for one-vs-rest curves ────────────────────────────────
# rf_final.classes_ gives the class order — must match columns of all_proba
class_order = rf_final.classes_   # alphabetical: backward, forward, left, right, slouching, upright
y_bin = label_binarize(all_true, classes=class_order)  # (23490, 6)

print(f"\nClass order: {list(class_order)}")
print(f"Probability matrix shape: {all_proba.shape}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ROC Curves (one per class)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

roc_aucs = {}
for i, cls in enumerate(class_order):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_proba[:, i])
    roc_auc     = auc(fpr, tpr)
    roc_aucs[cls] = roc_auc
    ax.plot(fpr, tpr,
            color=COLORS[cls],
            linewidth=2.2,
            label=f"{LABELS[cls]}  (AUC = {roc_auc:.3f})")

# Random baseline
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random classifier (AUC = 0.500)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curves — One-vs-Rest\nLOSO Cross-Validation · 45 Subjects · 23,490 Windows',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# AUC summary box
auc_text = "AUC Summary\n" + "\n".join(
    [f"{LABELS[c]}: {roc_aucs[c]:.3f}" for c in class_order]
)
ax.text(0.36, 0.32, auc_text, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#CBD5E1', alpha=0.95))

plt.tight_layout()
plt.savefig("roc_curves_loso.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: roc_curves_loso.png")

# Print AUC values
print("\n── ROC AUC per class ──")
for cls in class_order:
    print(f"  {LABELS[cls]:25s}  AUC = {roc_aucs[cls]:.4f}")
macro_auc = np.mean(list(roc_aucs.values()))
print(f"\n  Macro-average AUC: {macro_auc:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Precision-Recall Curves (one per class)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

pr_aucs = {}
for i, cls in enumerate(class_order):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], all_proba[:, i])
    ap = average_precision_score(y_bin[:, i], all_proba[:, i])
    pr_aucs[cls] = ap
    ax.plot(recall, precision,
            color=COLORS[cls],
            linewidth=2.2,
            label=f"{LABELS[cls]}  (AP = {ap:.3f})")

# Baseline: random classifier precision = class frequency = 1/6
baseline = 1 / len(class_order)
ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1,
           alpha=0.5, label=f'Random classifier (AP ≈ {baseline:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — One-vs-Rest\nLOSO Cross-Validation · 45 Subjects · 23,490 Windows',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("pr_curves_loso.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: pr_curves_loso.png")

# Print AP values
print("\n── Average Precision (AP) per class ──")
for cls in class_order:
    print(f"  {LABELS[cls]:25s}  AP = {pr_aucs[cls]:.4f}")
macro_ap = np.mean(list(pr_aucs.values()))
print(f"\n  Macro-average AP: {macro_ap:.4f}")
