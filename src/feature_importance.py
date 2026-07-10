import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\DELL\Desktop\GP\Aquisition\Labelled_trials_FIXED"

POSTURE_MAP = {
    "1": "backward_bending", "2": "upright", "3": "slouching",
    "4": "forward_bending",  "5": "right_bending", "6": "left_bending"
}

FEATURE_COLS = [
    "Acceleration X(g)", "Acceleration Y(g)", "Acceleration Z(g)",
    "Angular velocity X(°/s)", "Angular velocity Y(°/s)", "Angular velocity Z(°/s)",
    "Quaternions 0()", "Quaternions 1()", "Quaternions 2()", "Quaternions 3()",
    "Angle X(°)", "Angle Y(°)", "Angle Z(°)"
]

# Human-readable names for the 13 channels
CHANNEL_NAMES = [
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyro_X", "Gyro_Y", "Gyro_Z",
    "Quat_0", "Quat_1", "Quat_2", "Quat_3",
    "Roll(AngleX)", "Pitch(AngleY)", "Yaw(AngleZ)"
]

SENSOR_NAMES = ["C7", "L5", "T12", "T4"]

WINDOW_SIZE = 100
WINDOW_STEP = 50
removed_subjects = {29}


# ── Build feature names (196 total) ──────────────────────────────────────
def build_feature_names():
    names = []
    # B1 — mean + std + RMS per sensor (32 × 4 = 128)
    for s in SENSOR_NAMES:
        for ch in CHANNEL_NAMES:
            names.append(f"B1_mean_{s}_{ch}")
        for ch in CHANNEL_NAMES:
            names.append(f"B1_std_{s}_{ch}")
        for ch in CHANNEL_NAMES[:6]:   # RMS only for Acc+Gyro
            names.append(f"B1_rms_{s}_{ch}")
    # B2 — C7 vs L5 angle diff (6)
    for ax in ["Roll","Pitch","Yaw"]:
        names.append(f"B2_mean_C7-L5_{ax}")
    for ax in ["Roll","Pitch","Yaw"]:
        names.append(f"B2_std_C7-L5_{ax}")
    # B3 — pitch + roll mean/std/slope per sensor (24)
    for s in SENSOR_NAMES:
        names += [f"B3_mean_Pitch_{s}", f"B3_std_Pitch_{s}",
                  f"B3_mean_Roll_{s}",  f"B3_std_Roll_{s}",
                  f"B3_slope_Pitch_{s}", f"B3_slope_Roll_{s}"]
    # B4 — acc magnitude (16)
    for s in SENSOR_NAMES:
        names += [f"B4_accmag_mean_{s}", f"B4_accmag_std_{s}",
                  f"B4_accmag_max_{s}",  f"B4_accmag_min_{s}"]
    # B5 — curvature (2)
    names += ["B5_curv_mean", "B5_curv_std"]
    # B6 — spine slope (2)
    names += ["B6_slope", "B6_slope_abs"]
    # B7 — sign agreement (1)
    names += ["B7_sign_C7xL5"]
    # B8 — slouch discriminators (17)
    names += ["B8_L5-T12", "B8_abs_L5-T12", "B8_sign_L5-T12",
              "B8_mean_L5-T12_ts", "B8_std_L5-T12_ts", "B8_sign_mean_L5-T12_ts",
              "B8_frac_L5>T12",
              "B8_L5-T4", "B8_abs_L5-T4", "B8_T12-T4", "B8_abs_T12-T4",
              "B8_upper_grad", "B8_lower_grad", "B8_grad_ratio",
              "B8_accZ_C7-L5", "B8_accZ_T4-T12", "B8_accZ_var"]
    return names

FEATURE_NAMES = build_feature_names()
print(f"Feature names built: {len(FEATURE_NAMES)}")


def hampel_filter(df, window_size=10, n_sigmas=3):
    df_filtered = df.copy().astype(float)
    k = 1.4826
    for col in df_filtered.columns:
        series = df_filtered[col]
        med = series.rolling(2*window_size+1, center=True, min_periods=1).median()
        mad = (series-med).abs().rolling(2*window_size+1, center=True, min_periods=1).median()
        df_filtered.loc[(series-med).abs() > n_sigmas*k*mad, col] = med[(series-med).abs() > n_sigmas*k*mad]
    return df_filtered

def get_subject_number(f):
    return int(os.path.splitext(f)[0].split('_')[1])

def create_windows(df, upright_reference=None):
    n_sensors = 4; n_features = len(FEATURE_COLS)
    data = df.to_numpy()
    usable = (data.shape[0]//n_sensors)*n_sensors
    reshaped = data[:usable].reshape(-1, n_sensors, n_features)
    if upright_reference is not None:
        reshaped[:,:,10:13] -= upright_reference
    else:
        reshaped[:,:,10:13] -= np.mean(reshaped[:,:,10:13], axis=0)
    windows = []
    for start in range(0, reshaped.shape[0]-WINDOW_SIZE+1, WINDOW_STEP):
        windows.append(extract_features(reshaped[start:start+WINDOW_SIZE]))
    return windows

def extract_features(window):
    features = []
    n_timestamps, n_sensors, n_feat = window.shape

    for s in range(n_sensors):
        sensor_data = window[:,s,:]
        features.extend(np.mean(sensor_data, axis=0))
        features.extend(np.std(sensor_data, axis=0))
        features.extend(np.sqrt(np.mean(sensor_data[:,0:6]**2, axis=0)))

    angles = window[:,:,10:13]
    diff = angles[:,0,:] - angles[:,1,:]
    features.extend(np.mean(diff, axis=0))
    features.extend(np.std(diff, axis=0))

    for s in range(n_sensors):
        pitch = window[:,s,11]; roll = window[:,s,10]
        features += [np.mean(pitch), np.std(pitch), np.mean(roll), np.std(roll),
                     np.polyfit(np.arange(n_timestamps), pitch, 1)[0],
                     np.polyfit(np.arange(n_timestamps), roll,  1)[0]]

    for s in range(n_sensors):
        acc_mag = np.linalg.norm(window[:,s,0:3], axis=1)
        features += [np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag), np.min(acc_mag)]

    sp = np.array([0,1,2,3])
    pc = np.array([np.polyfit(sp, window[t,:,11], 2)[0] for t in range(n_timestamps)])
    features += [np.mean(pc), np.std(pc)]

    mps = np.mean(window[:,:,11], axis=0)
    sl = np.polyfit(sp, mps, 1)[0]
    features += [sl, abs(sl)]
    features.append(np.sign(mps[0]) * np.sign(mps[1]))

    pC7=window[:,0,11]; pT4=window[:,3,11]; pT12=window[:,2,11]; pL5=window[:,1,11]
    mC7=np.mean(pC7); mT4=np.mean(pT4); mT12=np.mean(pT12); mL5=np.mean(pL5)
    d=pL5-pT12
    features += [mL5-mT12, abs(mL5-mT12), np.sign(mL5-mT12),
                 np.mean(d), np.std(d), np.sign(np.mean(d)), np.mean(d>0),
                 mL5-mT4, abs(mL5-mT4), mT12-mT4, abs(mT12-mT4)]
    ug=mT4-mC7; lg=mL5-mT12
    features += [ug, lg, lg/(abs(ug)+1e-6)]
    features += [np.mean(window[:,0,2])-np.mean(window[:,1,2]),
                 np.mean(window[:,3,2])-np.mean(window[:,2,2]),
                 np.var([np.mean(window[:,0,2]), np.mean(window[:,3,2]),
                         np.mean(window[:,2,2]), np.mean(window[:,1,2])])]
    return np.array(features)


# ── Load data ─────────────────────────────────────────────────────────────
X_all, y_all, subj_all = [], [], []
print("Loading data with upright calibration...")

upright_refs_raw = {}
upright_folder = None
for folder in os.listdir(DATASET_PATH):
    if POSTURE_MAP.get(folder) == 'upright':
        upright_folder = os.path.join(DATASET_PATH, folder); break

if upright_folder:
    for fn in sorted(os.listdir(upright_folder)):
        if not fn.endswith(".csv"): continue
        subj = get_subject_number(fn)
        if subj in removed_subjects: continue
        df = pd.read_csv(os.path.join(upright_folder, fn), encoding='utf-8-sig')
        df = hampel_filter(df[FEATURE_COLS].copy())
        data = df.to_numpy(); usable = (data.shape[0]//4)*4
        reshaped = data[:usable].reshape(-1, 4, len(FEATURE_COLS))
        ref = np.mean(reshaped[:,:,10:13], axis=0)
        if subj not in upright_refs_raw: upright_refs_raw[subj] = []
        upright_refs_raw[subj].append(ref)

upright_refs = {s: np.median(np.array(t), axis=0) for s,t in upright_refs_raw.items()}
print(f"Upright references: {len(upright_refs)} subjects.")

for folder in sorted(os.listdir(DATASET_PATH)):
    fp = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(fp): continue
    label = POSTURE_MAP.get(folder)
    if label is None: continue
    for fn in sorted(os.listdir(fp)):
        if not fn.endswith(".csv"): continue
        subj = get_subject_number(fn)
        if subj in removed_subjects: continue
        df = pd.read_csv(os.path.join(fp, fn), encoding='utf-8-sig')
        df = hampel_filter(df[FEATURE_COLS].copy())
        ref = upright_refs.get(subj, None)
        for w in create_windows(df, upright_reference=ref):
            X_all.append(w); y_all.append(label); subj_all.append(subj)

X_all = np.array(X_all); y_all = np.array(y_all); subj_all = np.array(subj_all)
for c in range(X_all.shape[1]):
    m = ~np.isfinite(X_all[:,c])
    if m.any(): X_all[m,c] = np.nanmedian(X_all[:,c])

print(f"Total windows: {len(X_all)}  Features: {X_all.shape[1]}  Subjects: {len(np.unique(subj_all))}")

# ── Run LOSO and accumulate importance scores ─────────────────────────────
print("\nRunning LOSO to collect per-fold feature importances...")

unique_subjects = np.unique(subj_all)
all_importances = np.zeros(X_all.shape[1])

rf = RandomForestClassifier(
    n_estimators=1000, max_depth=25, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1, bootstrap=True
)

loso_accs = []
for fold_num, test_subj in enumerate(unique_subjects, 1):
    train_mask = subj_all != test_subj
    test_mask  = subj_all == test_subj

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask],  y_all[test_mask]

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, rf.predict(X_te))
    loso_accs.append(acc)
    all_importances += rf.feature_importances_

    print(f"  [{fold_num:2d}/45] Subject {test_subj}  Acc={acc:.4f}")

# Average importance across folds
avg_importances = all_importances / len(unique_subjects)
sorted_idx      = np.argsort(avg_importances)[::-1]

print(f"\nMean LOSO Window Accuracy: {np.mean(loso_accs):.4f}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Top 30 individual features
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"TOP 30 FEATURES BY AVERAGE LOSO IMPORTANCE")
print(f"{'='*65}")
for rank, idx in enumerate(sorted_idx[:30], 1):
    print(f"  #{rank:2d}  imp={avg_importances[idx]:.4f}  {FEATURE_NAMES[idx]}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Importance per SENSOR CHANNEL (which of 13 params matter)
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"IMPORTANCE BY SENSOR CHANNEL (averaged across sensors & feature types)")
print(f"{'='*65}")

channel_importance = {ch: 0.0 for ch in CHANNEL_NAMES}
channel_count      = {ch: 0   for ch in CHANNEL_NAMES}

for idx, name in enumerate(FEATURE_NAMES):
    for ch in CHANNEL_NAMES:
        if ch in name:
            channel_importance[ch] += avg_importances[idx]
            channel_count[ch]      += 1
            break

# Normalize by count so channels with more derived features aren't unfairly boosted
channel_avg = {ch: channel_importance[ch]/max(channel_count[ch],1) for ch in CHANNEL_NAMES}
sorted_channels = sorted(channel_avg.items(), key=lambda x: x[1], reverse=True)

for ch, imp in sorted_channels:
    bar = '█' * int(imp * 500)
    print(f"  {ch:20s}  avg_imp={imp:.5f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Importance by BLOCK
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"IMPORTANCE BY FEATURE BLOCK")
print(f"{'='*65}")

block_importance = {}
for idx, name in enumerate(FEATURE_NAMES):
    block = name.split('_')[0]
    if block not in block_importance: block_importance[block] = 0.0
    block_importance[block] += avg_importances[idx]

for block, imp in sorted(block_importance.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 30)
    print(f"  {block:8s}  total_imp={imp:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Which channels to KEEP vs DROP
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"RECOMMENDATION — WHICH SENSOR CHANNELS TO KEEP")
print(f"{'='*65}")

total_imp = sum(channel_avg.values())
cumulative = 0.0
kept = []
for ch, imp in sorted_channels:
    cumulative += imp
    pct = imp / total_imp * 100
    kept.append(ch)
    print(f"  {'KEEP' if cumulative/total_imp < 0.90 else 'OPTIONAL':8s}  {ch:20s}  {pct:.1f}%  (cumulative: {cumulative/total_imp*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# PLOT — Channel importance bar chart
# ══════════════════════════════════════════════════════════════════════════
channels = [ch for ch, _ in sorted_channels]
importances_vals = [imp for _, imp in sorted_channels]

colors = []
for ch in channels:
    if 'Angle' in ch or 'Roll' in ch or 'Pitch' in ch or 'Yaw' in ch:
        colors.append('#0D9488')   # teal = angles
    elif 'Acc' in ch:
        colors.append('#1B2A4A')   # navy = acceleration
    elif 'Gyro' in ch:
        colors.append('#D97706')   # amber = gyroscope
    else:
        colors.append('#64748B')   # gray = quaternions

plt.figure(figsize=(12, 6))
bars = plt.bar(channels, importances_vals, color=colors, edgecolor='white', linewidth=0.5)
plt.xlabel('Sensor Channel', fontsize=12)
plt.ylabel('Average Importance (across LOSO folds)', fontsize=12)
plt.title('Feature Importance by Sensor Channel\n196-Feature Model · LOSO · 45 Subjects', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#0D9488', label='Euler Angles'),
    Patch(facecolor='#1B2A4A', label='Acceleration'),
    Patch(facecolor='#D97706', label='Gyroscope'),
    Patch(facecolor='#64748B', label='Quaternions'),
]
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('channel_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: channel_importance.png")
