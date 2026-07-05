import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────
# PFADE - nur hier anpassen!
# ─────────────────────────────────────────
PFAD = "C:/Users/karst/OneDrive/Desktop/program"

# ─────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────
df1 = pd.read_csv(f"{PFAD}/T-1.csv", skipinitialspace=True)
df2 = pd.read_csv(f"{PFAD}/T-2.csv", skipinitialspace=True)

df1['time_sec'] = df1['elapsedTime']
df2['time_sec'] = df2['elapsedTime']

# Lineare Beschleunigung (ohne Schwerkraft)
for df in [df1, df2]:
    for axis in ['x', 'y', 'z']:
        df[f'lin_{axis}'] = df[f'Acceleration_{axis}'] - df[f'Gravity_{axis}']
    df['magnitude'] = np.sqrt(df['lin_x']**2 + df['lin_y']**2 + df['lin_z']**2)

print(f"T-1 (ruhig):   {len(df1)} Samples, {df1['time_sec'].max():.1f} Sekunden")
print(f"T-2 (unruhig): {len(df2)} Samples, {df2['time_sec'].max():.1f} Sekunden")


# ─────────────────────────────────────────
# 2. VISUALISIERUNG
# ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("AirPods Sensor-Vergleich: T-1 (ruhig) vs T-2 (unruhig)", fontsize=15, fontweight='bold')

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

plot_configs = [
    ("Lineare Beschleunigung X", 'lin_x',      'royalblue'),
    ("Lineare Beschleunigung Y", 'lin_y',      'darkorange'),
    ("Lineare Beschleunigung Z", 'lin_z',      'green'),
    ("Bewegungs-Magnitude",      'magnitude',  'purple'),
    ("Rotation X (Gyro)",        'Rotation_x', 'red'),
    ("Rotation Y (Gyro)",        'Rotation_y', 'brown'),
    ("Pitch",                    'pitch',      'teal'),
    ("Yaw",                      'yaw',        'magenta'),
]

for i, (title, col, color) in enumerate(plot_configs):
    row, c = divmod(i, 2)
    ax = fig.add_subplot(gs[row, c])
    ax.plot(df1['time_sec'], df1[col], color=color, alpha=0.8, linewidth=0.6, label='T-1 ruhig')
    ax.plot(df2['time_sec'], df2[col], color=color, alpha=0.35, linewidth=0.6, linestyle='--', label='T-2 unruhig')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Zeit (s)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.savefig(f"{PFAD}/airpods_signalvergleich.png", dpi=150, bbox_inches='tight')
print("Plot gespeichert: airpods_signalvergleich.png")


# ─────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features(df, label):
    f = {'label': label}

    # Lineare Beschleunigung
    for axis in ['x', 'y', 'z']:
        col = df[f'lin_{axis}']
        f[f'lin_{axis}_mean'] = col.mean()
        f[f'lin_{axis}_std']  = col.std()
        f[f'lin_{axis}_max']  = col.abs().max()

    # Magnitude
    f['magnitude_mean'] = df['magnitude'].mean()
    f['magnitude_std']  = df['magnitude'].std()
    f['magnitude_max']  = df['magnitude'].max()

    # Bewegungsereignisse
    threshold = df['magnitude'].quantile(0.75)
    f['movement_events'] = (df['magnitude'] > threshold).sum()
    f['stillness_ratio'] = (df['magnitude'] < 0.02).mean()

    # Gyroscope
    for axis in ['x', 'y', 'z']:
        col = df[f'Rotation_{axis}']
        f[f'rot_{axis}_mean'] = col.mean()
        f[f'rot_{axis}_std']  = col.std()
        f[f'rot_{axis}_max']  = col.abs().max()

    # Euler-Winkel
    for angle in ['pitch', 'roll', 'yaw']:
        col = df[angle]
        f[f'{angle}_mean']  = col.mean()
        f[f'{angle}_std']   = col.std()
        f[f'{angle}_range'] = col.max() - col.min()

    return f

feat1 = extract_features(df1, 'T-1 (ruhig)')
feat2 = extract_features(df2, 'T-2 (unruhig)')

feat_df = pd.DataFrame([feat1, feat2]).set_index('label').T

print("\n" + "="*60)
print("FEATURE-VERGLEICH")
print("="*60)
print(feat_df.to_string())


# ─────────────────────────────────────────
# 4. FEATURE-VERGLEICH PLOT
# ─────────────────────────────────────────
top_features = [
    'magnitude_mean', 'magnitude_std', 'magnitude_max',
    'movement_events', 'stillness_ratio',
    'rot_x_std', 'rot_y_std', 'rot_z_std',
    'pitch_range', 'yaw_range', 'roll_range'
]

vals1 = [feat1[f] for f in top_features]
vals2 = [feat2[f] for f in top_features]

max_vals = [max(abs(v1), abs(v2), 1e-9) for v1, v2 in zip(vals1, vals2)]
vals1_norm = [v / m for v, m in zip(vals1, max_vals)]
vals2_norm = [v / m for v, m in zip(vals2, max_vals)]

x = np.arange(len(top_features))
width = 0.35

fig2, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, vals1_norm, width, label='T-1 ruhig',   color='royalblue', alpha=0.8)
ax.bar(x + width/2, vals2_norm, width, label='T-2 unruhig', color='tomato',    alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(top_features, rotation=35, ha='right', fontsize=9)
ax.set_ylabel("Normalisierter Wert")
ax.set_title("Feature-Vergleich: ruhig vs. unruhig (normalisiert)", fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PFAD}/airpods_features.png", dpi=150, bbox_inches='tight')
print("Feature-Plot gespeichert: airpods_features.png")


# ─────────────────────────────────────────
# 5. VERTEILUNGSPLOTS (Histogramm / Dichte)
# ─────────────────────────────────────────
dist_cols = [
    ('magnitude',  'Bewegungs-Magnitude'),
    ('lin_x',      'Lineare Beschl. X'),
    ('lin_y',      'Lineare Beschl. Y'),
    ('lin_z',      'Lineare Beschl. Z'),
    ('Rotation_x', 'Gyro X'),
    ('Rotation_y', 'Gyro Y'),
]

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
fig3.suptitle("Verteilungsvergleich: ruhig vs. unruhig", fontsize=13, fontweight='bold')

for ax, (col, title) in zip(axes3.flat, dist_cols):
    d1 = df1[col].dropna().values
    d2 = df2[col].dropna().values
    lo = min(np.percentile(d1, 1),  np.percentile(d2, 1))
    hi = max(np.percentile(d1, 99), np.percentile(d2, 99))
    bins = np.linspace(lo, hi, 60)
    ax.hist(d1, bins=bins, density=True, alpha=0.5, color='royalblue', label='T-1 ruhig')
    ax.hist(d2, bins=bins, density=True, alpha=0.5, color='tomato',    label='T-2 unruhig')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel("Dichte", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PFAD}/airpods_verteilungen.png", dpi=150, bbox_inches='tight')
print("Verteilungsplot gespeichert: airpods_verteilungen.png")


# ─────────────────────────────────────────
# 6. VIOLIN-PLOTS
# ─────────────────────────────────────────
violin_cols   = ['magnitude', 'lin_x', 'lin_y', 'lin_z', 'Rotation_x', 'Rotation_y', 'Rotation_z', 'pitch', 'roll', 'yaw']
violin_labels = ['Magnitude', 'Lin X', 'Lin Y', 'Lin Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Pitch', 'Roll', 'Yaw']

fig4, axes4 = plt.subplots(2, 5, figsize=(18, 8))
fig4.suptitle("Violin-Plots: ruhig vs. unruhig", fontsize=13, fontweight='bold')

for ax, col, vlabel in zip(axes4.flat, violin_cols, violin_labels):
    d1 = df1[col].dropna().values
    d2 = df2[col].dropna().values
    p_lo = min(np.percentile(d1, 1),  np.percentile(d2, 1))
    p_hi = max(np.percentile(d1, 99), np.percentile(d2, 99))
    d1c = d1[(d1 >= p_lo) & (d1 <= p_hi)]
    d2c = d2[(d2 >= p_lo) & (d2 <= p_hi)]

    vp = ax.violinplot([d1c, d2c], positions=[1, 2], showmedians=True, showextrema=False)
    vp['bodies'][0].set_facecolor('royalblue')
    vp['bodies'][0].set_alpha(0.6)
    vp['bodies'][1].set_facecolor('tomato')
    vp['bodies'][1].set_alpha(0.6)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ruhig', 'unruhig'], fontsize=8)
    ax.set_title(vlabel, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{PFAD}/airpods_violins.png", dpi=150, bbox_inches='tight')
print("Violin-Plots gespeichert: airpods_violins.png")


# ─────────────────────────────────────────
# 7. SEPARIERBARKEITS-ÜBERSICHT
# ─────────────────────────────────────────
all_feat_keys = [f for f in feat1.keys() if f != 'label']
sep_diffs = [(f, ((feat2[f] - feat1[f]) / (abs(feat1[f]) + 1e-9)) * 100) for f in all_feat_keys]
sep_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

sep_names  = [d[0] for d in sep_diffs]
sep_values = [d[1] for d in sep_diffs]
sep_colors = ['tomato' if v > 0 else 'royalblue' for v in sep_values]

fig5, ax5 = plt.subplots(figsize=(11, max(8, len(sep_names) * 0.38)))
ax5.barh(range(len(sep_names)), sep_values, color=sep_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
ax5.set_yticks(range(len(sep_names)))
ax5.set_yticklabels(sep_names, fontsize=8)
ax5.axvline(0, color='black', lw=1)
ax5.set_xlabel("Δ% (unruhig vs. ruhig)", fontsize=10)
ax5.set_title("Feature-Separierbarkeit – sortiert nach |Δ%|", fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

for i, val in enumerate(sep_values):
    offset = 30 if val >= 0 else -30
    ha = 'left' if val >= 0 else 'right'
    ax5.text(val + offset, i, f'{val:+.0f}%', va='center', ha=ha, fontsize=7)

plt.tight_layout()
plt.savefig(f"{PFAD}/airpods_separierbarkeit.png", dpi=150, bbox_inches='tight')
print("Separierbarkeits-Plot gespeichert: airpods_separierbarkeit.png")


# ─────────────────────────────────────────
# 8. ZUSAMMENFASSUNG
# ─────────────────────────────────────────
print("\n" + "="*60)
print("ZUSAMMENFASSUNG")
print("="*60)
for f in ['magnitude_mean', 'magnitude_std', 'stillness_ratio', 'pitch_range', 'yaw_range']:
    v1, v2 = feat1[f], feat2[f]
    diff = ((v2 - v1) / (abs(v1) + 1e-9)) * 100
    print(f"{f:25s}  ruhig={v1:.4f}  unruhig={v2:.4f}  Δ={diff:+.1f}%")
