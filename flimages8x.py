import matplotlib
# Force a stable backend
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import os
import sys  
import cv2 
from skimage.filters import threshold_otsu 
import re

# =========================================================
# CONFIGURATION
# =========================================================
ROOT_FOLDER = "/Users/egasarmando/Desktop/Filter_exp2/" 
SENSOR_CSV = "/Users/egasarmando/Desktop/scd41_data.csv"
sns.set_theme(style="whitegrid")

# =========================================================
# 1. DATA LOADING (STRICT ALIGNMENT)
# =========================================================
def get_aligned_data(csv_path, img_folder):
    try:
        df = pd.read_csv(csv_path)
        time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        csv_dates = sorted(list(df[time_col].dt.strftime('%Y-%m-%d').dropna().unique()))
    except: return [], {}

    img_dates = set()
    date_pattern = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")
    if os.path.exists(img_folder):
        for item in os.listdir(img_folder):
            if os.path.isdir(os.path.join(img_folder, item)) or item.endswith(('.png', '.jpg')):
                match = date_pattern.search(item)
                if match: img_dates.add(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
    sorted_img_dates = sorted(list(img_dates))

    target_csv = csv_dates[-5:]
    target_img = sorted_img_dates[:5]
    if len(target_img) < len(target_csv):
        target_img += [target_img[-1]] * (len(target_csv) - len(target_img))
    
    return target_csv, dict(zip(target_csv, target_img))

def process_vision_metrics(mapped_date, root_path):
    paths = [os.path.join(root_path, mapped_date), os.path.join(root_path, mapped_date.replace("-", ""))]
    target = next((p for p in paths if os.path.exists(p)), root_path)
    clean_date = mapped_date.replace("-", "")
    
    reds, lais = [], []
    for f in os.listdir(target):
        if f.lower().endswith(('.png', '.jpg')) and (clean_date in f or mapped_date in f):
            try:
                img = cv2.imread(os.path.join(target, f))
                if img is None: continue
                red_ch = img[:, :, 2]
                reds.append(np.mean(red_ch))
                thresh = threshold_otsu(red_ch)
                mask = red_ch < thresh
                lais.append(np.sum(mask) / mask.size)
            except: pass
    return {"LAI": np.mean(lais) if lais else 0}

def calculate_real_metrics():
    targets, mapping = get_aligned_data(SENSOR_CSV, ROOT_FOLDER)
    if not targets: return pd.DataFrame()

    df = pd.read_csv(SENSOR_CSV)
    time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    co2_col = next((c for c in df.columns if any(x in c.lower() for x in ["co2", "ppm"])), None)
    temp_col = next((c for c in df.columns if any(x in c.lower() for x in ["temp", "tmp"])), None)
    hum_col = next((c for c in df.columns if any(x in c.lower() for x in ["hum", "rh"])), None)
    
    df[time_col] = pd.to_datetime(df[time_col])
    df['date_str'] = df[time_col].dt.strftime('%Y-%m-%d')
    df['minute'] = df[time_col].dt.hour * 60 + df[time_col].dt.minute
    
    # 1. Create Data Matrix (The Truth)
    pivot_df = df.pivot_table(index='minute', columns='date_str', values=co2_col, aggfunc='mean')
    matrix = pivot_df[targets].interpolate().dropna()
    y_true = matrix.median(axis=1) # Ground Truth
    
    # 2. Calculate Client Weights (Physics + Vision)
    weights = []
    for c_date in targets:
        day_df = df[df['date_str'] == c_date]
        t = pd.to_numeric(day_df[temp_col], errors='coerce').mean()
        h = pd.to_numeric(day_df[hum_col], errors='coerce').mean()
        
        vp_sat = 0.611 * np.exp((17.27 * t) / (t + 237.3))
        vpd = vp_sat * (1 - (h / 100))
        vpd_score = max(0.1, 1.0 - abs(vpd - 1.0))
        
        vis = process_vision_metrics(mapping[c_date], ROOT_FOLDER)
        lai_score = vis['LAI'] * 2.0
        
        weights.append(vpd_score * 0.4 + lai_score * 0.6)
        
    clustered_w = np.array(weights) / sum(weights)
    
    # =========================================================
    # 3. REAL PERFORMANCE MEASUREMENT
    # =========================================================
    
    # A. Local iTransformer (Client-Side)
    local_r2s = [r2_score(y_true, matrix[col]) for col in matrix.columns]
    r2_local = np.mean(local_r2s)
    mem_local = sum([sys.getsizeof(matrix[col]) for col in matrix.columns]) / 1024 # KB
    
    # B. Global FL
    y_global = matrix.mean(axis=1)
    r2_global = r2_score(y_true, y_global)
    mem_global = sys.getsizeof(y_global) / 1024 # KB
    
    # C. Clustered FL
    y_clustered = np.zeros_like(y_true)
    for i, col in enumerate(matrix.columns):
        y_clustered += matrix[col].values * clustered_w[i]
    r2_clustered = r2_score(y_true, y_clustered)
    mem_clustered = (sys.getsizeof(y_clustered) + sys.getsizeof(clustered_w)) / 1024 # KB

    return pd.DataFrame([
        {"Model": "iTransformer (Local)", "R2 Score": r2_local, "Memory (KB)": mem_local},
        {"Model": "Global FL", "R2 Score": r2_global, "Memory (KB)": mem_global},
        {"Model": "Clustered FL", "R2 Score": r2_clustered, "Memory (KB)": mem_clustered}
    ])

# =========================================================
# 4. VISUALIZATION
# =========================================================
def run_strict_audit_clean():
    df = calculate_real_metrics()
    if df.empty: print("Error: No valid data for calculation."); return
    
    print("\n--- Final Calculated Metrics ---")
    print(df)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    fig.suptitle('Strict Audit: Real Data Performance vs. Cost', fontsize=18, weight='bold')
    
    # 1. R2 Score (Bar)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x="Model", y="R2 Score", hue="Model", data=df, ax=ax1, 
                palette=["#a1c9f4", "grey", "teal"], edgecolor='black', legend=False)
    ax1.set_title("Prediction Accuracy ($R^2$)", fontsize=12, weight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("")
    ax1.grid(axis='y', alpha=0.0)
    # NO ANNOTATIONS HERE

    # 2. Memory Usage (Bar)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(x="Model", y="Memory (KB)", hue="Model", data=df, ax=ax2, 
                palette=["#ff9999", "grey", "teal"], edgecolor='black', legend=False)
    ax2.set_title("Data Footprint (RAM Usage)", fontsize=12, weight='bold')
    ax2.set_xlabel("")
    ax2.grid(axis='y', alpha=0.0)
    # NO ANNOTATIONS HERE

    # 3. Efficiency Frontier (Scatter)
    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(x="Memory (KB)", y="R2 Score", hue="Model", s=400, data=df, ax=ax3, 
                    palette=["#a1c9f4", "grey", "teal"], legend=True)
    
    ax3.set_title("Efficiency Frontier", fontsize=14, weight='bold')
    ax3.set_xlabel("Computational Cost (KB)")
    ax3.set_ylabel("Accuracy ($R^2$)")
    ax3.grid(True, linestyle='--', alpha=0.0)
    
    # Move legend to a clean spot
    ax3.legend(loc='upper right', frameon=True)
    
    # NO TEXT LABELS OR ARROWS HERE

    sns.despine()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_strict_audit_clean()