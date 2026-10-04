import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. HETEROGENEOUS DATA GENERATION
# ============================================================
day_labels = ["Aug 18", "Aug 19", "Aug 20", "Aug 21", "Aug 22", "Aug 23", 
              "Aug 25", "Aug 26", "Aug 27", "Aug 28", "Aug 29"]

np.random.seed(42)
t = np.linspace(0, 1, 24)
base_signal = 10 + 5 * np.sin(t * np.pi) 

full_day_data = np.tile(base_signal, (len(day_labels), 1))
full_day_data += np.random.normal(0, 0.1, full_day_data.shape)

# Inject Drifts
full_day_data[0:3] += 0.6   # Aug 18-20 Early Bias
full_day_data[8]   += 1.8   # Aug 27 Massive Outlier
full_day_data[9:11] += 0.8  # Aug 28-29 Late Bias

win_idx = [10, 12] # 10:00 - 12:00 window
X_win = full_day_data[:, win_idx]

# Ground Truth (Pseudo-truth from known stable core Aug 21-26)
stable_idx = [3, 4, 5, 6, 7] 
gt_full_day = np.mean(full_day_data[stable_idx], axis=0)

# ============================================================
# 2. EVALUATION & BENCHMARKING ENGINE
# ============================================================
results = []

def run_benchmark(name, logic_func):
    """Measures MAE, RMSE, R2, Time, and Peak Memory usage."""
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Execute Strategy
    preds = logic_func()
    
    end_time = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate Accuracy Metrics
    maes = [mean_absolute_error(gt_full_day, p) for p in preds]
    rmses = [np.sqrt(mean_squared_error(gt_full_day, p)) for p in preds]
    r2s = [r2_score(gt_full_day, p) for p in preds]
    
    results.append({
        "Strategy": name, 
        "MAE": np.mean(maes), 
        "RMSE": np.mean(rmses), 
        "R2": np.mean(r2s),
        "Time (ms)": (end_time - start_time) * 1000,
        "Memory (KB)": peak_mem / 1024
    })

# --- Strategy Definitions ---
def pure_local(): return full_day_data

def vanilla_fedavg(): return [np.mean(full_day_data, axis=0)] * len(day_labels)

def single_fedcal():
    ref_static = X_win[0] 
    return [full_day_data[i] + np.mean(ref_static - X_win[i]) for i in range(len(day_labels))]

def recursive_adaptive():
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_win)
    Z_w = linkage(scaled, method='ward')
    ids_w = fcluster(Z_w, t=1.5, criterion='distance')
    maj_w = pd.Series(ids_w).mode()[0]
    ref_w = np.mean(X_win[np.where(ids_w == maj_w)[0]], axis=0)
    return [full_day_data[i] + np.mean(ref_w - X_win[i]) for i in range(len(day_labels))]

def recursive_vectorized():
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_win)
    Z_w = linkage(scaled, method='ward')
    ids_w = fcluster(Z_w, t=1.5, criterion='distance')
    maj_w = pd.Series(ids_w).mode()[0]
    ref_w = np.mean(X_win[np.where(ids_w == maj_w)[0]], axis=0)
    
    offsets = np.mean(ref_w - X_win, axis=1)
    return full_day_data + offsets[:, np.newaxis]

# Execute Benchmarks
run_benchmark("Pure Local", pure_local)
run_benchmark("Vanilla FedAvg", vanilla_fedavg)
run_benchmark("Single FedCal", single_fedcal)
run_benchmark("Recursive Adaptive", recursive_adaptive)
run_benchmark("Recursive Vectorized", recursive_vectorized)

summary_df = pd.DataFrame(results)

# ============================================================
# 3. CONSOLE OUTPUTS (Table, Sliding Window, Audit Report)
# ============================================================

# --- Benchmark Table ---
print("\n" + "="*105)
print(f"{'Strategy':<25} | {'RMSE':<8} | {'R2':<8} | {'Time (ms)':<10} | {'Memory (KB)':<12}")
print("-" * 105)
for _, row in summary_df.iterrows():
    print(f"{row['Strategy']:<25} | {row['RMSE']:<8.4f} | {row['R2']:<8.4f} | {row['Time (ms)']:<10.3f} | {row['Memory (KB)']:<12.2f}")
print("="*105 + "\n")

# --- Sliding Window Tracker ---
print("-" * 50)
window_size = 5
for i in range(len(day_labels)):
    start_idx = max(0, i - window_size + 1)
    current_window = day_labels[start_idx : i + 1]
    print(f"{day_labels[i]:<10} | {current_window}")

# --- Scalable Federated Audit (Final Window) ---
# Isolate the final 5 days
final_idx = slice(-5, None)
X_final = X_win[final_idx]
days_final = day_labels[final_idx]

# Run isolated clustering for the audit
scaler = StandardScaler()
scaled_final = scaler.fit_transform(X_final)
Z_final = linkage(scaled_final, method='ward')
ids_final = fcluster(Z_final, t=1.5, criterion='distance')
maj_final = pd.Series(ids_final).mode()[0]
ref_final = np.mean(X_final[ids_final == maj_final], axis=0)

# Calculate exact offsets applied during this window
audit_offsets = np.mean(ref_final - X_final, axis=1)

print("\n============================================================")
print("      SCALABLE FEDERATED AUDIT (Final Window)")
print("============================================================")
print(f"{'Date':<15} {'Cluster':<8} {'Status':<12} {'Offset'}")
print("-" * 55)
for i in range(len(days_final)):
    status = "✅ STABLE" if ids_final[i] == maj_final else "🚨 DRIFT"
    print(f"{days_final[i]:<15} {ids_final[i]:<8} {status:<12} {audit_offsets[i]:>6.2f}")
print()

# ============================================================
# 4. COMPARATIVE PLOT (Grouped Bar Format)
# ============================================================
fig, ax1 = plt.subplots(figsize=(14, 7))

x = np.arange(len(summary_df['Strategy']))
width = 0.35 

# RMSE Bars
color_rmse = 'black'
ax1.set_xlabel('Calibration Strategy', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE', color=color_rmse, fontsize=12, fontweight='bold')
ax1.bar(x - width/2, summary_df['RMSE'], width, color=color_rmse, alpha=0.6, label='RMSE')
ax1.tick_params(axis='y', labelcolor=color_rmse)

# R2 Bars
ax2 = ax1.twinx()
color_r2 = 'blue'
ax2.set_ylabel('R2 Score', color=color_r2, fontsize=12, fontweight='bold')
ax2.bar(x + width/2, summary_df['R2'], width, color=color_r2, alpha=0.6, label='R2 Score')
ax2.tick_params(axis='y', labelcolor=color_r2)
ax2.set_ylim(-1, 1.1) 

ax1.set_xticks(x)
ax1.set_xticklabels(summary_df['Strategy'], rotation=15, ha="right")
ax1.grid(axis='y', linestyle='--', alpha=0.1)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower right')

plt.tight_layout()
plt.show()