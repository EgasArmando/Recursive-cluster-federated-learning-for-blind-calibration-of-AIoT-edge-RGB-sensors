import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

class ScalableFederatedCalibrator:
    def __init__(self, drift_threshold=1.5, window_size=5):
        """
        drift_threshold: Distance to trigger a cluster split.
        window_size: MAX history kept locally (Scalability: O(1) memory).
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.buffer_data = []  
        self.buffer_labels = []
        self.global_reference = None  # Simulates the "Federated Averaged" model
        self.current_report = pd.DataFrame()

    def process_new_client(self, day_label, window_values):
        # 1. SCALABILITY: Maintain a Sliding Window (Memory Management)
        if len(self.buffer_data) >= self.window_size:
            self.buffer_data.pop(0)
            self.buffer_labels.pop(0)
            
        self.buffer_data.append(window_values)
        self.buffer_labels.append(day_label)
        
        if len(self.buffer_data) < 2:
            return None 

        X = np.array(self.buffer_data)
        
        # 2. PRIVACY: Work in Standardized Latent Space
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. COMPUTATION: Hierarchical Re-arrangement (Ward's Method)
        Z = linkage(X_scaled, method='ward')
        cluster_ids = fcluster(Z, t=self.drift_threshold, criterion='distance')
        
        # 4. FEDERATED LOGIC: Identify the 'Honest Majority'
        unique_ids, counts = np.unique(cluster_ids, return_counts=True)
        majority_id = unique_ids[np.argmax(counts)]
        
        # 5. GLOBAL AGGREGATION: Update the Global Reference Centroid
        # In a real system, this 'mean' would be sent to/from a central server
        reference_indices = np.where(cluster_ids == majority_id)[0]
        self.global_reference = np.mean(X[reference_indices], axis=0)
        
        # 6. LOCAL CALIBRATION: Generate Report
        report_data = []
        for i, label in enumerate(self.buffer_labels):
            is_stable = cluster_ids[i] == majority_id
            # Offset calculation happens locally
            offset = np.mean(self.global_reference - X[i])
            report_data.append({
                "Date": label,
                "Cluster": int(cluster_ids[i]),
                "Status": "✅ STABLE" if is_stable else "🚨 DRIFT",
                "Offset": round(offset, 4)
            })
            
        self.current_report = pd.DataFrame(report_data).set_index("Date")
        return Z

# ============================================================
# SCALABLE EXECUTION SIMULATION
# ============================================================

# We use a window_size of 5 to show scalability in action
engine = ScalableFederatedCalibrator(drift_threshold=1.5, window_size=5)

raw_values = {
    "Aug 18": [2.8, 3.0], "Aug 19": [2.7, 2.9], "Aug 20": [2.7, 2.8],
    "Aug 21": [2.2, 2.4], "Aug 22": [2.3, 2.5], "Aug 23": [2.2, 2.4],
    "Aug 25": [2.3, 2.5], "Aug 26": [2.2, 2.4], "Aug 27": [3.8, 4.5], # Huge Drift
    "Aug 28": [2.9, 3.1], "Aug 29": [3.0, 3.2]
}

print(f"{'Day':<10} | {'Active Window (Max 5 days)'}")
print("-" * 50)

for day, values in raw_values.items():
    engine.process_new_client(day, values)
    active_days = engine.buffer_labels
    print(f"{day:<10} | {active_days}")

# Final Audit
print("\n" + "="*60)
print("      SCALABLE FEDERATED AUDIT (Final Window)")
print("="*60)
print(engine.current_report)