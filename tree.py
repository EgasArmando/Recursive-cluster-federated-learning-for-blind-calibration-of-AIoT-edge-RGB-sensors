import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================
# 1. DATA PREPARATION (Direct from Audit Report)
# ============================================================
data = {
    "Date": ["Aug 18", "Aug 19", "Aug 20", "Aug 21", "Aug 22", "Aug 23", "Aug 25", "Aug 26", "Aug 27", "Aug 28", "Aug 29"],
    "Cluster": [2, 2, 2, 1, 1, 1, 1, 1, 3, 2, 2],
    "Status": ["DRIFT", "DRIFT", "DRIFT", "STABLE", "STABLE", "STABLE", "STABLE", "STABLE", "DRIFT", "DRIFT", "DRIFT"],
    "Offset": [-0.56, -0.46, -0.41, 0.04, -0.06, 0.04, -0.06, 0.04, -1.81, -0.66, -0.76]
}
df = pd.DataFrame(data)

# ============================================================
# 2. TREE ARCHITECTURE GENERATION
# ============================================================
def plot_journal_discovery_tree(df):
    G = nx.Graph()
    root = "Blind\ncalibration\nCPFL\n(Recursive)"
    G.add_node(root, level=0)

    # Color mapping to match scientific standards
    # Blue: Consensus/Stable, Orange: Moderate Bias, Red: Severe Drift
    c_map = {1: '#3498db', 2: '#e67e22', 3: '#e74c3c'}
    
    pos = {root: (0, 10)}
    clusters = sorted(df['Cluster'].unique())
    
    for i, cluster in enumerate(clusters):
        cluster_node = f"Cluster C{cluster}"
        G.add_edge(root, cluster_node)
        
        # Horizontal spacing for clusters
        cluster_x = (i - (len(clusters)-1)/2) * 12
        pos[cluster_node] = (cluster_x, 7)
        
        # Filter days belonging to this cluster
        cluster_days = df[df['Cluster'] == cluster]
        for j, row in enumerate(cluster_days.itertuples()):
            G.add_edge(cluster_node, row.Date)
            # Spread leaf nodes horizontally
            leaf_x = cluster_x + (j - (len(cluster_days)-1)/2) * 2.2
            pos[row.Date] = (leaf_x, 4)
            
            # Bottom row: Audit Labels (Status & Offset)
            audit_label = f"{row.Status}\n{row.Offset:+.2f}"
            G.add_edge(row.Date, audit_label)
            pos[audit_label] = (leaf_x, 1)

    # Drawing the Graph
    fig, ax = plt.subplots(figsize=(18, 9))
    node_colors = []
    
    for node in G.nodes():
        if node == root:
            node_colors.append('#2c3e50') # Aggregator color
        elif "Cluster" in str(node):
            c_id = int(node[-1])
            node_colors.append(c_map[c_id])
        elif any(m in str(node) for m in ["Aug"]):
            c_id = df[df['Date'] == node]['Cluster'].values[0]
            node_colors.append(c_map[c_id])
        else:
            node_colors.append('#ecf0f1') # Bottom label color

    nx.draw(G, pos, with_labels=True, node_size=4200, node_color=node_colors, 
            font_size=8, font_weight='bold', edge_color='#bdc3c7', alpha=0.9)

    # Professional Annotation Legend
    textstr = (
        "C1: Stable (Consensus)\n"
        "C2: Systematic Bias (Early/Late Drift)\n"
        "C3: Critical Outlier (Severe Hardware Drift)"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#2c3e50')
    ax.text(0.05, 0.1, textstr, transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', bbox=props, fontweight='regular')

    #plt.title("Figure X: Hierarchical Discovery Tree for Federated Sensor Alignment", 
    #          fontsize=18, pad=30, fontweight='bold', family='serif')
    plt.tight_layout()
    plt.show()

# ============================================================
# 3. EXECUTION
# ============================================================
if __name__ == "__main__":
    plot_journal_discovery_tree(df)