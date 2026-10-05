from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score

from lib.model_utils import load_model_config

NUM_LAYERS = 29

# 1. Load Data
print("Loading cached layer vectors...")
config_path = Path(__file__).resolve().parent / "config" / "config.yml"
model_id, _, _ = load_model_config(config_path)
data = np.load(f"./data/{model_id}_all_layers.npz")
all_benign_states = data["benign_states"]
all_malicious_states = data["malicious_states"]

calib_size = 50
auc_scores = []

# 2. Sweep Calculation
print("Calculating RepE ROC-AUC across all layers...")
for layer_idx in range(NUM_LAYERS):
    X_benign_layer = all_benign_states[:, layer_idx, :]
    X_malicious_layer = all_malicious_states[:, layer_idx, :]
    
    mu_benign = np.mean(X_benign_layer[:calib_size], axis=0)
    mu_malicious = np.mean(X_malicious_layer[:calib_size], axis=0)
    
    harm_vector = mu_malicious - mu_benign
    harm_vector = harm_vector / np.linalg.norm(harm_vector)
    
    X_test = np.vstack([X_benign_layer[calib_size:], X_malicious_layer[calib_size:]])
    y_true = np.array([0] * (len(all_benign_states) - calib_size) + [1] * (len(all_malicious_states) - calib_size))
    
    scores = np.dot(X_test, harm_vector)
    auc = roc_auc_score(y_true, scores)
    auc_scores.append(auc)
    
    print(f"Layer {layer_idx:02d} ROC-AUC: {auc:.4f}")

# 3. Visualization
plt.figure(figsize=(12, 6))
plt.plot(range(NUM_LAYERS), auc_scores, marker='o', linewidth=2, color='#2563eb')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guessing (0.5)')

peak_layer = np.argmax(auc_scores)
peak_auc = auc_scores[peak_layer]
plt.plot(peak_layer, peak_auc, marker='*', color='red', markersize=15, label=f'Peak: Layer {peak_layer} ({peak_auc:.3f})')

plt.title(f'{model_id}: Malicious Intent Detection by Layer')
plt.xlabel('Hidden Layer Index (0 = Embeddings)')
plt.ylabel('RepE ROC-AUC Score')
plt.ylim(0.4, 1.05)
plt.xticks(range(0, NUM_LAYERS, 2))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

os.makedirs("./data", exist_ok=True)
plot_path = f"./data/{model_id}_layer_trajectory.png"
plt.savefig(plot_path, dpi=300)
print(f"\n✅ Plot saved to {plot_path}")