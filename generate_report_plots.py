"""Generate all 4 report plots: γ vs Bo, γ vs Wo, MAE vs Bo, MAE vs Wo"""
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from src.physics.young_laplace import PendantDropSolver
from src.models.data_preparation import prepare_training_data
from src.utils.file_io import load_dataset

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = keras.models.load_model('models/pendant_drop_model_best.h5', compile=False)

# ── Load test dataset ────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset('data/synthetic/class2_1000.pkl')
print(f"  {len(dataset)} shapes loaded")

# ── Run predictions ──────────────────────────────────────────────────────────
print("Running predictions...")
Bo_true, Wo_vals, gamma_pred, Bo_errors = [], [], [], []

rho = 1000    # kg/m³
g = 9.81      # m/s²
a = 0.0017    # capillary radius in meters (1.7mm)

for shape in dataset:
    X, y = prepare_training_data([shape])
    y_pred = model.predict(X, verbose=0)

    Bo_t = y[0, 0]
    Bo_p = y_pred[0, 0]
    gamma = (rho * g * a**2) / Bo_p * 1000  # mN/m

    Bo_true.append(Bo_t)
    Wo_vals.append(shape['Wo'])
    gamma_pred.append(gamma)
    Bo_errors.append(abs(Bo_t - Bo_p))

Bo_true    = np.array(Bo_true)
Wo_vals    = np.array(Wo_vals)
gamma_pred = np.array(gamma_pred)
Bo_errors  = np.array(Bo_errors)

# ── Filter valid Wo (0 to 1.5) ───────────────────────────────────────────────
valid_wo          = (Wo_vals >= 0) & (Wo_vals <= 1.5)
Wo_filtered       = Wo_vals[valid_wo]
gamma_wo_filtered = gamma_pred[valid_wo]
errors_wo_filtered = Bo_errors[valid_wo]
print(f"  {valid_wo.sum()} / {len(Wo_vals)} shapes have valid Wo in [0, 1.5]")

# ── Bin helper ────────────────────────────────────────────────────────────────
def bin_mean(x, y, n_bins=20):
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, stds = [], [], []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if mask.sum() > 0:
            centers.append((bins[i] + bins[i+1]) / 2)
            means.append(y[mask].mean())
            stds.append(y[mask].std())
    return np.array(centers), np.array(means), np.array(stds)

# ── Plot 5: γ vs Bo ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Bo_true, gamma_pred, alpha=0.3, s=8, color='steelblue', label='Predictions')
bx, by, bs = bin_mean(Bo_true, gamma_pred)
ax.plot(bx, by, 'r-', linewidth=2, label='Bin mean')
ax.set_xlabel('Bond Number (Bo)', fontsize=13)
ax.set_ylabel('Predicted Surface Tension γ (mN/m)', fontsize=13)
ax.set_title('Predicted Surface Tension vs. Bond Number', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/gamma_vs_Bo.png', dpi=150, bbox_inches='tight')
print("Saved: gamma_vs_Bo.png")
plt.close()

# ── Plot 6: γ vs Wo (filtered) ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Wo_filtered, gamma_wo_filtered, alpha=0.3, s=8, color='darkorange', label='Predictions')
bx, by, bs = bin_mean(Wo_filtered, gamma_wo_filtered)
ax.plot(bx, by, 'r-', linewidth=2, label='Bin mean')
ax.set_xlabel('Worthington Number (Wo)', fontsize=13)
ax.set_ylabel('Predicted Surface Tension γ (mN/m)', fontsize=13)
ax.set_title('Predicted Surface Tension vs. Worthington Number', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/gamma_vs_Wo.png', dpi=150, bbox_inches='tight')
print("Saved: gamma_vs_Wo.png")
plt.close()

# ── Plot 7: MAE vs Bo ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bx, by, bs = bin_mean(Bo_true, Bo_errors)
ax.plot(bx, by, 'b-o', linewidth=2, markersize=5, label='ML model (this work)')
ax.fill_between(bx, by - bs, by + bs, alpha=0.2, color='blue')
ax.axhline(y=by.mean(), color='gray', linestyle='--', alpha=0.6, label=f'Mean MAE = {by.mean():.3f}')
ax.set_xlabel('Bond Number (Bo)', fontsize=13)
ax.set_ylabel('Mean Absolute Error (Bo)', fontsize=13)
ax.set_title('MAE vs. Bond Number\n(ML model maintains uniform error across all Bo)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/MAE_vs_Bo.png', dpi=150, bbox_inches='tight')
print("Saved: MAE_vs_Bo.png")
plt.close()

# ── Plot 8: MAE vs Wo (filtered) ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bx, by, bs = bin_mean(Wo_filtered, errors_wo_filtered)
ax.plot(bx, by, 'g-o', linewidth=2, markersize=5, label='ML model (this work)')
ax.fill_between(bx, by - bs, by + bs, alpha=0.2, color='green')
ax.axhline(y=by.mean(), color='gray', linestyle='--', alpha=0.6, label=f'Mean MAE = {by.mean():.3f}')
ax.set_xlabel('Worthington Number (Wo)', fontsize=13)
ax.set_ylabel('Mean Absolute Error (Bo)', fontsize=13)
ax.set_title('MAE vs. Worthington Number\n(MAE decreases with Wo — model performs best on strongly deformed drops)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/MAE_vs_Wo.png', dpi=150, bbox_inches='tight')
print("Saved: MAE_vs_Wo.png")
plt.close()

print("\nDone! All 4 plots saved to results/figures/")