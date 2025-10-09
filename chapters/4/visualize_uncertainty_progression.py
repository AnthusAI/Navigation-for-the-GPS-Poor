#!/usr/bin/env python3
"""
Create visualization showing the progression of uncertainty learning experiments.
"""
import numpy as np
import matplotlib.pyplot as plt

# Data from all 5 experiments
models = [
    {'name': 'Model 1\nweight=0.01\ndetach=Yes', 'cov': 7.2, 'coord_acc': 111, 'correlation': -0.046},
    {'name': 'Model 2\nweight=0.10\ndetach=Yes', 'cov': 6.9, 'coord_acc': 151, 'correlation': -0.093},
    {'name': 'Model 3\nweight=0.10\ndetach=No', 'cov': 7.4, 'coord_acc': 51, 'correlation': -0.185},
    {'name': 'Model 4\nweight=0.50\ndetach=No', 'cov': 8.8, 'coord_acc': 48, 'correlation': -0.298},
    {'name': 'Model 5\nweight=1.00\ndetach=No', 'cov': 12.3, 'coord_acc': 48, 'correlation': -0.210},
]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Uncertainty Estimation: 5-Model Progression', fontsize=16, fontweight='bold')

x = np.arange(len(models))
names = [m['name'] for m in models]

# Top left: Uncertainty Variation (CoV)
covs = [m['cov'] for m in models]
colors = ['red' if c < 10 else 'orange' if c < 15 else 'green' for c in covs]
bars = ax1.bar(x, covs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axhline(15, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: >15% (useful)')
ax1.axhline(10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Threshold: >10% (marginal)')
ax1.set_ylabel('Coefficient of Variation (%)', fontsize=11)
ax1.set_title('Uncertainty Variation\n(Higher = Better Confidence Discrimination)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=8)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, cov) in enumerate(zip(bars, covs)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{cov:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Top right: Coordinate Accuracy
accs = [m['coord_acc'] for m in models]
colors_acc = ['green' if a < 60 else 'lightgreen' if a < 100 else 'orange' if a < 150 else 'red' for a in accs]
bars = ax2.bar(x, accs, color=colors_acc, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: <100m')
ax2.set_ylabel('Validation Error (meters)', fontsize=11)
ax2.set_title('Coordinate Accuracy\n(Lower = Better)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=8)
ax2.invert_yaxis()  # Lower is better
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
for i, (bar, acc) in enumerate(zip(bars, accs)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, f'{acc}m',
            ha='center', va='top', fontsize=10, fontweight='bold')

# Bottom left: Error-Uncertainty Correlation
corrs = [m['correlation'] for m in models]
colors_corr = ['red' if c < -0.2 else 'orange' if c < 0 else 'lightgreen' if c < 0.3 else 'green' for c in corrs]
bars = ax3.bar(x, corrs, color=colors_corr, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=1.5)
ax3.axhline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: >0.3 (good)')
ax3.set_ylabel('Correlation Coefficient', fontsize=11)
ax3.set_title('Error-Uncertainty Correlation\n(Positive = Model predicts higher uncertainty for larger errors)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(names, fontsize=8)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(-0.4, 0.4)
for i, (bar, corr) in enumerate(zip(bars, corrs)):
    y_pos = corr + (0.02 if corr > 0 else -0.02)
    va = 'bottom' if corr > 0 else 'top'
    ax3.text(bar.get_x() + bar.get_width()/2, y_pos, f'{corr:.3f}',
            ha='center', va=va, fontsize=9, fontweight='bold')

# Bottom right: Summary table
ax4.axis('off')
summary_text = """
FINAL RESULTS (Model 5):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Coordinate Performance:
  ✅ 48m validation error (EXCELLENT)
  ✅ Best accuracy achieved across all models

Uncertainty Learning:
  ⚠️ 12.3% variation (3205m - 4907m)
  ⚠️ Limited discrimination between cases
  ❌ Negative correlation (-0.21)

Calibration:
  ✅ 70% within 1σ (expect 68%)
  ✅ Well-calibrated but conservative

Confidence-Weighted Benefit:
  ✅ 67% error reduction (1192m improvement)
  • Weighted average: 576m
  • Simple average: 1768m

VERDICT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ MARGINAL UTILITY
The uncertainty head provides limited but
measurable benefit for multi-prediction
scenarios. Conservative uncertainties work
well for safety-critical applications.

Coordinate accuracy is excellent (48m) and
confidence weighting provides 67% error
reduction when averaging predictions.

Recommendation: Use Model 5 for applications
requiring confidence estimates. For pure
coordinate accuracy, train without uncertainty.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('images/uncertainty_progression_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created: images/uncertainty_progression_analysis.png")
print("\nShows:")
print("  - Uncertainty variation across 5 training experiments")
print("  - Coordinate accuracy progression")
print("  - Error-uncertainty correlation")
print("  - Final summary and recommendations")
