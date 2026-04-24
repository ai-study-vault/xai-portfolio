"""
XAI Portfolio - Generate all explanation plots
Produces: LIME, SHAP, and PDP visualizations for the GitHub Pages site
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
import lime
import lime.lime_tabular
import shap

# ── Style ──────────────────────────────────────────────────────────────────────
BG      = "#0d0f14"
PANEL   = "#161b22"
BORDER  = "#30363d"
ACCENT1 = "#58a6ff"   # blue
ACCENT2 = "#3fb950"   # green
ACCENT3 = "#f78166"   # red/orange
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

def apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(BG)
    for ax in ax_list:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

# ── Data & Model ───────────────────────────────────────────────────────────────
data   = load_breast_cancer()
X, y   = data.data, data.target
feat   = data.feature_names

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
model.fit(X_tr, y_tr)
print(f"Model accuracy: {model.score(X_te, y_te):.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LIME  – local explanation for one test instance
# ══════════════════════════════════════════════════════════════════════════════
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_tr,
    feature_names=feat,
    class_names=data.target_names,
    mode='classification',
    random_state=42
)

idx  = 5   # interesting test instance
exp  = explainer_lime.explain_instance(X_te[idx], model.predict_proba, num_features=10)
items = exp.as_list()
items.sort(key=lambda x: x[1])

labels = [i[0] for i in items]
vals   = [i[1] for i in items]
colors = [ACCENT2 if v > 0 else ACCENT3 for v in vals]

fig, ax = plt.subplots(figsize=(9, 5))
apply_dark_style(fig, [ax])
bars = ax.barh(labels, vals, color=colors, edgecolor=BORDER, linewidth=0.6)
ax.axvline(0, color=MUTED, linewidth=0.8, linestyle='--')
ax.set_title("LIME — Local Explanation (Single Prediction)", fontsize=13, pad=14, color=TEXT)
ax.set_xlabel("Feature Weight  (→ Malignant)", color=MUTED, fontsize=9)
patch_pos = mpatches.Patch(color=ACCENT2, label='Pushes → Benign')
patch_neg = mpatches.Patch(color=ACCENT3, label='Pushes → Malignant')
ax.legend(handles=[patch_pos, patch_neg], facecolor=PANEL, edgecolor=BORDER,
          labelcolor=TEXT, fontsize=8)
pred_prob = model.predict_proba(X_te[idx].reshape(1,-1))[0]
ax.text(0.98, 0.02,
        f"P(Benign) = {pred_prob[1]:.2f}   P(Malignant) = {pred_prob[0]:.2f}",
        transform=ax.transAxes, ha='right', va='bottom',
        color=MUTED, fontsize=8)
plt.tight_layout()
plt.savefig("docs/assets/lime_plot.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ LIME plot saved")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAP  – beeswarm summary + waterfall for one instance
# ══════════════════════════════════════════════════════════════════════════════
explainer_shap = shap.TreeExplainer(model)
shap_vals = explainer_shap.shap_values(X_te)

# Use class-1 (Benign) SHAP values
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

# --- Beeswarm / dot summary ---
fig, ax = plt.subplots(figsize=(9, 6))
apply_dark_style(fig, [ax])

# Manual beeswarm-style dot plot
feat_importance = np.abs(sv).mean(0)
top_idx = np.argsort(feat_importance)[-10:]
sv_top  = sv[:, top_idx]
X_top   = X_te[:, top_idx]

# Normalize feature values for colouring
for i, fi in enumerate(top_idx):
    x_norm = (X_te[:, fi] - X_te[:, fi].min()) / (np.ptp(X_te[:, fi]) + 1e-9)
    jitter = np.random.RandomState(fi).uniform(-0.3, 0.3, len(sv_top))
    sc = ax.scatter(sv_top[:, i], np.full(len(sv_top), i) + jitter,
                    c=x_norm, cmap='coolwarm', alpha=0.6, s=8, linewidths=0)

ax.set_yticks(range(10))
ax.set_yticklabels([feat[j] for j in top_idx], fontsize=8.5)
ax.axvline(0, color=MUTED, linewidth=0.8, linestyle='--')
ax.set_title("SHAP — Global Feature Impact (Beeswarm)", fontsize=13, pad=14, color=TEXT)
ax.set_xlabel("SHAP Value  (impact on model output)", color=MUTED, fontsize=9)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=ax, pad=0.01)
cbar.set_label("Feature value (low → high)", color=MUTED, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)
cbar.outline.set_edgecolor(BORDER)
plt.tight_layout()
plt.savefig("docs/assets/shap_beeswarm.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ SHAP beeswarm saved")

# --- Bar summary ---
fig, ax = plt.subplots(figsize=(9, 5))
apply_dark_style(fig, [ax])
order   = np.argsort(feat_importance)[-12:]
imp_top = feat_importance[order]
ax.barh([feat[j] for j in order], imp_top,
        color=ACCENT1, edgecolor=BORDER, linewidth=0.5, alpha=0.85)
ax.set_title("SHAP — Mean |SHAP| Feature Importance", fontsize=13, pad=14, color=TEXT)
ax.set_xlabel("Mean |SHAP value|", color=MUTED, fontsize=9)
plt.tight_layout()
plt.savefig("docs/assets/shap_bar.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ SHAP bar saved")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PDP  – 2 × 2 grid for top-4 features
# ══════════════════════════════════════════════════════════════════════════════
top4 = np.argsort(feat_importance)[-4:][::-1]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
apply_dark_style(fig, axes.flat)

for ax, fi in zip(axes.flat, top4):
    feat_vals   = np.linspace(X_te[:, fi].min(), X_te[:, fi].max(), 80)
    X_copy      = X_te.copy()
    pdp_preds   = []
    for v in feat_vals:
        X_copy[:, fi] = v
        pdp_preds.append(model.predict_proba(X_copy)[:, 1].mean())

    ax.plot(feat_vals, pdp_preds, color=ACCENT1, linewidth=2)
    ax.fill_between(feat_vals, pdp_preds, alpha=0.12, color=ACCENT1)
    ax.set_title(feat[fi], fontsize=9, color=TEXT)
    ax.set_xlabel("Feature value", color=MUTED, fontsize=8)
    ax.set_ylabel("P(Benign)", color=MUTED, fontsize=8)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=MUTED, linewidth=0.6, linestyle=':')

fig.suptitle("Partial Dependence Plots — Top 4 Features", fontsize=13, color=TEXT, y=1.01)
plt.tight_layout()
plt.savefig("docs/assets/pdp_grid.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ PDP grid saved")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Hero banner – XAI comparison overview
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
apply_dark_style(fig, axes)

# LIME mini
mini_vals  = [v for _, v in items]
mini_cols  = [ACCENT2 if v > 0 else ACCENT3 for v in mini_vals]
axes[0].barh(range(len(mini_vals)), mini_vals, color=mini_cols, edgecolor='none')
axes[0].axvline(0, color=MUTED, lw=0.8, ls='--')
axes[0].set_title("LIME", fontsize=15, color=ACCENT1, pad=10)
axes[0].set_yticks([])
axes[0].set_xlabel("Local weights", color=MUTED, fontsize=8)

# SHAP mini bar
axes[1].barh([feat[j] for j in order[-6:]], imp_top[-6:],
             color=ACCENT1, edgecolor='none', alpha=0.85)
axes[1].set_title("SHAP", fontsize=15, color=ACCENT1, pad=10)
axes[1].set_xlabel("Mean |SHAP|", color=MUTED, fontsize=8)

# PDP mini
fi = top4[0]
axes[2].plot(feat_vals, pdp_preds, color=ACCENT2, linewidth=2)
axes[2].fill_between(feat_vals, pdp_preds, alpha=0.15, color=ACCENT2)
axes[2].set_title("PDP", fontsize=15, color=ACCENT1, pad=10)
axes[2].set_xlabel(feat[fi], color=MUTED, fontsize=8)
axes[2].set_ylabel("P(Benign)", color=MUTED, fontsize=8)
axes[2].set_ylim(0, 1)

fig.suptitle("Explainable AI  ·  Three Lenses on Model Behaviour",
             fontsize=13, color=TEXT, y=1.02)
plt.tight_layout()
plt.savefig("docs/assets/hero_banner.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Hero banner saved")
print("\nAll assets generated successfully!")
