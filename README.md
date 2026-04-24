# XAI Portfolio — LIME · SHAP · PDP

> A professional portfolio project demonstrating three key Explainable AI (XAI) 
> techniques applied to a real medical classification task.

**Live site →** `https://<your-username>.github.io/xai-portfolio`

---

## What's inside

| Technique | Type | Library |
|-----------|------|---------|
| **LIME** | Local | `lime` |
| **SHAP** | Local + Global | `shap` |
| **PDP** | Global | `scikit-learn` |

**Dataset:** UCI Breast Cancer Wisconsin (569 samples, 30 features)  
**Model:** Gradient Boosting Classifier — 95.6% test accuracy

---

## Project structure

```
xai-portfolio/
├── docs/                   ← GitHub Pages root
│   ├── index.html          ← Portfolio website
│   └── assets/             ← Generated plots (PNG)
│       ├── hero_banner.png
│       ├── lime_plot.png
│       ├── shap_beeswarm.png
│       ├── shap_bar.png
│       └── pdp_grid.png
├── notebooks/
│   └── xai_analysis.ipynb  ← Jupyter walkthrough
├── generate_plots.py       ← Reproduces all assets
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/ai-study-vault/xai-portfolio.git
cd xai-portfolio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Regenerate all plots
python generate_plots.py

# 4. Open the site locally
open docs/index.html
```

---

## Deploying to GitHub Pages

1. Push this repo to GitHub
2. Go to **Settings → Pages**
3. Source: `Deploy from a branch`
4. Branch: `main` / `docs` folder
5. Save — your site is live in ~60 seconds

---

## Key concepts

### LIME
Perturbs the input around a single data point and fits a local linear model 
to approximate the black-box model's behaviour nearby. Best for **explaining 
individual predictions** to end users or auditors.

### SHAP
Uses Shapley values from cooperative game theory to assign each feature a fair 
contribution to the prediction. Satisfies desirable axioms (efficiency, symmetry, 
dummy player). Works for both local and global explanations.

### PDP
Marginalises over all other features to show the average effect of one feature 
on the model output. Best for **communicating global feature effects** to 
non-technical stakeholders.

---

## References

- Ribeiro et al. (2016) — ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- Lundberg & Lee (2017) — [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- Friedman (2001) — [Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

---

*Built with Python, scikit-learn, LIME, SHAP, and GitHub Pages.*
