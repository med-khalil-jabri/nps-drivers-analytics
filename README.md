# NPS Drivers Analytics — Deutsche Bahn Case Study

End-to-end machine learning pipeline for **NPS driver analysis**: predict Detractor risk from operational CRM features, explain predictions with SHAP, and simulate a naive “zero delay” counterfactual on active Live travellers.

**Business question:** Which drivers move satisfaction—and if a top driver (e.g. delays) were eliminated, how much would Detractor risk fall, and what levers matter next?

**Approach:** EAV → wide pivot (Polars) → LightGBM (multi-class + binary Detractor) → TreeSHAP → naive counterfactual intervention (`src/causal.py`).

The full narrative, EDA, and results live in the main notebook (Sections 1–7).

---

## Project structure

```
nps-drivers-analytics/
├── data/                          # Input CSVs (not in git — see Setup)
├── notebooks/
│   └── 01_nps_hybrid_analysis.ipynb   # Main analysis (run this)
├── src/
│   ├── data_prep.py               # EAV pivot, type casting, Live two-pass helpers
│   ├── features.py                # Encoding, collinearity filter, severe-delay flag
│   ├── train.py                   # Temporal split, feature matrix, final fit
│   ├── optimization.py            # Optuna + TimeSeriesSplit HPO
│   ├── predict.py                 # Evaluation, F2 threshold, Expected NPS
│   ├── explainability.py          # TreeSHAP global & waterfall plots
│   └── causal.py                  # Naive counterfactual simulation
├── models/                        # Saved encoder/schema (optional; notebook can retrain)
├── main.py                        # Placeholder entrypoint
├── pyproject.toml                 # Dependencies (managed by uv)
└── uv.lock
```

### Notebook

| File | Role |
|------|------|
| `notebooks/01_nps_hybrid_analysis.ipynb` | Single executable deliverable: EDA → features → training → SHAP → Live scoring → counterfactuals → discussion |

### Source modules (imported by the notebook)

| Module | Role |
|--------|------|
| `data_prep` | Parse driver definitions; `load_and_pivot_data`; active-user extraction for large Live files |
| `features` | Multi-categorical encoding, semantic/Spearman deduplication, `is_severe_delay_10m_plus` |
| `train` | Out-of-time split, `get_feature_matrix`, `train_final_model` |
| `optimization` | Optuna hyperparameter search with temporal CV |
| `predict` | Metrics, F2 threshold tuning, Expected NPS |
| `explainability` | `get_tree_explainer`, global SHAP, individual waterfalls |
| `causal` | `simulate_naive_counterfactual` |

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package & environment manager)
- Python **3.9+** (project pins `3.9` in `.python-version`)
- Enough disk/RAM for the case-study files (Train ~2 GB; Live ~42 GB). The notebook **samples** the Live file by default to stay within typical laptop limits.

### 1. Clone and install dependencies

From the repository root:

```bash
cd nps-drivers-analytics
uv sync
```

This creates `.venv/` and installs dependencies from `pyproject.toml` (Polars, LightGBM, SHAP, Optuna, Jupyter, etc.).

### 2. Add data files

Place the three case-study CSV files in a `data/` folder at the repo root:

```
data/
├── case_study_deutschebahn_driver_definitions.csv
├── case_study_deutschebahn_drivers_flat_users_train.csv
└── case_study_deutschebahn_drivers_flat_users_live.csv
```

| File | Description |
|------|-------------|
| `case_study_deutschebahn_driver_definitions.csv` | Driver names and types (`numeric`, `categorical`, `multi_categorical`) |
| `case_study_deutschebahn_drivers_flat_users_train.csv` | Train EAV: surveyed customers with NPS labels |
| `case_study_deutschebahn_drivers_flat_users_live.csv` | Live EAV: full CRM population (very large) |

The `data/` directory is gitignored; you must obtain these files from the case-study provider.

### 3. Run the notebook

**Option A — Jupyter in the browser**

```bash
uv run jupyter notebook notebooks/01_nps_hybrid_analysis.ipynb
```

**Option B — VS Code / Cursor**

1. Open `notebooks/01_nps_hybrid_analysis.ipynb`.
2. Select the interpreter: **`.venv/bin/python`** (the environment created by `uv sync`).
3. **Run All** (or run cells top to bottom).

The first code cell adds the repo root to `sys.path` so `from src....` imports work when the notebook is run from `notebooks/`.

### Runtime notes

- **Train** loads fully in memory after pivot.
- **Live** is loaded in **row-limited samples** in the notebook (e.g. lazy scan with `n_rows` caps). Adjust `LIVE_EAV_SAMPLE` / `LIVE_SAMPLE_ROWS` in the notebook if you need a smaller or larger Live subset.
- Section 6 (counterfactuals) subsamples up to 100k active rows for SHAP speed (`N_CF_SAMPLE`).
- A full **Run All** on a machine with limited RAM may take a long time or fail on the uncapped Live pivot; reduce Live sample sizes first if needed.

---

## Tech stack

Polars · pandas · LightGBM · Optuna · scikit-learn · SHAP · Matplotlib · Seaborn · Jupyter
