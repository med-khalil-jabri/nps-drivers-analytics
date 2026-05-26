"""causal.py
==========
Causal counterfactual simulation for the Deutsche Bahn Actionable NPS pipeline.

**Phase 6 — Naive Counterfactual**

The function in this module implements a "zero-out" intervention: all columns
listed in ``intervention_cols`` are forced to zero, and the model is re-scored
on this modified matrix to estimate the Average Treatment Effect (ATE).

⚠️  **Out-of-Distribution (OOD) Warning — Positivity Assumption Violation**

This is a *naive* counterfactual, not a causal one.  Forcing average delay
duration to zero while leaving correlated secondary features (trip duration,
occupancy level, cancellation rate, ticket price) at their observed values
creates synthetic customer profiles that are *out of the support of the
training distribution*.  The model has never seen a passenger with a zero
average delay and a non-zero cancellation percentage — the interaction term
is unrepresented in the joint distribution.

In formal causal terms, the **Positivity Assumption** is violated: the
probability of receiving the "zero-delay treatment" conditional on the
observed covariates is not bounded away from zero for every covariate
profile in the data.

Consequence: the model's predicted probability in the counterfactual regime
is an *extrapolation*, not an interpolation.  The ATE reported here should
be interpreted as a directional signal — "delays are the dominant lever" —
rather than a precise causal estimate.

A proper causal analysis would require either:
* A structural causal model (SCM) with explicit intervention do(delay = 0)
  propagated through all downstream variables (realising that zero delay
  implies lower trip duration variance, different cancellation patterns, etc.),
* A propensity-score matched comparison between low-delay and high-delay
  passengers on the same route/time-of-day/ticket-price cell, or
* A difference-in-differences design exploiting natural experiments (e.g.
  strike/disruption events that exogenously shift delay distributions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_naive_counterfactual(
    model: Any,
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    intervention_cols: list[str],
) -> dict[str, Any]:
    """Estimate the NPS impact of zeroing out a set of operational drivers.

    For each column listed in ``intervention_cols``, the function sets every
    value to ``0`` (the "no-problem" state) and re-scores the population
    with the fitted binary LightGBM model.  SHAP values are recomputed on
    the counterfactual matrix to reveal which drivers become the new priority
    levers in the hypothetical world.

    ⚠️ See module docstring for the OOD / Positivity Assumption caveat.

    Parameters
    ----------
    model:
        Fitted ``LGBMClassifier`` (binary, Detractor = 1).
    explainer:
        :class:`shap.TreeExplainer` produced by ``get_tree_explainer(model)``.
    X:
        Observed feature matrix for the population to simulate on.  Columns
        must match the model's training schema.  Typically a sample of
        ``X_live_processed``.
    intervention_cols:
        List of column names to force to ``0``.  Columns absent from ``X``
        are silently skipped.

    Returns
    -------
    dict with keys:
        ``baseline_rate``   — mean predicted Detractor probability on ``X``.
        ``cf_rate``         — mean predicted Detractor probability on ``X_cf``.
        ``ate``             — ``baseline_rate − cf_rate`` (Average Treatment
                              Effect; positive means intervention reduces risk).
        ``X_cf``            — the modified counterfactual feature matrix.
        ``new_shap_values`` — :class:`shap.Explanation` computed on ``X_cf``.
    """
    # ── Baseline scoring ────────────────────────────────────────────────────
    baseline_proba = model.predict_proba(X)[:, 1]
    baseline_rate: float = float(baseline_proba.mean())

    # ── Build counterfactual matrix ─────────────────────────────────────────
    X_cf: pd.DataFrame = X.copy()
    skipped = []
    for col in intervention_cols:
        if col not in X_cf.columns:
            skipped.append(col)
            continue
        # Cast the zero value to match the column's dtype to avoid pandas
        # SettingWithCopyWarning and dtype promotion issues.
        dtype = X_cf[col].dtype
        if hasattr(dtype, "numpy_dtype"):
            # pandas ExtensionDtype (Int8, Float32, …)
            zero = pd.array([0], dtype=dtype)[0]
        else:
            zero = dtype.type(0) if hasattr(dtype, "type") else 0
        X_cf[col] = zero

    if skipped:
        import warnings
        warnings.warn(
            f"simulate_naive_counterfactual: the following intervention "
            f"columns were not found in X and were skipped: {skipped}",
            stacklevel=2,
        )

    # ── Counterfactual scoring ───────────────────────────────────────────────
    cf_proba = model.predict_proba(X_cf)[:, 1]
    cf_rate: float = float(cf_proba.mean())

    # ── Re-compute SHAP on the counterfactual world ─────────────────────────
    new_shap_values: shap.Explanation = explainer(X_cf)

    return {
        "baseline_rate":   baseline_rate,
        "cf_rate":         cf_rate,
        "ate":             baseline_rate - cf_rate,
        "X_cf":            X_cf,
        "new_shap_values": new_shap_values,
    }
