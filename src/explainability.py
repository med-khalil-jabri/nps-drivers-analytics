"""explainability.py
====================
SHAP-based explainability layer for the Phase 5 NPS pipeline.

We use :func:`shap.TreeExplainer` because LightGBM's gradient-boosted
trees admit an *exact* polynomial-time Shapley computation via TreeSHAP
(Lundberg et al., 2018).  No sampling, no kernel approximation: the
returned attributions are provably the unique allocation of the model's
log-odds margin satisfying local accuracy, missingness, and consistency.

Two distinct interpretation surfaces live here:

* **Global** — ``plot_global_shap`` renders both the bar (mean ``|SHAP|``)
  and beeswarm (per-sample directional) summaries.  The bar plot answers
  *which features matter most*, the beeswarm answers *how they matter
  and for whom*.
* **Individual** — ``plot_individual_waterfall`` renders the standard
  SHAP waterfall for a single row, decomposing that customer's predicted
  log-odds into per-feature contributions starting from the model's base
  value.  This is the per-customer artefact that downstream CRM workflows
  consume.

All outputs operate in the model's native log-odds (margin) space,
which is the right scale for interpreting tree splits without the
sigmoid's diminishing-returns distortion near 0 and 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tree_explainer(model: LGBMClassifier) -> shap.TreeExplainer:
    """Construct a :class:`shap.TreeExplainer` for a fitted LightGBM model.

    Parameters
    ----------
    model:
        Fitted ``LGBMClassifier`` (binary or multi-class).

    Returns
    -------
    shap.TreeExplainer
        TreeSHAP explainer operating in the model's log-odds (margin) space.
    """
    return shap.TreeExplainer(model)


def plot_global_shap(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    max_display: int = 15,
) -> shap.Explanation:
    """Render the bar (magnitude) and beeswarm (directional) global summaries.

    Parameters
    ----------
    explainer:
        A SHAP TreeExplainer produced by :func:`get_tree_explainer`.
    X:
        Feature matrix on which to compute SHAP values.  Columns must
        align with the model's training schema.
    max_display:
        Number of top features to display on each plot.  Defaults to 15.

    Returns
    -------
    shap.Explanation
        The full SHAP Explanation object — handy if the caller wants to
        post-process the values without recomputing them.
    """
    shap_values = explainer(X)
    pos_values = _select_positive_class(shap_values)

    # Bar plot — global feature importance ranked by mean(|SHAP|).
    plt.figure()
    shap.summary_plot(
        pos_values,
        X,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.gcf().suptitle(
        "Global Feature Importance — mean(|SHAP|)", y=1.02, fontsize=12
    )
    plt.tight_layout()
    plt.show()

    # Beeswarm — directional impact per customer.
    plt.figure()
    shap.summary_plot(
        pos_values,
        X,
        max_display=max_display,
        show=False,
    )
    plt.gcf().suptitle(
        "Directional Impact — SHAP beeswarm (log-odds)", y=1.02, fontsize=12
    )
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_individual_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    row_index: int,
) -> None:
    """Render the SHAP waterfall plot for a single customer.

    The waterfall decomposes the customer's predicted log-odds into a sum
    of per-feature contributions starting from the model's base value
    ``E[f(X)]``.  Positive bars push the prediction toward Detractor;
    negative bars push it toward Non-Detractor.

    Parameters
    ----------
    explainer:
        A SHAP TreeExplainer produced by :func:`get_tree_explainer`.
    X:
        Feature matrix containing the row at integer position ``row_index``.
    row_index:
        Integer position (not label) of the row to explain.
    """
    if not 0 <= row_index < len(X):
        raise IndexError(
            f"row_index {row_index} out of bounds for X of length {len(X)}."
        )

    shap_values = explainer(X)
    pos_values = _select_positive_class(shap_values)

    plt.figure()
    shap.plots.waterfall(pos_values[row_index], show=False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _select_positive_class(shap_values: shap.Explanation) -> shap.Explanation:
    """Reduce a SHAP Explanation to the positive class for binary models.

    Different SHAP versions return either:
      * a 2-D Explanation with shape ``(n_rows, n_features)`` — already
        the log-odds for the positive class, or
      * a 3-D Explanation with shape ``(n_rows, n_features, n_classes)`` —
        one slice per class.

    For binary classifiers we always want the *Detractor* (positive)
    slice.  This helper normalises both shapes so downstream plotting
    code can be uniform.
    """
    if shap_values.values.ndim == 3 and shap_values.values.shape[-1] == 2:
        return shap_values[..., 1]
    return shap_values
