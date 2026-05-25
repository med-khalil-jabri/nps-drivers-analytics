"""predict.py
==============
Inference, evaluation, and business-metric computation for Phase 3.

Two concerns live here:

1. **Diagnostic evaluation** — classification report and confusion matrix
   on the OOT test set.  These confirm the model behaves reasonably
   across all three NPS classes.
2. **Business-metric translation** — converting the classifier's calibrated
   probabilities into the continuous **Expected NPS** score that Deutsche
   Bahn's CX team can act on:

       Expected NPS  =  ( P(Promoter) − P(Detractor) )  ×  100

   This is a per-customer continuous score on the standard -100 → +100
   NPS scale, suitable for ranking, segmentation, and downstream causal
   uplift modelling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLASS_LABELS: dict[int, str] = {
    0: "Detractor",
    1: "Passive",
    2: "Promoter",
}

_DETRACTOR_CLASS: int = 0
_PROMOTER_CLASS: int = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print a classification report and plot the confusion matrix.

    Parameters
    ----------
    model:
        Fitted ``LGBMClassifier``.
    X_test:
        OOT test feature matrix.
    y_test:
        OOT test target vector.
    """
    y_pred = model.predict(X_test)

    print("=" * 64)
    print("Classification Report — OOT Test Set")
    print("=" * 64)
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(_CLASS_LABELS.keys()),
            target_names=list(_CLASS_LABELS.values()),
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix (rows = true, cols = predicted)
    cm = confusion_matrix(y_test, y_pred, labels=list(_CLASS_LABELS.keys()))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    annotations = np.array([
        [f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=list(_CLASS_LABELS.values()),
        yticklabels=list(_CLASS_LABELS.values()),
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix — OOT Test Set\nAbsolute counts + row-normalised %", pad=10)
    plt.tight_layout()
    plt.show()


def calculate_expected_nps(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Compute the Expected NPS continuous score per customer.

    .. math::

        \\text{Expected NPS}_i  =  (P(\\text{Promoter}_i) - P(\\text{Detractor}_i)) \\times 100

    The standard NPS definition aggregates over a population; here we apply
    the same arithmetic at the individual level so that downstream uplift /
    causal modelling can operate on a continuous outcome.

    Parameters
    ----------
    model:
        Fitted ``LGBMClassifier``.  Must have been trained with all three
        NPS classes ``{0, 1, 2}`` represented.
    X_test:
        Feature matrix on which to score.

    Returns
    -------
    pd.Series
        Continuous Expected NPS scores on the ``-100`` → ``+100`` scale,
        indexed identically to ``X_test``.
    """
    proba = model.predict_proba(X_test)

    # Locate the columns of model.classes_ corresponding to Detractor and Promoter.
    classes = list(model.classes_)
    if _DETRACTOR_CLASS not in classes or _PROMOTER_CLASS not in classes:
        raise ValueError(
            f"Model classes {classes} do not contain both Detractor "
            f"({_DETRACTOR_CLASS}) and Promoter ({_PROMOTER_CLASS})."
        )

    detractor_idx = classes.index(_DETRACTOR_CLASS)
    promoter_idx = classes.index(_PROMOTER_CLASS)

    expected_nps = (proba[:, promoter_idx] - proba[:, detractor_idx]) * 100.0

    return pd.Series(expected_nps, index=X_test.index, name="expected_nps")
