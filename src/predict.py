"""predict.py
==============
Inference, evaluation, and business-metric computation.

Three concerns live here:

1. **Diagnostic evaluation** — classification report and confusion matrix
   on the OOT test set, supporting both multi-class and binary models and
   a configurable decision threshold for the binary case.
2. **Decision-threshold calibration** — ``optimize_decision_threshold``
   plots the Precision / Recall / F-beta surface across all achievable
   cut-points and returns the threshold that maximises the F-beta score.
   Default ``beta=2`` weights recall twice as highly as precision, which
   is the right prior for a churn-prevention CRM workflow (missing a
   Detractor is more costly than a spurious retention touchpoint).
3. **Business-metric translation** — converting the multi-class model's
   calibrated probabilities into the continuous **Expected NPS** score:

       Expected NPS  =  ( P(Promoter) − P(Detractor) )  ×  100
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Friendly labels for the multi-class NPS targets.  Binary mode uses a
# separate map because the integer codes have a different semantic meaning:
# in binary mode 1 = Detractor (positive class), 0 = Non-Detractor.
_MULTI_CLASS_LABELS: dict[int, str] = {
    0: "Detractor",
    1: "Passive",
    2: "Promoter",
}
_BINARY_CLASS_LABELS: dict[int, str] = {
    0: "Non-Detractor",
    1: "Detractor",
}

_DETRACTOR_CLASS: int = 0
_PROMOTER_CLASS: int = 2


def _resolve_class_labels(class_codes: list[int]) -> dict[int, str]:
    """Pick a label map that matches the model's ``classes_`` array.

    Falls back to a literal-int label if the codes don't match either of the
    known schemas, so the function never crashes on an unexpected target.
    """
    if set(class_codes) == set(_MULTI_CLASS_LABELS.keys()):
        return _MULTI_CLASS_LABELS
    if set(class_codes) == set(_BINARY_CLASS_LABELS.keys()):
        return _BINARY_CLASS_LABELS
    return {code: f"Class {code}" for code in class_codes}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> None:
    """Print a classification report and plot the confusion matrix.

    Works for both multi-class (three NPS segments) and binary
    (Detractor vs. Non-Detractor) models — labels are derived from
    ``model.classes_`` rather than hard-coded.

    For binary models, predictions are derived from ``predict_proba``
    and the supplied ``threshold`` so that a business-calibrated cut-point
    (e.g. one optimised for F2-score) can be applied directly without
    re-training the model.

    Parameters
    ----------
    model:
        Fitted ``LGBMClassifier``.
    X_test:
        OOT test feature matrix.
    y_test:
        OOT test target vector.
    threshold:
        Decision threshold applied to the positive-class probability in
        **binary mode only**.  Defaults to ``0.5``.  Has no effect on
        multi-class models (``model.classes_`` contains more than two codes)
        because those do not expose a single scalar cut-point.
    """
    class_codes: list[int] = [int(c) for c in model.classes_]
    label_map = _resolve_class_labels(class_codes)
    label_names = [label_map[c] for c in class_codes]
    is_binary = len(class_codes) == 2

    if is_binary:
        # Locate the column index for the positive class (Detractor = 1).
        pos_class = max(class_codes)
        pos_idx = list(model.classes_).index(pos_class)
        probs = model.predict_proba(X_test)[:, pos_idx]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        threshold = 0.5  # meaningless for multi-class; reset for display clarity

    mode_tag = "Binary" if is_binary else "Multi-class"
    threshold_tag = f" | threshold = {threshold:.3f}" if is_binary else ""

    print("=" * 64)
    print(f"Classification Report — {mode_tag}{threshold_tag} — OOT Test Set")
    print("=" * 64)
    print(
        classification_report(
            y_test,
            y_pred,
            labels=class_codes,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix (rows = true, cols = predicted)
    cm = confusion_matrix(y_test, y_pred, labels=class_codes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    annotations = np.array([
        [f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(6.5, 5.5) if is_binary else (7, 5.5))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    title_suffix = f"\nthreshold = {threshold:.3f}" if is_binary else ""
    ax.set_title(
        f"Confusion Matrix — {mode_tag} — OOT Test Set\n"
        f"Absolute counts + row-normalised %{title_suffix}",
        pad=10,
    )
    plt.tight_layout()
    plt.show()


def optimize_decision_threshold(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    beta: float = 2.0,
) -> float:
    """Find the decision threshold that maximises the F-beta score.

    In a churn-prevention setting, a missed Detractor (False Negative) is
    substantially more costly than a spurious retention offer (False
    Positive).  The F-beta score generalises F1 to reflect this asymmetry:
    ``beta > 1`` up-weights recall relative to precision, so the optimal
    threshold shifts below ``0.5`` to aggressively surface at-risk customers.

    The function plots a presentation-ready curve of Precision, Recall, and
    F-beta against every achievable threshold, marking the optimum with a
    vertical dashed line.

    Parameters
    ----------
    model:
        Fitted binary ``LGBMClassifier`` (``model.classes_ == [0, 1]``).
    X_test:
        OOT test feature matrix.
    y_test:
        OOT test target vector (binary: 1 = Detractor, 0 = Non-Detractor).
    beta:
        The beta parameter of the F-beta score.  ``beta=2.0`` (default)
        weights recall twice as highly as precision.

    Returns
    -------
    float
        The threshold in ``[0, 1]`` that maximises the F-beta score on the
        supplied test set.

    Raises
    ------
    ValueError
        If the model is not binary (does not have exactly two classes).
    """
    class_codes: list[int] = [int(c) for c in model.classes_]
    if len(class_codes) != 2:
        raise ValueError(
            f"optimize_decision_threshold requires a binary model; "
            f"got classes {class_codes}."
        )

    pos_class = max(class_codes)
    pos_idx = list(model.classes_).index(pos_class)
    probs: np.ndarray = model.predict_proba(X_test)[:, pos_idx]

    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    # precision_recall_curve appends a sentinel (precision=1, recall=0) at the
    # end with no corresponding threshold entry.  Align all three arrays so
    # they share the same index space.
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    beta_sq = beta ** 2
    denom = (beta_sq * precisions) + recalls
    # Guard against 0/0 at degenerate thresholds (all predicted positive or all
    # predicted negative).
    fbeta: np.ndarray = np.where(
        denom > 0,
        (1 + beta_sq) * (precisions * recalls) / denom,
        0.0,
    )

    best_idx: int = int(np.argmax(fbeta))
    optimal_threshold: float = float(thresholds[best_idx])

    # ── Visualisation ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, precisions, label="Precision", color="#4C72B0", linewidth=2)
    ax.plot(thresholds, recalls,    label="Recall",    color="#C44E52", linewidth=2)
    ax.plot(
        thresholds, fbeta,
        label=f"F{beta:g}-Score",
        color="#55A868",
        linewidth=2.5,
        linestyle="--",
    )

    ax.axvline(
        optimal_threshold,
        color="#DD8452",
        linewidth=1.8,
        linestyle=":",
        label=f"Optimal threshold = {optimal_threshold:.3f}",
    )
    ax.scatter(
        [optimal_threshold],
        [fbeta[best_idx]],
        color="#DD8452",
        s=80,
        zorder=5,
    )
    ax.annotate(
        f"F{beta:g} = {fbeta[best_idx]:.3f}",
        xy=(optimal_threshold, fbeta[best_idx]),
        xytext=(optimal_threshold + 0.03, fbeta[best_idx] - 0.06),
        fontsize=9,
        color="#DD8452",
        arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2),
    )

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Precision / Recall / F{beta:g}-Score vs. Decision Threshold\n"
        f"Detractor vs. Non-Detractor — OOT Test Set",
        pad=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

    return optimal_threshold


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
