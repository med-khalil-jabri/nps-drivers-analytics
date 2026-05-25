"""train.py
=============
Training utilities for the NPS classifier — supports both multi-class and
binary deployment modes via a single ``target_mode`` switch.

This module owns:

* The **Out-of-Time (OOT) split** that splits ``wide_train_fe`` chronologically
  by ``npsDate`` — never randomly.
* The **feature-matrix builder** that converts the Polars feature frame into
  a pandas DataFrame with the proper ``category`` dtypes that LightGBM
  consumes natively.  In ``binary`` mode the original 3-class target
  ``{Detractor=0, Passive=1, Promoter=2}`` is collapsed to
  ``{Non-Detractor=0, Detractor=1}``.
* The **final-model trainer** that fits a single ``LGBMClassifier`` with the
  Optuna-selected hyperparameters on the full training fold.  Multi-class
  mode keeps probabilities calibrated for Expected NPS; binary mode uses
  ``class_weight='balanced'`` to maximise Detractor recall.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_META_COLS: tuple[str, ...] = (
    "sourceId",
    "sourceUniqueId",
    "npsDate",
    "npsScore",
    "nps_class",
)

_TARGET_COL: str = "nps_class"
_NUM_CLASSES: int = 3

# Original multi-class label for Detractors — promoted to the positive class
# (1) when in binary mode, because Detractor identification is the actionable
# business question.
_DETRACTOR_LABEL: int = 0

TargetMode = Literal["multi", "binary"]
_VALID_MODES: frozenset[str] = frozenset({"multi", "binary"})


def _validate_mode(target_mode: str) -> None:
    if target_mode not in _VALID_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(_VALID_MODES)}; got {target_mode!r}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    df: pl.DataFrame,
    date_col: str = "npsDate",
    test_size: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a Polars DataFrame chronologically into Train and OOT Test.

    The oldest ``1 - test_size`` fraction becomes Train; the most recent
    ``test_size`` becomes Test.  This is the only split that mirrors a
    production deployment scenario, where the model is trained on history
    and scored on customers whose surveys arrived later.

    Parameters
    ----------
    df:
        Wide-format Polars DataFrame post Phase-2 feature engineering.
    date_col:
        Column name carrying the survey date.  Must be a ``pl.Date`` or
        sortable type.
    test_size:
        Fraction of rows to assign to the OOT test set (must be in
        ``(0, 1)``).

    Returns
    -------
    (train_df, test_df)
        Two Polars DataFrames, each preserving the chronological ordering.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}.")
    if date_col not in df.columns:
        raise KeyError(f"date column '{date_col}' not found in DataFrame.")

    sorted_df = df.sort(date_col)
    n_total = sorted_df.height
    n_train = int(n_total * (1.0 - test_size))

    train_df = sorted_df.head(n_train)
    test_df = sorted_df.slice(n_train, n_total - n_train)

    return train_df, test_df


def get_feature_matrix(
    df: pl.DataFrame,
    target_mode: TargetMode = "multi",
    target_col: str = _TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a Polars feature frame into a LightGBM-ready (X, y) pair.

    Metadata columns (``sourceId``, ``sourceUniqueId``, ``npsDate``,
    ``npsScore``, ``nps_class``) are removed.  Categorical columns are
    cast to pandas ``category`` dtype so that LightGBM picks them up via
    its default ``categorical_feature='auto'`` setting.

    Parameters
    ----------
    df:
        Polars DataFrame containing both features and the target column.
    target_mode:
        - ``"multi"`` (default) — preserve the original three NPS classes
          ``{Detractor=0, Passive=1, Promoter=2}``.
        - ``"binary"`` — collapse to ``{Non-Detractor=0, Detractor=1}``,
          promoting Detractors to the positive class so that
          ``class_weight='balanced'`` and recall-oriented metrics target
          the actionable churn-risk segment.
    target_col:
        Name of the source target column.  Defaults to ``"nps_class"``.

    Returns
    -------
    (X, y)
        ``X``: pandas DataFrame of features (numeric + ``category`` dtypes).
        ``y``: pandas Series of integer class labels.
    """
    _validate_mode(target_mode)
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not found in DataFrame.")

    cols_to_drop = [c for c in _META_COLS if c in df.columns]
    X_pl = df.drop(cols_to_drop)
    X_pd: pd.DataFrame = X_pl.to_pandas()

    # Cast Polars Categorical → pandas category for native LGBM handling.
    for col in X_pl.columns:
        if isinstance(X_pl.schema[col], pl.Categorical):
            X_pd[col] = X_pd[col].astype("category")

    y_raw: pd.Series = df[target_col].to_pandas().astype("int8")

    if target_mode == "binary":
        # Detractor (original label 0) → 1  (positive class)
        # Passive / Promoter (original 1, 2) → 0  (negative class)
        y = (y_raw == _DETRACTOR_LABEL).astype("int8")
        y.name = "is_detractor"
    else:
        y = y_raw

    return X_pd, y


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
    target_mode: TargetMode = "multi",
    n_estimators: int = 600,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit the final LightGBM model on the full Train fold.

    The full-fit boosting round budget (``n_estimators``) is set higher
    than the per-trial budget used during HPO, because at this point we are
    no longer paying the cross-validation cost.

    Parameters
    ----------
    X_train:
        Training feature matrix (already pandas, ``category`` dtypes set).
    y_train:
        Training target vector.
    best_params:
        Optuna's ``study.best_params`` dict.
    target_mode:
        ``"multi"`` (calibrated probabilities, no class re-weighting) or
        ``"binary"`` (``objective='binary'`` with ``class_weight='balanced'``
        to maximise Detractor recall).
    n_estimators:
        Number of boosting rounds for the final fit.  Defaults to ``600``.
    random_state:
        Seed for deterministic training.

    Returns
    -------
    LGBMClassifier
        Fitted classifier ready for inference.
    """
    _validate_mode(target_mode)

    if target_mode == "binary":
        model = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **best_params,
        )
    else:
        model = LGBMClassifier(
            objective="multiclass",
            num_class=_NUM_CLASSES,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **best_params,
        )

    model.fit(X_train, y_train)
    return model
