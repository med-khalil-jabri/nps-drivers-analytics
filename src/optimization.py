"""optimization.py
==================
Optuna-driven hyperparameter optimisation for the NPS classifier.

Supports both **multi-class** mode (calibrated probabilities for Expected
NPS) and **binary** mode (Detractor vs. Non-Detractor with class re-balancing
for recall on the actionable churn-risk segment).

Design notes
------------
* **TimeSeriesSplit, not KFold.**  Customer NPS responses are dated.  Random
  K-fold would let the model peek into the future of the same survey wave it
  is trying to predict — a textbook temporal-leakage trap.  We use
  ``TimeSeriesSplit(n_splits=3)`` which produces expanding-window folds
  ordered by ``npsDate``.

* **Log-loss as the objective in both modes.**  Log-loss is the proper
  scoring rule for probability calibration regardless of class cardinality.
  In multi-class mode we use ``sklearn.metrics.log_loss`` over the three
  NPS labels; in binary mode we use the same function with ``labels=[0, 1]``,
  which is mathematically equivalent to binary cross-entropy.

* **Class weighting policy depends on the mode.**  Multi-class mode
  intentionally avoids ``class_weight='balanced'`` because Expected NPS is
  a probability-arithmetic KPI and re-weighting biases ``P(Detractor)``.
  Binary mode, by contrast, is optimised for Detractor recall — the
  calibration concern no longer applies because there is no
  ``P(Promoter) − P(Detractor)`` arithmetic, so we re-enable balanced
  weighting to surface the minority class at the trees' early splits.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

# Keep Optuna chatter out of notebook outputs.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MULTI_LABELS: tuple[int, int, int] = (0, 1, 2)   # Detractor / Passive / Promoter
_BINARY_LABELS: tuple[int, int] = (0, 1)          # Non-Detractor / Detractor
_N_SPLITS: int = 3

TargetMode = Literal["multi", "binary"]
_VALID_MODES: frozenset[str] = frozenset({"multi", "binary"})


def _validate_mode(target_mode: str) -> None:
    if target_mode not in _VALID_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(_VALID_MODES)}; got {target_mode!r}."
        )


def _build_lgbm(
    target_mode: TargetMode,
    params: dict[str, Any],
    random_state: int,
) -> LGBMClassifier:
    """Construct a mode-appropriate ``LGBMClassifier`` from the trial params."""
    if target_mode == "binary":
        return LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(_MULTI_LABELS),
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_lgbm_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
    target_mode: TargetMode = "multi",
    random_state: int = 42,
) -> optuna.Study:
    """Run an Optuna study for the temporal LightGBM NPS classifier.

    The feature matrix ``X`` and target ``y`` are assumed to be **already
    sorted by ``npsDate``**.  ``TimeSeriesSplit`` operates on this ordering;
    if it is broken, fold separation no longer reflects calendar order.

    Parameters
    ----------
    X:
        Feature matrix with categorical columns already cast to pandas
        ``category`` dtype (LightGBM auto-detects them).
    y:
        Target vector.
        - ``target_mode="multi"`` → values in ``{0, 1, 2}``.
        - ``target_mode="binary"`` → values in ``{0, 1}`` (Detractor = 1).
    n_trials:
        Number of Optuna trials.  Defaults to ``20``.
    target_mode:
        ``"multi"`` (default) optimises a 3-class log-loss with no class
        re-weighting; ``"binary"`` optimises binary cross-entropy with
        ``class_weight='balanced'`` for Detractor recall.
    random_state:
        Seed for both the Optuna sampler and LightGBM.

    Returns
    -------
    optuna.Study
        Completed study.  Best parameters live in ``study.best_params`` and
        the best objective value (mean fold log-loss) in ``study.best_value``.
    """
    _validate_mode(target_mode)
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")

    labels = list(_BINARY_LABELS) if target_mode == "binary" else list(_MULTI_LABELS)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        splitter = TimeSeriesSplit(n_splits=_N_SPLITS)
        fold_losses: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            model = _build_lgbm(target_mode, params, random_state)
            model.fit(X_fold_train, y_fold_train)

            y_pred_proba = model.predict_proba(X_fold_val)
            fold_loss = log_loss(y_fold_val, y_pred_proba, labels=labels)
            fold_losses.append(fold_loss)

            # Prune aggressively if the running mean drifts up between folds.
            trial.report(float(np.mean(fold_losses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study
