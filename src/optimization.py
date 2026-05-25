"""optimization.py
==================
Optuna-driven hyperparameter optimisation for the Phase 3 NPS classifier.

Design notes
------------
* **TimeSeriesSplit, not KFold.**  Customer NPS responses are dated.  Random
  K-fold would let the model peek into the future of the same survey wave it
  is trying to predict — a textbook temporal-leakage trap.  We use
  ``TimeSeriesSplit(n_splits=3)`` which produces expanding-window folds
  ordered by ``npsDate``.
* **``multi_logloss`` objective, not Macro-F1.**  Our downstream KPI is
  Expected NPS ``= P(Promoter) − P(Detractor)``, which is a function of the
  *raw probabilities*.  Optimising a threshold-dependent metric (F1) would
  pick a model with strong rankings but poorly-calibrated probabilities.
  Log-loss is the proper scoring rule for probability calibration.
* **No ``class_weight='balanced'``.**  Re-weighting samples to equalise
  classes shifts the predicted probabilities away from the empirical class
  prior, which would silently bias the Expected NPS computation downward
  (Detractors get up-weighted → predicted ``P(Detractor)`` becomes too high).
  Class re-weighting is a tool for threshold-based F1 / recall, not for
  probability-calibrated regression-of-probabilities pipelines.
"""

from __future__ import annotations

from typing import Any

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

_NPS_CLASS_LABELS: tuple[int, int, int] = (0, 1, 2)  # Detractor / Passive / Promoter
_N_SPLITS: int = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_lgbm_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
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
        Target vector containing the three NPS classes ``{0, 1, 2}``.
    n_trials:
        Number of Optuna trials.  Defaults to ``20``, which empirically
        converges well on this driver space.
    random_state:
        Seed for both the Optuna sampler and LightGBM.

    Returns
    -------
    optuna.Study
        Completed study.  Best parameters live in ``study.best_params`` and
        the best objective value (mean fold log-loss) in ``study.best_value``.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y length mismatch: {len(X)} vs {len(y)}."
        )

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

            model = LGBMClassifier(
                objective="multiclass",
                num_class=len(_NPS_CLASS_LABELS),
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
                **params,
            )
            model.fit(X_fold_train, y_fold_train)

            y_pred_proba = model.predict_proba(X_fold_val)
            fold_loss = log_loss(
                y_fold_val,
                y_pred_proba,
                labels=list(_NPS_CLASS_LABELS),
            )
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
