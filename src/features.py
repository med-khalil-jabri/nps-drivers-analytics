"""features.py
==============
Feature engineering for the Deutsche Bahn Actionable NPS pipeline.

Only ``multi_categorical`` drivers are transformed here. Numeric and standard
categorical drivers are passed through unchanged so that LightGBM's native
NaN routing and Fisher categorical-partitioning algorithms can operate
directly on them.

Design notes
------------
* Multi-categorical values are stored as ``||``-separated token strings
  (e.g. ``"LUGGAGE||SEAT||BICYCLE"``).  The default ``CountVectorizer``
  regex tokenizer (``\\b\\w\\w+\\b``) would shatter multi-word tokens such
  as ``"FOOD & DRINKS"`` and ``"WIFI / ENTERTAINMENT"`` into pieces, so we
  override it with an explicit ``||`` split.
* The encoder is stateful (fit + transform) so the *same* top-5 vocabulary
  learned on Train can be reapplied to the Live set in Phase 5 without
  vocabulary drift.
* The output columns are dense ``UInt8`` indicators (0/1) — sparse storage
  is overkill for ≤5 dummies per source column and degrades LightGBM's
  feature-binning step.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NULL_TOKEN: str = "NO_OPTIONS"
_SEPARATOR: str = "||"
_DEFAULT_MAX_FEATURES: int = 5

# Columns that are never feature candidates for the collinearity filter.
_META_COLS: frozenset[str] = frozenset(
    {"sourceId", "sourceUniqueId", "npsDate", "npsScore", "nps_class"}
)

# Manually audited categorical→numeric redundancies.  Each key is a
# discretised categorical driver whose continuous counterpart (value) lives
# in the same dataset and dominates it in information content.
_SEMANTIC_REDUNDANCIES: dict[str, str] = {
    "transactionalTravelTripDelayDurationBracketV0Last30d":
        "transactionalTravelTripAverageDelayDurationV0Last30d",
    "transactionalLastTicketPricePositionV0":
        "transactionalLastTicketPriceInEurosV0",
    "transactionalAverageSectionOccupancyLevelV0Last30d":
        "transactionalAveragePassengerCountPerSectionV0Last30d",
}


def _split_tokens(value: str) -> list[str]:
    """Split a raw multi-categorical cell into individual tokens.

    Splits on ``||``, strips whitespace, drops empty fragments.
    """
    return [tok.strip() for tok in value.split(_SEPARATOR) if tok.strip()]


def _sanitize(token: str) -> str:
    """Make a vocabulary token safe to use as a Polars column suffix."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", token).strip("_")
    return cleaned or "UNK"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultiCategoricalEncoder:
    """Top-K token encoder for ``||``-separated multi-categorical drivers.

    For each input column the encoder:

    1. Fills nulls with the sentinel string ``"NO_OPTIONS"``.
    2. Fits a ``CountVectorizer(max_features=k)`` whose vocabulary is the
       ``k`` most frequent tokens after splitting on ``"||"``.
    3. Emits ``k`` dense ``UInt8`` indicator columns named
       ``"{source_col}__{sanitized_token}"`` and drops the raw source column.

    The same fitted vocabulary is reused at ``transform`` time so that
    Live-set inference (Phase 5) produces the exact same feature space.

    Parameters
    ----------
    columns:
        Iterable of multi-categorical column names to encode.
    max_features:
        Number of top tokens to retain per column (default ``5``).
    """

    def __init__(
        self,
        columns: Iterable[str],
        max_features: int = _DEFAULT_MAX_FEATURES,
    ) -> None:
        self.columns: list[str] = list(columns)
        self.max_features: int = max_features
        self._vectorizers: dict[str, CountVectorizer] = {}
        self._output_names: dict[str, list[str]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pl.DataFrame) -> "MultiCategoricalEncoder":
        """Learn the top-K vocabulary for each configured column."""
        for col in self.columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

            raw = (
                df.select(pl.col(col).fill_null(_NULL_TOKEN))
                .to_series()
                .to_list()
            )

            vectorizer = CountVectorizer(
                tokenizer=_split_tokens,
                lowercase=False,
                max_features=self.max_features,
                binary=True,
                token_pattern=None,
            )
            vectorizer.fit(raw)
            self._vectorizers[col] = vectorizer
            self._output_names[col] = [
                f"{col}__{_sanitize(tok)}"
                for tok in vectorizer.get_feature_names_out().tolist()
            ]

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the fitted vocabularies to ``df``.

        Returns a new DataFrame with each source column replaced by its
        ``max_features`` indicator columns.
        """
        if not self._fitted:
            raise RuntimeError("MultiCategoricalEncoder must be fit before transform.")

        new_columns: list[pl.Series] = []
        cols_to_drop: list[str] = []

        for col in self.columns:
            if col not in df.columns:
                continue

            raw = (
                df.select(pl.col(col).fill_null(_NULL_TOKEN))
                .to_series()
                .to_list()
            )
            counts = self._vectorizers[col].transform(raw)  # sparse, binary
            dense = counts.toarray().astype("uint8")

            for idx, new_name in enumerate(self._output_names[col]):
                new_columns.append(pl.Series(new_name, dense[:, idx], dtype=pl.UInt8))

            cols_to_drop.append(col)

        out = df.drop(cols_to_drop) if cols_to_drop else df
        if new_columns:
            out = out.with_columns(new_columns)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit on ``df`` and return the transformed DataFrame."""
        return self.fit(df).transform(df)

    @property
    def output_columns(self) -> dict[str, list[str]]:
        """Mapping ``{source_column: [generated_column_names]}`` after fit."""
        if not self._fitted:
            raise RuntimeError("MultiCategoricalEncoder has not been fit yet.")
        return {col: list(names) for col, names in self._output_names.items()}


# ---------------------------------------------------------------------------
# Functional convenience wrapper (matches the spec's "class or function" ask)
# ---------------------------------------------------------------------------

def encode_multi_categorical(
    df: pl.DataFrame,
    columns: Iterable[str],
    max_features: int = _DEFAULT_MAX_FEATURES,
    encoder: Optional[MultiCategoricalEncoder] = None,
) -> tuple[pl.DataFrame, MultiCategoricalEncoder]:
    """One-shot helper: fit (or reuse) and transform in a single call.

    Parameters
    ----------
    df:
        Polars DataFrame containing the multi-categorical columns.
    columns:
        Names of the multi-categorical columns to encode.
    max_features:
        Top-K vocabulary size per column.
    encoder:
        Optional pre-fitted encoder (e.g. fitted on Train, reused on Live).
        If ``None``, a fresh encoder is fit on ``df``.

    Returns
    -------
    (transformed_df, fitted_encoder)
        The encoder is returned so it can be reused for Live-set inference.
    """
    if encoder is None:
        encoder = MultiCategoricalEncoder(columns=columns, max_features=max_features)
        return encoder.fit_transform(df), encoder
    return encoder.transform(df), encoder


# ---------------------------------------------------------------------------
# Collinearity Filters
# ---------------------------------------------------------------------------

def semantic_redundancy_dropper(df: pl.DataFrame) -> pl.DataFrame:
    """Drop manually-audited categorical drivers that are strictly redundant.

    Three discretised categorical drivers in this dataset are bracketised
    versions of continuous numeric drivers that already exist in the same
    matrix:

    * ``TravelTripDelayDurationBracket`` is a binning of
      ``TravelTripAverageDelayDuration``.
    * ``LastTicketPricePosition`` is a binning of ``LastTicketPriceInEuros``.
    * ``AverageSectionOccupancyLevel`` is a binning of
      ``AveragePassengerCountPerSection``.

    Keeping both representations of the same underlying signal would cause
    LightGBM to randomly distribute split importance between them, which
    destroys SHAP attribution stability and any downstream causal Average
    Treatment Effect estimate.  We therefore drop the categorical
    bracketised columns and let the continuous version carry the signal.

    Parameters
    ----------
    df:
        Input wide-format Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        ``df`` minus the redundant categorical columns (if present).
    """
    to_drop: list[str] = [c for c in _SEMANTIC_REDUNDANCIES if c in df.columns]

    if not to_drop:
        print("Semantic redundancy filter: no audited bracket columns present.")
        return df

    print(
        f"Semantic redundancy filter — dropped {len(to_drop)} categorical "
        f"bracket(s) in favour of their continuous numeric counterparts:"
    )
    for col in to_drop:
        keeper = _SEMANTIC_REDUNDANCIES[col]
        print(f"  • {col}")
        print(f"      kept instead → {keeper}")

    return df.drop(to_drop)


def numeric_collinearity_filter(
    df: pl.DataFrame,
    threshold: float = 0.85,
    priority_features: list[str] | None = None,
) -> pl.DataFrame:
    """Drop one column from each highly-correlated numeric pair.

    Strategy
    --------
    1. Select all strictly numeric columns (excluding metadata: ``sourceId``,
       ``sourceUniqueId``, ``npsDate``, ``npsScore``, ``nps_class``).
    2. Compute the absolute Spearman rank correlation matrix.  Spearman is
       mandated to capture non-linear monotonic redundancies (e.g. order
       counts vs. order amounts) and to remain robust against the heavy
       operational outliers present in the rail dataset.
    3. Enumerate all pairs ``(a, b)`` with ``|ρ(a, b)| > threshold`` and
       process them in decreasing order of correlation.  For each pair where
       neither column has already been marked for drop, resolve the drop
       using this precedence:

       a. If only one column is in ``priority_features``, keep it and drop
          the other (protects causal levers such as delay duration in
          minutes).
       b. Otherwise, drop the column with the *higher* null rate.
       c. If null rates tie, keep the lexicographically smaller name.

    4. Columns with undefined correlations (zero variance → NaN) are skipped
       safely — ``NaN > threshold`` evaluates to ``False`` and they are kept.

    Parameters
    ----------
    df:
        Input Polars DataFrame.  Mixed dtype columns are tolerated; only the
        numeric subset is considered.
    threshold:
        Absolute correlation above which a pair is considered redundant.
        Defaults to ``0.85``.
    priority_features:
        Column names that must be retained when correlated with non-priority
        features.  Defaults to ``None`` (no protected columns).

    Returns
    -------
    pl.DataFrame
        ``df`` minus the dropped numeric columns.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold!r}.")

    priority: set[str] = set(priority_features or [])

    candidate_cols: list[str] = [
        c for c in df.columns
        if c not in _META_COLS and df.schema[c].is_numeric()
    ]

    if len(candidate_cols) < 2:
        print(
            f"Numeric collinearity filter: only {len(candidate_cols)} numeric "
            f"column(s) — nothing to filter."
        )
        return df

    numeric_pd: pd.DataFrame = df.select(candidate_cols).to_pandas()
    corr_abs: pd.DataFrame = numeric_pd.corr(method="spearman").abs()

    null_rate: dict[str, float] = {
        col: float(df[col].is_null().mean()) for col in candidate_cols
    }

    # Collect all (|ρ|, a, b) triples above threshold from the upper triangle.
    pairs: list[tuple[float, str, str]] = []
    for i in range(len(candidate_cols)):
        for j in range(i + 1, len(candidate_cols)):
            rho = corr_abs.iat[i, j]
            if pd.notna(rho) and rho > threshold:
                pairs.append((float(rho), candidate_cols[i], candidate_cols[j]))

    # Resolve high-correlation pairs first so that transitive redundancies
    # are handled greedily by importance.
    pairs.sort(key=lambda t: t[0], reverse=True)

    drop_log: list[tuple[str, str, float, str]] = []
    dropped: set[str] = set()

    for rho, col_a, col_b in pairs:
        if col_a in dropped or col_b in dropped:
            continue

        a_priority = col_a in priority
        b_priority = col_b in priority

        if a_priority and not b_priority:
            keep_col, drop_col = col_a, col_b
            reason = "priority feature protected"
        elif b_priority and not a_priority:
            keep_col, drop_col = col_b, col_a
            reason = "priority feature protected"
        else:
            null_a, null_b = null_rate[col_a], null_rate[col_b]
            if null_a > null_b:
                drop_col, keep_col = col_a, col_b
            elif null_b > null_a:
                drop_col, keep_col = col_b, col_a
            else:
                keep_col, drop_col = sorted([col_a, col_b])
            reason = (
                f"nulls: {null_rate[drop_col] * 100:.1f}% vs "
                f"{null_rate[keep_col] * 100:.1f}% kept"
            )

        dropped.add(drop_col)
        drop_log.append((drop_col, keep_col, rho, reason))

    if not drop_log:
        print(
            f"Numeric collinearity filter (Spearman |ρ| > {threshold}): "
            f"no redundant pairs found across {len(candidate_cols)} numeric columns."
        )
        return df

    if priority:
        protected = [c for c in priority if c in candidate_cols and c not in dropped]
        print(
            f"Numeric collinearity filter (Spearman |ρ| > {threshold}) — "
            f"dropped {len(drop_log)} column(s) of {len(candidate_cols)} "
            f"(priority protected: {len(protected)}):"
        )
    else:
        print(
            f"Numeric collinearity filter (Spearman |ρ| > {threshold}) — "
            f"dropped {len(drop_log)} column(s) of {len(candidate_cols)}:"
        )
    for drop_col, keep_col, rho, reason in drop_log:
        print(f"  • Dropped {drop_col}")
        print(f"      reason: |ρ|={rho:.3f} with {keep_col}  ({reason})")

    return df.drop(list(dropped))


# ---------------------------------------------------------------------------
# Delay Severity Flag
# ---------------------------------------------------------------------------

_DELAY_SOURCE_COL: str = "transactionalTravelTripAverageDelayDurationV0Last30d"
_DELAY_FLAG_COL: str = "is_severe_delay_10m_plus"


def add_severe_delay_flag(
    df: pl.DataFrame,
    threshold: float = 10.0,
) -> pl.DataFrame:
    """Append a binary severity flag for operationally significant delays.

    **Motivation — the Information Gain Trap**

    The average delay distribution is heavily right-skewed: the vast
    majority of customers experience negligible or zero delay, while a small
    minority suffer severe disruptions that are causally responsible for the
    bulk of detractor conversions.  In the absence of an explicit partition
    signal, a gradient-boosted tree evaluates splits greedily on log-loss
    reduction across the *full* population.  Because the majority-class
    "no-delay" node is already nearly pure, splitting it yields a large
    nominal information gain — far larger, in absolute terms, than isolating
    the high-delay tail even though the tail dominates detractor risk.  The
    tree therefore relegates the causal delay signal to mid-depth branches
    where it competes with dozens of other drivers.

    By surfacing an explicit ``is_severe_delay_10m_plus`` flag we hand the
    model a *pre-computed high-quality partition* at the root level.  LightGBM
    can then use ``TravelTripAverageDelayDuration`` continuously further down
    the branches to resolve finer causal inflection points, rather than
    rediscovering the coarse 10-minute boundary through brute-force search.

    **Null preservation**

    Customers for whom ``TravelTripAverageDelayDuration`` is null did not
    travel in the 30-day window.  The flag is left as ``null`` for these
    customers, preserving LightGBM's native NaN-routing logic.  This is
    critical: imputing 0 (no severe delay) for non-travellers would inject
    a false negative signal into the causal analysis.

    Parameters
    ----------
    df:
        Wide-format Polars DataFrame containing the delay duration column.
    threshold:
        Delay duration in minutes above which a journey is classified as
        severe.  Defaults to ``10.0``, which aligns with Deutsche Bahn's
        internal on-time definition (arrivals > 10 min late are officially
        "delayed").

    Returns
    -------
    pl.DataFrame
        Input DataFrame with ``is_severe_delay_10m_plus`` appended as
        ``pl.Int8`` (``1`` = severe, ``0`` = not severe, ``null`` = no travel).
    """
    if _DELAY_SOURCE_COL not in df.columns:
        raise KeyError(
            f"Expected column '{_DELAY_SOURCE_COL}' not found in DataFrame. "
            f"Ensure the delay duration driver was not dropped by an earlier filter."
        )

    flag_expr = (
        pl.when(pl.col(_DELAY_SOURCE_COL).is_null())
        .then(pl.lit(None, dtype=pl.Int8))
        .when(pl.col(_DELAY_SOURCE_COL) > threshold)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias(_DELAY_FLAG_COL)
    )

    out = df.with_columns(flag_expr)

    # Diagnostic summary
    n_severe  = out[_DELAY_FLAG_COL].eq(1).sum()
    n_normal  = out[_DELAY_FLAG_COL].eq(0).sum()
    n_null    = out[_DELAY_FLAG_COL].is_null().sum()
    print(
        f"Severe delay flag added (threshold > {threshold:.0f} min):\n"
        f"  Severe (1) : {n_severe:>8,}  ({n_severe / out.height * 100:.1f}%)\n"
        f"  Normal (0) : {n_normal:>8,}  ({n_normal / out.height * 100:.1f}%)\n"
        f"  No travel  : {n_null:>8,}  ({n_null  / out.height * 100:.1f}%)"
    )

    return out
