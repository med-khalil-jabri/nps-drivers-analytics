"""data_prep.py
=============
Production-grade Polars data pipeline for the Deutsche Bahn Actionable NPS
case study.

Responsibilities
----------------
1. ``parse_driver_definitions`` — reads the 63-driver metadata CSV and returns
   a ``{driverKey: polars_dtype}`` mapping that drives all downstream casting.

2. ``load_and_pivot_data`` — ingests the raw EAV (Entity-Attribute-Value) flat
   CSV, pivots to wide format (one row per customer), and applies strict type
   casting per the definitions map.

Design notes
------------
* All values arrive as ``Utf8`` when ``infer_schema_length=0`` is set (we own
  the schema, the file should not).  Empty strings are normalised to ``null``
  before any cast so that sparsity is represented uniformly as ``null``.
* The pivot uses ``aggregate_function="first"`` to be resilient against
  accidental duplicate (customer, driver) rows in the source data.
* ``is_lazy=True`` leverages ``pl.scan_csv`` for initial I/O, which keeps
  memory pressure low when sampling the 42 GB Live file for EDA.  Full
  streaming inference (Phase 5) uses a separate chunked pipeline.
* Multi-categorical drivers (Utf8) are intentionally left as raw strings;
  ``src/features.py`` owns their CountVectorizer expansion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Maps the ``driverType`` field in the definitions CSV → Polars target dtype.
_DRIVER_TYPE_TO_POLARS: dict[str, pl.DataType] = {
    "numeric": pl.Float32,
    "categorical": pl.Categorical,
    "multi_categorical": pl.Utf8,   # kept as raw string; expanded in features.py
}

# Metadata columns that are never treated as driver features.
_META_COLS: frozenset[str] = frozenset(
    {"sourceId", "sourceUniqueId", "npsDate", "npsScore", "driverKey", "value"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_driver_definitions(filepath: str | Path) -> dict[str, pl.DataType]:
    """Parse the driver definitions CSV into a ``{driverKey: polars_dtype}`` map.

    Parameters
    ----------
    filepath:
        Path to ``case_study_deutschebahn_driver_definitions.csv``.

    Returns
    -------
    dict[str, pl.DataType]
        Keys are the raw ``key`` strings from the definitions file (i.e. the
        ``driverKey`` values that appear in the flat EAV CSVs).  Values are
        the Polars dtype that column should carry in the wide DataFrame.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If a ``driverType`` value in the definitions is not recognised.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Driver definitions not found: {filepath}")

    defs = pl.read_csv(filepath, has_header=True)

    unknown_types: list[str] = []
    # The definitions CSV column layout (confirmed empirically):
    #   "driverType" → business context domain (Arrival, Basket, Travel, …)
    #   "dataType"   → actual data type (numeric | categorical | multi_categorical)
    # We use "dataType" as the casting discriminator.
    dtype_map: dict[str, pl.DataType] = {}

    for row in defs.iter_rows(named=True):
        key: str = row["key"]
        # NOTE: the CSV column "driverType" holds the business context domain
        # (e.g. "Arrival", "Basket"); the actual data type (numeric / categorical /
        # multi_categorical) lives in the "dataType" column.
        driver_type: str = row["dataType"]

        polars_dtype = _DRIVER_TYPE_TO_POLARS.get(driver_type)
        if polars_dtype is None:
            unknown_types.append(f"'{key}' has unrecognised driverType '{driver_type}'")
            continue
        dtype_map[key] = polars_dtype

    if unknown_types:
        # Warn but do not crash — future driver types should not block the pipeline.
        import warnings
        warnings.warn(
            f"Unrecognised driverType(s) — these columns will remain as Utf8:\n"
            + "\n".join(f"  • {m}" for m in unknown_types),
            stacklevel=2,
        )

    return dtype_map


def load_and_pivot_data(
    filepath: str | Path,
    definitions_map: dict[str, pl.DataType],
    is_lazy: bool = False,
    n_rows: Optional[int] = None,
) -> pl.DataFrame:
    """Load, pivot, and type-cast the raw EAV flat CSV.

    The function is schema-agnostic with respect to ``npsScore``: it detects
    whether the column is present (Train) or absent (Live) and builds the
    pivot index accordingly.

    Parameters
    ----------
    filepath:
        Path to the raw flat CSV (Train or Live).
    definitions_map:
        Output of :func:`parse_driver_definitions`.
    is_lazy:
        If ``True``, reads the file with ``pl.scan_csv`` (low memory) and
        collects only after optional row capping.  Useful for sampling the
        large Live CSV during EDA.  Note: ``pivot`` is an eager operation in
        Polars; the LazyFrame is always collected before pivoting.
    n_rows:
        Maximum number of *raw EAV rows* to read.  ``None`` reads everything.
        When sampling the Live set for EDA, set this to
        ``n_target_customers × 63`` (approximate, because not every customer
        has all 63 drivers populated).

    Returns
    -------
    pl.DataFrame
        Wide-format DataFrame: one row per (sourceId, sourceUniqueId), one
        column per driver key, plus ``npsDate`` and ``npsScore`` where present.
        Numeric columns are ``Float32``; categorical columns are ``Categorical``;
        multi-categorical columns remain ``Utf8``.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # ------------------------------------------------------------------
    # Step 1 — Load raw EAV rows, all as Utf8 (we control the schema)
    # ------------------------------------------------------------------
    if is_lazy:
        lazy_frame: pl.LazyFrame = pl.scan_csv(
            filepath,
            infer_schema_length=0,   # treat every column as Utf8
        )
        if n_rows is not None:
            lazy_frame = lazy_frame.head(n_rows)
        raw: pl.DataFrame = lazy_frame.collect()
    else:
        raw = pl.read_csv(
            filepath,
            infer_schema_length=0,
            n_rows=n_rows,
        )

    # ------------------------------------------------------------------
    # Step 2 — Determine pivot index (auto-detect npsScore presence)
    # ------------------------------------------------------------------
    has_nps_score: bool = "npsScore" in raw.columns

    index_cols: list[str] = ["sourceId", "sourceUniqueId", "npsDate"]
    if has_nps_score:
        index_cols.append("npsScore")

    # ------------------------------------------------------------------
    # Step 3 — Pivot: EAV → wide (one row per customer)
    # aggregate_function="first" handles accidental duplicate driver rows
    # ------------------------------------------------------------------
    wide: pl.DataFrame = raw.pivot(
        on="driverKey",
        index=index_cols,
        values="value",
        aggregate_function="first",
    )

    # ------------------------------------------------------------------
    # Step 4 — Type casting with empty-string normalisation
    # ------------------------------------------------------------------
    cast_exprs: list[pl.Expr] = _build_cast_exprs(
        columns=wide.columns,
        definitions_map=definitions_map,
        index_cols=set(index_cols),
    )
    if cast_exprs:
        wide = wide.with_columns(cast_exprs)

    # ------------------------------------------------------------------
    # Step 5 — Clean up metadata columns
    # ------------------------------------------------------------------
    # npsDate: strip timezone suffix and parse to pl.Date
    wide = wide.with_columns(
        pl.col("npsDate")
        .str.slice(0, 10)                          # "YYYY-MM-DD", drop tz suffix
        .str.to_date(format="%Y-%m-%d", strict=False)
        .alias("npsDate")
    )

    # npsScore: cast to Int8 (values are 0 / 7 / 10 — fit in 8 bits)
    if has_nps_score:
        wide = wide.with_columns(
            pl.col("npsScore").cast(pl.Int8, strict=False)
        )

    return wide


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_cast_exprs(
    columns: list[str],
    definitions_map: dict[str, pl.DataType],
    index_cols: set[str],
) -> list[pl.Expr]:
    """Return a list of Polars expressions that cast driver columns to their
    declared types, normalising empty strings to ``null`` first.

    Multi-categorical columns (Utf8) are only null-normalised — no actual
    dtype change is needed since the pivot already produces Utf8.
    """
    exprs: list[pl.Expr] = []

    for col_name in columns:
        if col_name in index_cols:
            continue  # leave metadata columns untouched

        target_dtype = definitions_map.get(col_name)
        if target_dtype is None:
            continue  # driver not in definitions — leave as Utf8

        # Normalise empty strings to null (CSV empty cell = "" after Utf8 read)
        null_normalised = (
            pl.when(pl.col(col_name) == "")
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(pl.col(col_name))
        )

        if target_dtype == pl.Float32:
            # strict=False: non-parseable strings (e.g. stray text) → null
            exprs.append(
                null_normalised.cast(pl.Float32, strict=False).alias(col_name)
            )
        elif target_dtype == pl.Categorical:
            exprs.append(
                null_normalised.cast(pl.Categorical).alias(col_name)
            )
        else:
            # multi_categorical: Utf8 → Utf8 (null-normalise only)
            exprs.append(null_normalised.alias(col_name))

    return exprs
