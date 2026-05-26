"""batch_inference.py
=====================
Memory-safe full-population Detractor scoring for the Deutsche Bahn Live
dataset (≈ 42 GB, 314 M raw EAV rows, ≈ 7 M unique customers).

Why hash-sharding?
------------------
The Train → Live transformation requires an EAV-to-wide pivot
(``driverKey → columns``).  A pivot is a *blocking* operation: Polars
cannot stream it the way it streams a filter or an aggregation, because
the output schema depends on the *set* of distinct pivot keys observed.
On an Apple M2 / unified-memory laptop, attempting to pivot the entire
42 GB file in a single pass trips the kernel's OOM killer.

The fix used here is a **deterministic hash partition** over ``sourceId``:

    ``pl.col("sourceId").hash(seed=42) % num_shards == i``

Because ``hash(sourceId)`` is a per-row, deterministic function, *every*
EAV row for a given customer falls into the same shard.  The pivot
inside each flush is therefore correct for the customers it contains.

The pipeline performs a **single streaming pass** over the Live CSV:

1. Read fixed-size batches via ``pl.read_csv_batched`` (bounded RAM).
2. Hash-route each batch row into per-shard in-memory buffers.
3. When a buffer exceeds ``flush_rows``, pivot/score/append the
   *complete* customers in that buffer (the last ``sourceId`` is held
   back in case its EAV rows continue in the next batch).
4. Flush any remaining buffers at EOF.

This replaces the original design that called ``lazy.collect()`` once
*per shard*, which both re-read the 39 GB file ``num_shards`` times
and OOM'd while materialising filtered rows during the full-file scan.

CLI usage
---------
From the project root::

    python src/batch_inference.py

Or from a notebook in ``notebooks/``::

    !python ../src/batch_inference.py

Required artefacts
------------------
Three pickle files produced by the training notebook must exist before
the script runs:

* ``models/lgbm_binary.pkl``    — the fitted ``LGBMClassifier``
* ``models/mc_encoder.pkl``     — the fitted ``MultiCategoricalEncoder``
* ``models/feature_schema.pkl`` — an empty pandas DataFrame
  (``X_train_bin.iloc[:0]``) carrying the training column order **and**
  the pandas ``category`` dtypes.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import polars as pl

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / ".cursor" / "debug-02278f.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": "02278f",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _mem_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return -1.0
# #endregion

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_prep import parse_driver_definitions, pivot_eav_block  # noqa: E402
from src.features import MultiCategoricalEncoder, add_severe_delay_flag  # noqa: E402
from src.train import get_feature_matrix  # noqa: E402


_DEFAULT_INPUT       = _PROJECT_ROOT / "data" / "case_study_deutschebahn_drivers_flat_users_live.csv"
_DEFAULT_OUTPUT      = _PROJECT_ROOT / "data" / "predictions_live.csv"
_DEFAULT_DEFINITIONS = _PROJECT_ROOT / "data" / "case_study_deutschebahn_driver_definitions.csv"
_DEFAULT_MODEL       = _PROJECT_ROOT / "models" / "lgbm_binary.pkl"
_DEFAULT_ENCODER     = _PROJECT_ROOT / "models" / "mc_encoder.pkl"
_DEFAULT_SCHEMA      = _PROJECT_ROOT / "models" / "feature_schema.pkl"
_DEFAULT_NUM_SHARDS  = 10
_DEFAULT_HASH_SEED   = 42
_DEFAULT_BATCH_SIZE  = 500_000
_DEFAULT_FLUSH_ROWS  = 500_000

_POSITIVE_CLASS_LABEL = 1
_SOURCE_ID_COL        = "sourceId"
_OUTPUT_PROB_COL      = "detractor_probability"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_inference")


def _require_file(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {kind} not found: {path}\n"
            f"Generate it from the training notebook before running this script."
        )


def _align_to_training_schema(X: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    X = X.reindex(columns=schema.columns)
    for col in schema.columns:
        dtype = schema[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            X[col] = pd.Categorical(
                X[col],
                categories=dtype.categories,
                ordered=dtype.ordered,
            )
    return X


def _resolve_positive_class_index(model) -> int:
    classes = list(model.classes_)
    if _POSITIVE_CLASS_LABEL not in classes:
        raise ValueError(
            f"Model classes {classes} do not contain the positive label "
            f"{_POSITIVE_CLASS_LABEL}; this script expects a binary "
            f"Detractor-vs-Non-Detractor model."
        )
    return classes.index(_POSITIVE_CLASS_LABEL)


def _ensure_driver_columns(
    wide: pl.DataFrame,
    definitions_map: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Add missing driver columns as null after a partial EAV pivot.

    Streaming flushes only pivot the driver keys present in that chunk.
    LightGBM and ``add_severe_delay_flag`` expect the full driver schema,
    so we back-fill any absent columns from ``definitions_map``.
    """
    missing_exprs: list[pl.Expr] = []
    for key, dtype in definitions_map.items():
        if key in wide.columns:
            continue
        if dtype == pl.Float32:
            missing_exprs.append(pl.lit(None, dtype=pl.Float32).alias(key))
        elif dtype == pl.Categorical:
            missing_exprs.append(pl.lit(None, dtype=pl.Categorical).alias(key))
        else:
            missing_exprs.append(pl.lit(None, dtype=pl.String).alias(key))
    if missing_exprs:
        wide = wide.with_columns(missing_exprs)
    return wide


def _split_carry_last_customer(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a raw EAV block into (flush_now, carry_over).

    The last ``sourceId`` in ``raw`` may have more EAV rows in the next
    CSV batch, so it is held back until the buffer is flushed at EOF or
    until that customer's rows are complete.
    """
    if raw.height == 0:
        return raw, raw
    last_id = raw[_SOURCE_ID_COL][-1]
    flush_part = raw.filter(pl.col(_SOURCE_ID_COL) != last_id)
    carry = raw.filter(pl.col(_SOURCE_ID_COL) == last_id)
    return flush_part, carry


def _append_predictions(
    sourceids: pl.Series,
    probabilities,
    output_path: Path,
    write_header: bool,
) -> bool:
    out_df = pl.DataFrame(
        {
            _SOURCE_ID_COL: sourceids,
            _OUTPUT_PROB_COL: pl.Series(_OUTPUT_PROB_COL, probabilities, dtype=pl.Float32),
        }
    )
    if write_header:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_csv(output_path, include_header=True)
        return False
    with output_path.open("ab") as fh:
        out_df.write_csv(fh, include_header=False)
    return write_header


def _score_raw_block(
    raw: pl.DataFrame,
    shard_idx: int,
    num_shards: int,
    definitions_map: dict[str, pl.DataType],
    encoder: MultiCategoricalEncoder,
    model,
    pos_class_idx: int,
    schema: pd.DataFrame,
    output_path: Path,
    write_header: bool,
) -> tuple[int, int, bool]:
    """Pivot, feature-engineer, score, and append one raw EAV block."""
    if raw.height == 0:
        return 0, 0, write_header

    n_raw = raw.height
    t0 = time.perf_counter()

    wide = pivot_eav_block(raw, definitions_map)
    wide = _ensure_driver_columns(wide, definitions_map)
    del raw
    gc.collect()

    wide = encoder.transform(wide)
    wide = add_severe_delay_flag(wide, threshold=10.0)
    sourceids = wide[_SOURCE_ID_COL]

    X_chunk, _ = get_feature_matrix(wide, is_inference=True)
    del wide
    gc.collect()

    X_chunk = _align_to_training_schema(X_chunk, schema)
    probs = model.predict_proba(X_chunk)[:, pos_class_idx]
    n_customers = len(probs)

    write_header = _append_predictions(sourceids, probs, output_path, write_header)

    log.info(
        "  scored flush | shard=%3d/%d | raw=%8s | customers=%8s | %.1fs | rss=%.0f MB",
        shard_idx + 1,
        num_shards,
        f"{n_raw:,}",
        f"{n_customers:,}",
        time.perf_counter() - t0,
        _mem_mb(),
    )
    # #region agent log
    _debug_log(
        "C",
        "batch_inference.py:_score_raw_block",
        "flush_scored",
        {
            "shard_idx": shard_idx,
            "raw_rows": n_raw,
            "customers": n_customers,
            "rss_mb": round(_mem_mb(), 1),
        },
        run_id="post-fix",
    )
    # #endregion

    del X_chunk, probs, sourceids
    gc.collect()
    return n_raw, n_customers, write_header


def _stream_score_live_csv(
    input_path: Path,
    output_path: Path,
    num_shards: int,
    hash_seed: int,
    batch_size: int,
    flush_rows: int,
    definitions_map: dict[str, pl.DataType],
    encoder: MultiCategoricalEncoder,
    model,
    pos_class_idx: int,
    schema: pd.DataFrame,
) -> tuple[int, int]:
    """Single-pass streaming inference with bounded in-memory shard buffers."""
    pending: list[list[pl.DataFrame]] = [[] for _ in range(num_shards)]
    pending_counts = [0] * num_shards
    write_header = True
    batches_seen = 0
    total_raw = 0
    total_customers = 0
    t0 = time.perf_counter()

    log.info(
        "Streaming inference (single pass) | batch_size=%s | flush_rows=%s | num_shards=%d",
        f"{batch_size:,}",
        f"{flush_rows:,}",
        num_shards,
    )
    # #region agent log
    _debug_log(
        "A",
        "batch_inference.py:_stream_score_live_csv:start",
        "stream_start",
        {
            "num_shards": num_shards,
            "batch_size": batch_size,
            "flush_rows": flush_rows,
            "input_gb": round(input_path.stat().st_size / (1024 ** 3), 2),
            "full_file_scans": 1,
        },
        run_id="post-fix",
    )
    # #endregion

    def _flush_shard(shard_idx: int, *, final: bool = False) -> None:
        nonlocal write_header, total_raw, total_customers
        if pending_counts[shard_idx] == 0:
            return

        raw = (
            pl.concat(pending[shard_idx], how="vertical")
            if len(pending[shard_idx]) > 1
            else pending[shard_idx][0]
        )
        pending[shard_idx] = []
        pending_counts[shard_idx] = 0

        if final:
            blocks = [raw]
        else:
            flush_part, carry = _split_carry_last_customer(raw)
            if carry.height:
                pending[shard_idx] = [carry]
                pending_counts[shard_idx] = carry.height
            blocks = [flush_part] if flush_part.height else []

        for block in blocks:
            n_raw, n_customers, write_header = _score_raw_block(
                block,
                shard_idx=shard_idx,
                num_shards=num_shards,
                definitions_map=definitions_map,
                encoder=encoder,
                model=model,
                pos_class_idx=pos_class_idx,
                schema=schema,
                output_path=output_path,
                write_header=write_header,
            )
            total_raw += n_raw
            total_customers += n_customers

    reader = pl.read_csv_batched(
        str(input_path),
        batch_size=batch_size,
        infer_schema_length=0,
    )
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        batch = batches[0]
        batches_seen += 1

        for shard_idx in range(num_shards):
            shard_batch = batch.filter(
                pl.col(_SOURCE_ID_COL).hash(seed=hash_seed) % num_shards == shard_idx
            )
            if shard_batch.height == 0:
                continue
            pending[shard_idx].append(shard_batch)
            pending_counts[shard_idx] += shard_batch.height
            if pending_counts[shard_idx] >= flush_rows:
                _flush_shard(shard_idx, final=False)

        if batches_seen % 50 == 0:
            pending_sum = sum(pending_counts)
            log.info(
                "  stream progress | batches=%s | buffered_rows=%s | rss=%.0f MB | %.0fs",
                f"{batches_seen:,}",
                f"{pending_sum:,}",
                _mem_mb(),
                time.perf_counter() - t0,
            )
            # #region agent log
            _debug_log(
                "B",
                "batch_inference.py:_stream_score_live_csv:progress",
                "stream_batch_progress",
                {
                    "batches_seen": batches_seen,
                    "buffered_rows": pending_sum,
                    "rss_mb": round(_mem_mb(), 1),
                    "elapsed_secs": round(time.perf_counter() - t0, 1),
                },
                run_id="post-fix",
            )
            # #endregion

        del batch
        gc.collect()

    for shard_idx in range(num_shards):
        _flush_shard(shard_idx, final=True)

    elapsed = time.perf_counter() - t0
    log.info(
        "DONE | batches=%s | raw=%s | customers=%s | %.1fs (≈ %.1f min)",
        f"{batches_seen:,}",
        f"{total_raw:,}",
        f"{total_customers:,}",
        elapsed,
        elapsed / 60.0,
    )
    # #region agent log
    _debug_log(
        "A",
        "batch_inference.py:_stream_score_live_csv:done",
        "stream_complete",
        {
            "batches_seen": batches_seen,
            "total_raw": total_raw,
            "total_customers": total_customers,
            "elapsed_secs": round(elapsed, 1),
            "rss_mb": round(_mem_mb(), 1),
        },
        run_id="post-fix",
    )
    # #endregion
    return total_raw, total_customers


def run_batch_inference(
    num_shards: int = _DEFAULT_NUM_SHARDS,
    hash_seed: int = _DEFAULT_HASH_SEED,
    input_path: Path = _DEFAULT_INPUT,
    output_path: Path = _DEFAULT_OUTPUT,
    definitions_path: Path = _DEFAULT_DEFINITIONS,
    model_path: Path = _DEFAULT_MODEL,
    encoder_path: Path = _DEFAULT_ENCODER,
    schema_path: Path = _DEFAULT_SCHEMA,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    flush_rows: int = _DEFAULT_FLUSH_ROWS,
) -> Path:
    """Run hash-sharded streaming inference on the Live EAV dataset."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1; got {num_shards}.")
    if batch_size < 1 or flush_rows < 1:
        raise ValueError("batch_size and flush_rows must be >= 1.")

    _require_file(input_path,       "input Live CSV")
    _require_file(definitions_path, "driver definitions CSV")
    _require_file(model_path,       "trained model pickle")
    _require_file(encoder_path,     "fitted encoder pickle")
    _require_file(schema_path,      "training feature schema pickle")

    log.info("Loading artefacts …")
    definitions_map = parse_driver_definitions(definitions_path)
    encoder: MultiCategoricalEncoder = joblib.load(encoder_path)
    model = joblib.load(model_path)
    schema: pd.DataFrame = joblib.load(schema_path)
    pos_class_idx = _resolve_positive_class_index(model)

    log.info("Loaded model classes = %s | feature count = %d", list(model.classes_), len(schema.columns))
    log.info("Input  : %s", input_path)
    log.info("Output : %s", output_path)

    if output_path.exists():
        output_path.unlink()

    _stream_score_live_csv(
        input_path=input_path,
        output_path=output_path,
        num_shards=num_shards,
        hash_seed=hash_seed,
        batch_size=batch_size,
        flush_rows=flush_rows,
        definitions_map=definitions_map,
        encoder=encoder,
        model=model,
        pos_class_idx=pos_class_idx,
        schema=schema,
    )
    log.info("Predictions written to %s", output_path)
    return output_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Hash-sharded batch scoring of the Live EAV dataset. "
            "Writes per-customer Detractor probabilities to a single CSV."
        ),
    )
    p.add_argument("--input",       type=Path, default=_DEFAULT_INPUT)
    p.add_argument("--output",      type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--definitions", type=Path, default=_DEFAULT_DEFINITIONS)
    p.add_argument("--model",       type=Path, default=_DEFAULT_MODEL)
    p.add_argument("--encoder",     type=Path, default=_DEFAULT_ENCODER)
    p.add_argument("--schema",      type=Path, default=_DEFAULT_SCHEMA)
    p.add_argument("--num-shards",  type=int,  default=_DEFAULT_NUM_SHARDS)
    p.add_argument("--hash-seed",   type=int,  default=_DEFAULT_HASH_SEED)
    p.add_argument("--batch-size",  type=int,  default=_DEFAULT_BATCH_SIZE)
    p.add_argument("--flush-rows",  type=int,  default=_DEFAULT_FLUSH_ROWS)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    run_batch_inference(
        num_shards=args.num_shards,
        hash_seed=args.hash_seed,
        input_path=args.input,
        output_path=args.output,
        definitions_path=args.definitions,
        model_path=args.model,
        encoder_path=args.encoder,
        schema_path=args.schema,
        batch_size=args.batch_size,
        flush_rows=args.flush_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
