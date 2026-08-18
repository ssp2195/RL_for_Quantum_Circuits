"""Deterministic publication reporting primitives for Article V1.

The module is intentionally independent of the runner and search environment.
It accepts raw run mappings, persists them without dropping failures, and
rebuilds every CSV, table, and SVG artifact from ``raw_runs.jsonl`` alone.
The held-out target--not an evaluation-seed trajectory--is the statistical
unit throughout aggregation and bootstrap uncertainty estimation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any


ARTICLE_V1_RAW_RUN_SCHEMA = "article-v1-raw-run-v4"
ARTICLE_V1_REPORT_SCHEMA = "article-v1-publication-report-v4"
DEFAULT_STATISTICS_SEED = 20_260_815
DEFAULT_BOOTSTRAP_SAMPLES = 10_000

_MISSING = object()
_NONE_CHECKPOINT = "none"
_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#4b5563",
    "#be185d",
)


def _path_value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _first_value(
    record: Mapping[str, Any],
    paths: Sequence[str],
    *,
    name: str,
    default: Any = _MISSING,
) -> Any:
    for path in paths:
        value = _path_value(record, path)
        if value is not _MISSING and value is not None:
            return value
    if default is not _MISSING:
        return default
    raise KeyError(f"raw run is missing required {name}: expected one of {tuple(paths)!r}")


def _canonical_json_value(value: Any) -> Any:
    """Return a deterministic JSON value or reject unsupported/nonfinite data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("run identity and raw records must not contain NaN/Infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    # NumPy scalar-like values deliberately use ``item`` without importing
    # NumPy into the dependency-free reporting layer.
    item = getattr(value, "item", None)
    if callable(item):
        return _canonical_json_value(item())
    raise TypeError(f"value {value!r} is not JSON serializable")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        _canonical_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def run_identity_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete deterministic identity of one evaluation run."""

    if not isinstance(run, Mapping):
        raise TypeError("run must be a mapping")
    payload = {
        "target_id": _first_value(
            run,
            ("target_id", "target_identity", "target.id", "target.fingerprint"),
            name="target identity",
        ),
        "config_digest": _first_value(
            run,
            ("config_digest", "corpus_config_digest", "config.digest"),
            name="corpus configuration digest",
            default="unspecified",
        ),
        "scheduler": _first_value(run, ("scheduler",), name="scheduler"),
        "resource_budget": _first_value(
            run,
            ("resource_budget", "budget", "config.resource_budget"),
            name="resource budget",
        ),
        "expansion_budget": _first_value(
            run,
            ("expansion_budget", "max_steps", "config.expansion_budget"),
            name="expansion budget",
        ),
        "checkpoint_digest": _first_value(
            run,
            (
                "checkpoint_digest",
                "policy_weight_digest",
                "checkpoint.digest",
                "policy.policy_weight_digest",
            ),
            name="checkpoint digest",
            default=_NONE_CHECKPOINT,
        ),
        "training_seed": _first_value(
            run,
            ("training_seed", "checkpoint.training_seed", "policy.training_seed"),
            name="training seed",
            default=None,
        ),
        "evaluation_seed": _first_value(
            run,
            ("evaluation_seed", "eval_seed", "seed"),
            name="evaluation seed",
        ),
        "feature_schema": _first_value(
            run,
            (
                "feature_schema_version",
                "feature_schema",
                "schemas.feature",
                "profile.feature_schema",
            ),
            name="feature schema",
        ),
        "feature_evaluator_schema": _first_value(
            run,
            (
                "feature_evaluator_schema_version",
                "feature_evaluator_schema",
                "schemas.feature_evaluator",
                "profile.feature_evaluator_schema",
            ),
            name="feature evaluator schema",
        ),
        "reward_schema": _first_value(
            run,
            (
                "reward_schema_version",
                "reward_schema",
                "schemas.reward",
                "profile.reward_schema",
                "reward_mode",
            ),
            name="reward schema",
        ),
        "reward_parameters": _first_value(
            run,
            ("reward_parameters", "config.reward_parameters"),
            name="reward parameters",
            default={},
        ),
        "target_metric_schema": _first_value(
            run,
            (
                "target_metric_schema_version",
                "target_metric_schema",
                "schemas.target_metric",
                "profile.target_metric_schema",
            ),
            name="target metric schema",
        ),
        "certifier_schema": _first_value(
            run,
            (
                "certification_schema_version",
                "certifier_schema",
                "schemas.certifier",
                "profile.certification_schema",
            ),
            name="certifier schema",
        ),
        "certification_parameters": _first_value(
            run,
            ("certification_parameters", "config.certification_parameters"),
            name="certification parameters",
            default={},
        ),
        "search_reduction": _first_value(
            run,
            ("search_reduction", "config.search_reduction"),
            name="search reduction configuration",
            default={
                "canonicalization_enabled": True,
                "pareto_dominance_enabled": True,
                "absorb_clifford_angles": True,
                "canonicalization_mode": "enhanced",
            },
        ),
        "code_version": _first_value(
            run,
            (
                "code_version",
                "code_commit_sha",
                "commit_sha",
                "git.commit_sha",
            ),
            name="code version",
        ),
        "source_worktree_digest": _first_value(
            run,
            (
                "source_worktree_digest",
                "code.source_worktree_digest",
                "git.source_worktree_digest",
            ),
            name="source worktree digest",
        ),
    }
    normalized = _canonical_json_value(payload)
    expansion_budget = normalized["expansion_budget"]
    if (
        isinstance(expansion_budget, bool)
        or not isinstance(expansion_budget, int)
        or expansion_budget <= 0
    ):
        raise ValueError("expansion_budget must be a positive integer")
    training_seed = normalized["training_seed"]
    if training_seed is not None and (
        isinstance(training_seed, bool) or not isinstance(training_seed, int)
    ):
        raise ValueError("training_seed must be an integer or null")
    if (
        normalized["checkpoint_digest"] != _NONE_CHECKPOINT
        and training_seed is None
    ):
        raise ValueError("a checkpoint-backed run must serialize its training_seed")
    if normalized["checkpoint_digest"] == _NONE_CHECKPOINT and training_seed is not None:
        raise ValueError("a checkpoint-free run must serialize training_seed as null")
    source_digest = normalized["source_worktree_digest"]
    if not isinstance(source_digest, str) or not source_digest:
        raise ValueError("source_worktree_digest must be a nonempty string")
    config_digest = normalized["config_digest"]
    if not isinstance(config_digest, str) or not config_digest:
        raise ValueError("config_digest must be a nonempty string")
    return normalized


def unique_run_key(run: Mapping[str, Any]) -> str:
    """Hash every field capable of changing an Article-V1 run outcome."""

    encoded = _canonical_json(run_identity_payload(run)).encode("utf-8")
    return f"article-v1-run:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class AppendOnlyJSONLRunStore:
    """Atomic append-by-rewrite store with crash-safe final-line recovery."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read(self, *, repair: bool) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        complete_end = raw.rfind(b"\n") + 1
        complete = raw[:complete_end]
        partial = raw[complete_end:]
        records: list[dict[str, Any]] = []
        records_by_key: dict[str, dict[str, Any]] = {}
        for line_number, encoded_line in enumerate(complete.splitlines(), start=1):
            if not encoded_line.strip():
                continue
            try:
                decoded = json.loads(encoded_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"corrupt complete JSONL record at line {line_number}"
                ) from error
            if not isinstance(decoded, dict):
                raise ValueError(f"JSONL record at line {line_number} is not an object")
            if decoded.get("raw_run_schema") != ARTICLE_V1_RAW_RUN_SCHEMA:
                raise ValueError(
                    f"JSONL record at line {line_number} uses an unsupported "
                    "Article V1 raw-run schema"
                )
            observed_key = decoded.get("run_key")
            expected_key = unique_run_key(decoded)
            if observed_key != expected_key:
                raise ValueError(
                    f"JSONL record at line {line_number} has an invalid run_key"
                )
            previous = records_by_key.get(expected_key)
            if previous is not None:
                if _canonical_json(previous) != _canonical_json(decoded):
                    raise ValueError(f"conflicting completed run key {expected_key}")
                continue
            records.append(decoded)
            records_by_key[expected_key] = decoded
        if partial and repair:
            # A line without its commit newline is not a completed record even
            # when its prefix happens to be valid JSON.  Discard it atomically.
            _atomic_write_bytes(self.path, complete)
        return records

    def load_records(self, *, repair_partial: bool = True) -> list[dict[str, Any]]:
        return self._read(repair=repair_partial)

    def completed_keys(self, *, repair_partial: bool = True) -> set[str]:
        return {
            str(record["run_key"])
            for record in self.load_records(repair_partial=repair_partial)
        }

    def append(self, run: Mapping[str, Any]) -> bool:
        """Append one completed run; return false for an identical resume hit."""

        normalized = _canonical_json_value(dict(run))
        declared_schema = normalized.get(
            "raw_run_schema", normalized.get("schema_version", ARTICLE_V1_RAW_RUN_SCHEMA)
        )
        if declared_schema != ARTICLE_V1_RAW_RUN_SCHEMA:
            raise ValueError("unsupported Article V1 raw-run schema")
        normalized["raw_run_schema"] = ARTICLE_V1_RAW_RUN_SCHEMA
        key = unique_run_key(normalized)
        normalized["run_key"] = key
        existing = self.load_records(repair_partial=True)
        for record in existing:
            if record["run_key"] != key:
                continue
            if _canonical_json(record) != _canonical_json(normalized):
                raise ValueError(f"completed run key {key} has conflicting payloads")
            return False
        lines = [_canonical_json(record) for record in (*existing, normalized)]
        _atomic_write_bytes(
            self.path,
            ("\n".join(lines) + "\n").encode("utf-8"),
        )
        return True


# Compact alias for callers that already know the store is JSONL.
ArticleV1RunStore = AppendOnlyJSONLRunStore


def load_completed_run_keys(path: str | Path) -> set[str]:
    return AppendOnlyJSONLRunStore(path).completed_keys()


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _record_success(record: Mapping[str, Any]) -> bool:
    value = _first_value(
        record,
        ("certified", "success", "passed"),
        name="certified outcome",
        default=False,
    )
    return bool(value)


def _hit_expansion(record: Mapping[str, Any]) -> int | None:
    if not _record_success(record):
        return None
    return _as_int(
        _first_value(
            record,
            ("t_hit", "hit_expansion", "expansions_to_solution", "expansions"),
            name="hit expansion",
            default=None,
        )
    )


_TIMING_PATHS: dict[str, tuple[str, ...]] = {
    "wall_time_seconds": (
        "timings.wall_time_seconds",
        "timing.wall_time_seconds",
        "wall_time_seconds",
        "runtime_seconds",
    ),
    "ranking_time_seconds": (
        "timings.ranking_time_seconds",
        "timing.ranking_time_seconds",
        "ranking_time_seconds",
    ),
    "feature_time_seconds": (
        "timings.feature_time_seconds",
        "timing.feature_time_seconds",
        "feature_time_seconds",
    ),
    "target_metric_time_seconds": (
        "timings.target_metric_time_seconds",
        "timing.target_metric_time_seconds",
        "target_metric_time_seconds",
    ),
    "canonicalization_time_seconds": (
        "timings.canonicalization_time_seconds",
        "timing.canonicalization_time_seconds",
        "canonicalization_time_seconds",
    ),
    "archive_time_seconds": (
        "timings.archive_time_seconds",
        "timing.archive_time_seconds",
        "archive_time_seconds",
    ),
    "certification_time_seconds": (
        "timings.certification_time_seconds",
        "timing.certification_time_seconds",
        "certification_time_seconds",
    ),
    "feature_evaluations": (
        "metrics.feature_evaluations",
        "search_metrics.feature_evaluations",
        "feature_evaluations",
    ),
    "dense_target_evaluations": (
        "metrics.dense_target_evaluations",
        "search_metrics.dense_target_evaluations",
        "dense_target_evaluations",
    ),
    "target_metric_cache_hits": (
        "metrics.target_metric_cache_hits",
        "search_metrics.target_metric_cache_hits",
        "target_metric_cache_hits",
    ),
    "peak_frontier": (
        "metrics.peak_frontier",
        "search_metrics.frontier_peak",
        "search_metrics.frontier_max",
        "peak_frontier",
    ),
    "peak_archive": (
        "metrics.peak_archive",
        "search_metrics.archive_peak",
        "peak_archive",
        "archive_size_final",
    ),
}


def _timing_values(record: Mapping[str, Any]) -> dict[str, float | None]:
    values = {
        name: _as_float(
            _first_value(record, paths, name=name, default=None)
        )
        for name, paths in _TIMING_PATHS.items()
    }
    hits = values["target_metric_cache_hits"]
    evaluations = values["dense_target_evaluations"]
    values["target_metric_cache_hit_rate"] = (
        None
        if hits is None or evaluations is None or hits + evaluations <= 0.0
        else hits / (hits + evaluations)
    )
    expansions = _as_float(record.get("expansions"))
    wall_time = values["wall_time_seconds"]
    values["expansions_per_wall_second"] = (
        None
        if expansions is None or wall_time is None or wall_time <= 0.0
        else expansions / wall_time
    )
    return values


_RESOURCE_NAMES = ("t_count", "two_qubit_count", "gate_count", "depth")


def _resource_values(record: Mapping[str, Any]) -> dict[str, float] | None:
    vector = record.get("solution_resource_vector")
    if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)):
        values = [_as_float(value) for value in vector]
        if len(values) >= 3 and all(value is not None for value in values[:3]):
            depths = [value for value in values[3:] if value is not None]
            return {
                "t_count": float(values[0]),
                "two_qubit_count": float(values[1]),
                "gate_count": float(values[2]),
                "depth": float(max(depths, default=0.0)),
            }
    mapping = record.get("solution_resources")
    if isinstance(mapping, Mapping):
        aliases = {
            "t_count": ("t_count", "num_t"),
            "two_qubit_count": ("two_qubit_count", "cnot_count", "num_2q"),
            "gate_count": ("gate_count", "num_gates"),
            "depth": ("depth",),
        }
        result: dict[str, float] = {}
        for name, paths in aliases.items():
            value = _as_float(_first_value(mapping, paths, name=name, default=None))
            if value is not None:
                result[name] = value
        return result or None
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _derived_seed(stats_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{stats_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    stats_seed: int = DEFAULT_STATISTICS_SEED,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> tuple[float | None, float | None]:
    """Return a deterministic target-level percentile-bootstrap 95% CI."""

    if isinstance(stats_seed, bool) or not isinstance(stats_seed, int):
        raise TypeError("stats_seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    numeric = [float(value) for value in values]
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return numeric[0], numeric[0]
    rng = random.Random(stats_seed)
    count = len(numeric)
    estimates = sorted(
        statistics.fmean(numeric[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _group_fields(run: Mapping[str, Any]) -> dict[str, Any]:
    identity = run_identity_payload(run)

    def display(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return str(value)
        return _canonical_json(value)

    return {
        "split": str(run.get("split", "unspecified")),
        "difficulty": str(run.get("difficulty", "unspecified")),
        "scheduler": str(identity["scheduler"]),
        "checkpoint_digest": str(identity["checkpoint_digest"]),
        "training_seed": identity["training_seed"],
        "resource_budget": _canonical_json(identity["resource_budget"]),
        "feature_schema": display(identity["feature_schema"]),
        "feature_evaluator_schema": display(
            identity["feature_evaluator_schema"]
        ),
        "reward_schema": display(identity["reward_schema"]),
        "reward_parameters": display(identity["reward_parameters"]),
        "target_metric_schema": display(identity["target_metric_schema"]),
        "certifier_schema": display(identity["certifier_schema"]),
        "certification_parameters": display(identity["certification_parameters"]),
        "search_reduction": display(identity["search_reduction"]),
        "code_version": display(identity["code_version"]),
        "source_worktree_digest": display(identity["source_worktree_digest"]),
    }


def _group_key(fields: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(fields[name])
        for name in (
            "split",
            "difficulty",
            "scheduler",
            "checkpoint_digest",
            "training_seed",
            "resource_budget",
            "feature_schema",
            "feature_evaluator_schema",
            "reward_schema",
            "reward_parameters",
            "target_metric_schema",
            "certifier_schema",
            "certification_parameters",
            "search_reduction",
            "code_version",
            "source_worktree_digest",
        )
    )


def _trajectory_seed(record: Mapping[str, Any]) -> str:
    return _canonical_json(run_identity_payload(record)["evaluation_seed"])


def _per_target_row(
    records: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
    budget: int,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    by_seed: dict[str, Mapping[str, Any]] = {}
    for record in sorted(
        records,
        key=lambda item: (
            int(run_identity_payload(item)["expansion_budget"]),
            str(item.get("run_key", unique_run_key(item))),
        ),
    ):
        if int(run_identity_payload(record)["expansion_budget"]) != budget:
            continue
        # Article features contain the remaining-budget coordinate.  A run
        # executed with a larger cap therefore is not a valid observation at a
        # smaller cap, even when its first hit occurred before that cap.
        by_seed.setdefault(_trajectory_seed(record), record)
    selected = list(by_seed.values())
    outcomes: list[bool] = []
    successful_expansions: list[float] = []
    timings = {name: [] for name in (*_TIMING_PATHS, "target_metric_cache_hit_rate", "expansions_per_wall_second")}
    resources = {name: [] for name in _RESOURCE_NAMES}
    for record in selected:
        hit = _hit_expansion(record)
        succeeded = bool(hit is not None and hit <= budget)
        outcomes.append(succeeded)
        if succeeded:
            successful_expansions.append(float(hit))
            resource = _resource_values(record)
            if resource is not None:
                for name in _RESOURCE_NAMES:
                    if name in resource:
                        resources[name].append(resource[name])
        timing = _timing_values(record)
        for name, value in timing.items():
            if value is not None:
                timings[name].append(value)
    row: dict[str, Any] = {
        **fields,
        "expansion_budget": budget,
        "target_id": target_id,
        "trajectory_count": len(selected),
        "successes": sum(outcomes),
        "failures": len(outcomes) - sum(outcomes),
        "success_probability": (
            statistics.fmean(float(value) for value in outcomes) if outcomes else 0.0
        ),
        "conditional_successful_expansions_mean": _mean(successful_expansions),
        "conditional_successful_expansions_median": (
            statistics.median(successful_expansions)
            if successful_expansions
            else None
        ),
        "conditional_successful_expansions_std": (
            statistics.pstdev(successful_expansions)
            if successful_expansions
            else None
        ),
    }
    row.update({f"{name}_mean": _mean(values) for name, values in timings.items()})
    row.update(
        {f"solution_{name}_mean": _mean(values) for name, values in resources.items()}
    )
    return row


def _learner_seed_summary(
    per_target: Sequence[Mapping[str, Any]],
    *,
    stats_seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Report each learner first, then separate target and learner variation."""

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "split",
        "difficulty",
        "scheduler",
        "resource_budget",
        "feature_schema",
        "feature_evaluator_schema",
        "reward_schema",
        "reward_parameters",
        "certifier_schema",
        "certification_parameters",
        "search_reduction",
        "code_version",
        "source_worktree_digest",
        "expansion_budget",
    )
    for row in per_target:
        if row["scheduler"] != "article_sarsa":
            continue
        groups[tuple(str(row[name]) for name in fields)].append(row)

    output: list[dict[str, Any]] = []
    learner_output: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        prefix = dict(zip(fields, key))
        prefix["expansion_budget"] = int(prefix["expansion_budget"])
        by_learner: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            training_seed = row.get("training_seed")
            if isinstance(training_seed, bool) or not isinstance(training_seed, int):
                raise ValueError(
                    "article_sarsa aggregate rows require an integer training_seed"
                )
            by_learner[(training_seed, str(row["checkpoint_digest"]))].append(row)

        context_learner_rows: list[dict[str, Any]] = []
        for (training_seed, checkpoint_digest), learner_rows in sorted(by_learner.items()):
            learner_success = [
                float(row["success_probability"]) for row in learner_rows
            ]
            learner_expansions = [
                float(value)
                for row in learner_rows
                if (value := row["conditional_successful_expansions_mean"])
                is not None
            ]
            learner_success_stats = _summary(learner_success)
            learner_expansion_stats = _summary(learner_expansions)
            learner_label = (
                f"{_canonical_json(prefix)}|{training_seed}|{checkpoint_digest}"
            )
            learner_success_ci = bootstrap_mean_ci(
                learner_success,
                stats_seed=_derived_seed(
                    stats_seed, f"{learner_label}|within-learner-success"
                ),
                samples=bootstrap_samples,
            )
            learner_expansion_ci = bootstrap_mean_ci(
                learner_expansions,
                stats_seed=_derived_seed(
                    stats_seed, f"{learner_label}|within-learner-expansions"
                ),
                samples=bootstrap_samples,
            )
            learner_row = {
                **prefix,
                "training_seed": training_seed,
                "checkpoint_digest": checkpoint_digest,
                "target_count": len(learner_rows),
                "success_rate_mean": learner_success_stats["mean"],
                "success_rate_median": learner_success_stats["median"],
                "success_rate_std": learner_success_stats["std"],
                "success_rate_ci95_low": learner_success_ci[0],
                "success_rate_ci95_high": learner_success_ci[1],
                "targets_with_success": len(learner_expansions),
                "conditional_successful_expansions_mean": learner_expansion_stats[
                    "mean"
                ],
                "conditional_successful_expansions_median": learner_expansion_stats[
                    "median"
                ],
                "conditional_successful_expansions_std": learner_expansion_stats[
                    "std"
                ],
                "conditional_successful_expansions_ci95_low": learner_expansion_ci[0],
                "conditional_successful_expansions_ci95_high": learner_expansion_ci[1],
                "statistical_unit": "held-out target within one frozen learner seed",
            }
            context_learner_rows.append(learner_row)
            learner_output.append(learner_row)

        by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_target[str(row["target_id"])].append(row)
        target_success = [
            statistics.fmean(
                float(row["success_probability"]) for row in target_rows
            )
            for _, target_rows in sorted(by_target.items())
        ]
        target_expansions: list[float] = []
        for _, target_rows in sorted(by_target.items()):
            successful = [
                float(value)
                for row in target_rows
                if (value := row["conditional_successful_expansions_mean"])
                is not None
            ]
            if successful:
                target_expansions.append(statistics.fmean(successful))
        success_stats = _summary(target_success)
        expansion_stats = _summary(target_expansions)
        learner_success_rates = [
            float(row["success_rate_mean"]) for row in context_learner_rows
        ]
        learner_expansion_rates = [
            float(value)
            for row in context_learner_rows
            if (value := row["conditional_successful_expansions_mean"]) is not None
        ]
        learner_success_stats = _summary(learner_success_rates)
        learner_expansion_stats = _summary(learner_expansion_rates)
        label = _canonical_json(prefix)
        success_ci = bootstrap_mean_ci(
            target_success,
            stats_seed=_derived_seed(stats_seed, f"{label}|learner-success"),
            samples=bootstrap_samples,
        )
        expansion_ci = bootstrap_mean_ci(
            target_expansions,
            stats_seed=_derived_seed(stats_seed, f"{label}|learner-expansions"),
            samples=bootstrap_samples,
        )
        learner_success_ci = bootstrap_mean_ci(
            learner_success_rates,
            stats_seed=_derived_seed(stats_seed, f"{label}|between-learner-success"),
            samples=bootstrap_samples,
        )
        learner_expansion_ci = bootstrap_mean_ci(
            learner_expansion_rates,
            stats_seed=_derived_seed(
                stats_seed, f"{label}|between-learner-expansions"
            ),
            samples=bootstrap_samples,
        )
        output.append(
            {
                **prefix,
                # A checkpoint instance is identified by both seed and digest:
                # independent seeds can legitimately converge to identical
                # weights and must still remain independent reported learners.
                "checkpoint_count": len(by_learner),
                "learner_seed_count": len(
                    {training_seed for training_seed, _ in by_learner}
                ),
                "unique_checkpoint_digest_count": len(
                    {checkpoint_digest for _, checkpoint_digest in by_learner}
                ),
                "training_seeds": sorted(
                    {training_seed for training_seed, _ in by_learner}
                ),
                "target_count": len(by_target),
                "target_seed_pair_count": len(rows),
                # Primary estimate: average learner outcomes inside each target,
                # then treat held-out targets as the statistical unit.
                "success_rate_mean": success_stats["mean"],
                "success_rate_median": success_stats["median"],
                "success_rate_std": success_stats["std"],
                "success_rate_ci95_low": success_ci[0],
                "success_rate_ci95_high": success_ci[1],
                "conditional_successful_expansions_mean": expansion_stats["mean"],
                "conditional_successful_expansions_median": expansion_stats["median"],
                "conditional_successful_expansions_std": expansion_stats["std"],
                "conditional_successful_expansions_ci95_low": expansion_ci[0],
                "conditional_successful_expansions_ci95_high": expansion_ci[1],
                # Between-learner columns expose checkpoint/seed variability
                # rather than mislabelling variation between target means.
                "learner_success_rate_mean": learner_success_stats["mean"],
                "learner_success_rate_median": learner_success_stats["median"],
                "learner_success_rate_std": learner_success_stats["std"],
                "learner_success_rate_ci95_low": learner_success_ci[0],
                "learner_success_rate_ci95_high": learner_success_ci[1],
                "learner_conditional_successful_expansions_mean": (
                    learner_expansion_stats["mean"]
                ),
                "learner_conditional_successful_expansions_median": (
                    learner_expansion_stats["median"]
                ),
                "learner_conditional_successful_expansions_std": (
                    learner_expansion_stats["std"]
                ),
                "learner_conditional_successful_expansions_ci95_low": (
                    learner_expansion_ci[0]
                ),
                "learner_conditional_successful_expansions_ci95_high": (
                    learner_expansion_ci[1]
                ),
                "aggregation_order": (
                    "checkpoint outcomes averaged within each target, then "
                    "targets summarized/bootstrap-resampled; each checkpoint seed "
                    "is also summarized across targets and learner variability is "
                    "reported in separately named columns"
                ),
            }
        )
    return output, learner_output


def aggregate_article_v1_runs(
    runs: Iterable[Mapping[str, Any]],
    *,
    budgets: Sequence[int] | None = None,
    stats_seed: int = DEFAULT_STATISTICS_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Aggregate raw runs with the held-out target as the statistical unit."""

    if isinstance(stats_seed, bool) or not isinstance(stats_seed, int):
        raise TypeError("stats_seed must be an integer")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    rows = [dict(run) for run in runs]
    for row in rows:
        unique_run_key(row)  # complete identity validation
    if budgets is None:
        normalized_budgets = sorted(
            {int(run_identity_payload(row)["expansion_budget"]) for row in rows}
        )
    else:
        normalized_budgets = sorted(set(int(value) for value in budgets))
        if any(value <= 0 for value in normalized_budgets):
            raise ValueError("budgets must contain only positive integers")

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    fields_by_group: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        fields = _group_fields(row)
        key = _group_key(fields)
        grouped[key].append(row)
        fields_by_group[key] = fields

    per_target: list[dict[str, Any]] = []
    for key in sorted(grouped):
        fields = fields_by_group[key]
        targets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in grouped[key]:
            target_id = str(run_identity_payload(row)["target_id"])
            targets[target_id].append(row)
        executed_caps = {
            target_id: {
                int(run_identity_payload(row)["expansion_budget"])
                for row in target_rows
            }
            for target_id, target_rows in targets.items()
        }
        for budget in normalized_budgets:
            for target_id in sorted(targets):
                if budget not in executed_caps[target_id]:
                    continue
                per_target.append(
                    _per_target_row(
                        targets[target_id],
                        target_id=target_id,
                        budget=budget,
                        fields=fields,
                    )
                )

    curve_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_target:
        curve_key = _group_key(row) + (str(row["expansion_budget"]),)
        curve_groups[curve_key].append(row)

    success_curves: list[dict[str, Any]] = []
    timing_breakdown: list[dict[str, Any]] = []
    for key in sorted(curve_groups):
        target_rows = sorted(curve_groups[key], key=lambda row: row["target_id"])
        prefix = {
            name: target_rows[0][name]
            for name in (
                "split",
                "difficulty",
                "scheduler",
                "checkpoint_digest",
                "training_seed",
                "resource_budget",
                "feature_schema",
                "feature_evaluator_schema",
                "reward_schema",
                "reward_parameters",
                "certifier_schema",
                "certification_parameters",
                "search_reduction",
                "code_version",
                "source_worktree_digest",
                "expansion_budget",
            )
        }
        success_values = [float(row["success_probability"]) for row in target_rows]
        conditional_values = [
            float(value)
            for row in target_rows
            if (value := row["conditional_successful_expansions_mean"]) is not None
        ]
        success_stats = _summary(success_values)
        conditional_stats = _summary(conditional_values)
        label = _canonical_json(prefix)
        success_ci = bootstrap_mean_ci(
            success_values,
            stats_seed=_derived_seed(stats_seed, f"{label}|success"),
            samples=bootstrap_samples,
        )
        conditional_ci = bootstrap_mean_ci(
            conditional_values,
            stats_seed=_derived_seed(stats_seed, f"{label}|expansions"),
            samples=bootstrap_samples,
        )
        success_curves.append(
            {
                **prefix,
                "target_count": len(target_rows),
                "trajectory_count": sum(int(row["trajectory_count"]) for row in target_rows),
                "success_rate": success_stats["mean"],
                "success_rate_median": success_stats["median"],
                "success_rate_std": success_stats["std"],
                "success_rate_ci95_low": success_ci[0],
                "success_rate_ci95_high": success_ci[1],
                "targets_with_success": len(conditional_values),
                "conditional_successful_expansions_mean": conditional_stats["mean"],
                "conditional_successful_expansions_median": conditional_stats["median"],
                "conditional_successful_expansions_std": conditional_stats["std"],
                "conditional_successful_expansions_ci95_low": conditional_ci[0],
                "conditional_successful_expansions_ci95_high": conditional_ci[1],
                **{
                    f"solution_{name}_mean": _mean(
                        row[f"solution_{name}_mean"] for row in target_rows
                    )
                    for name in _RESOURCE_NAMES
                },
            }
        )
        timing_breakdown.append(
            {
                **prefix,
                "target_count": len(target_rows),
                **{
                    f"{name}_mean": _mean(row[f"{name}_mean"] for row in target_rows)
                    for name in (
                        *_TIMING_PATHS,
                        "target_metric_cache_hit_rate",
                        "expansions_per_wall_second",
                    )
                },
            }
        )

    paired_differences, paired_per_target_differences = _paired_differences(
        per_target,
        stats_seed=stats_seed,
        bootstrap_samples=bootstrap_samples,
    )
    learner_seed_summary, learner_seed_results = _learner_seed_summary(
        per_target,
        stats_seed=stats_seed,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "schema_version": ARTICLE_V1_REPORT_SCHEMA,
        "raw_run_count": len(rows),
        "budgets": normalized_budgets,
        "statistics": {
            "unit": "held-out target",
            "bootstrap": "95% percentile bootstrap of target-level means",
            "stats_seed": stats_seed,
            "bootstrap_samples": bootstrap_samples,
        },
        "per_target": per_target,
        "success_curves": success_curves,
        "timing_breakdown": timing_breakdown,
        "learner_seed_summary": learner_seed_summary,
        "learner_seed_results": learner_seed_results,
        "paired_differences": paired_differences,
        "paired_per_target_differences": paired_per_target_differences,
    }


def _paired_differences(
    per_target: Sequence[Mapping[str, Any]],
    *,
    stats_seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts: dict[
        tuple[str, ...],
        dict[tuple[str, str, str], dict[str, Mapping[str, Any]]],
    ] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in per_target:
        context = tuple(
            str(row[name])
            for name in (
                "split",
                "difficulty",
                "resource_budget",
                "feature_schema",
                "feature_evaluator_schema",
                "reward_schema",
                "reward_parameters",
                "certifier_schema",
                "certification_parameters",
                "search_reduction",
                "code_version",
                "source_worktree_digest",
                "expansion_budget",
            )
        )
        scheduler = (
            str(row["scheduler"]),
            str(row["checkpoint_digest"]),
            str(row.get("training_seed")),
        )
        contexts[context][scheduler][str(row["target_id"])] = row

    output: list[dict[str, Any]] = []
    per_target_output: list[dict[str, Any]] = []
    for context in sorted(contexts):
        scheduler_rows = contexts[context]
        schedulers = sorted(scheduler_rows)
        for left_index, left in enumerate(schedulers):
            for right in schedulers[left_index + 1 :]:
                targets = sorted(set(scheduler_rows[left]) & set(scheduler_rows[right]))
                success_differences = [
                    float(scheduler_rows[left][target]["success_probability"])
                    - float(scheduler_rows[right][target]["success_probability"])
                    for target in targets
                ]
                expansion_differences = []
                for target in targets:
                    left_row = scheduler_rows[left][target]
                    right_row = scheduler_rows[right][target]
                    left_value = left_row[
                        "conditional_successful_expansions_mean"
                    ]
                    right_value = right_row[
                        "conditional_successful_expansions_mean"
                    ]
                    expansion_difference = None
                    if left_value is not None and right_value is not None:
                        expansion_difference = float(left_value) - float(right_value)
                        expansion_differences.append(expansion_difference)
                    per_target_output.append(
                        {
                            "split": context[0],
                            "difficulty": context[1],
                            "resource_budget": context[2],
                            "feature_schema": context[3],
                            "feature_evaluator_schema": context[4],
                            "reward_schema": context[5],
                            "reward_parameters": context[6],
                            "certifier_schema": context[7],
                            "certification_parameters": context[8],
                            "search_reduction": context[9],
                            "code_version": context[10],
                            "source_worktree_digest": context[11],
                            "expansion_budget": int(context[12]),
                            "target_id": target,
                            "left_scheduler": left[0],
                            "left_checkpoint_digest": left[1],
                            "left_training_seed": left[2],
                            "right_scheduler": right[0],
                            "right_checkpoint_digest": right[1],
                            "right_training_seed": right[2],
                            "left_success_probability": float(
                                left_row["success_probability"]
                            ),
                            "right_success_probability": float(
                                right_row["success_probability"]
                            ),
                            "success_difference": float(
                                left_row["success_probability"]
                            )
                            - float(right_row["success_probability"]),
                            "left_conditional_successful_expansions": left_value,
                            "right_conditional_successful_expansions": right_value,
                            "conditional_expansion_difference": expansion_difference,
                        }
                    )
                label = f"{context}|{left}|{right}"
                success_stats = _summary(success_differences)
                expansion_stats = _summary(expansion_differences)
                success_ci = bootstrap_mean_ci(
                    success_differences,
                    stats_seed=_derived_seed(stats_seed, f"{label}|success"),
                    samples=bootstrap_samples,
                )
                expansion_ci = bootstrap_mean_ci(
                    expansion_differences,
                    stats_seed=_derived_seed(stats_seed, f"{label}|expansions"),
                    samples=bootstrap_samples,
                )
                output.append(
                    {
                        "split": context[0],
                        "difficulty": context[1],
                        "resource_budget": context[2],
                        "feature_schema": context[3],
                        "feature_evaluator_schema": context[4],
                        "reward_schema": context[5],
                        "reward_parameters": context[6],
                        "certifier_schema": context[7],
                        "certification_parameters": context[8],
                        "search_reduction": context[9],
                        "code_version": context[10],
                        "source_worktree_digest": context[11],
                        "expansion_budget": int(context[12]),
                        "left_scheduler": left[0],
                        "left_checkpoint_digest": left[1],
                        "left_training_seed": left[2],
                        "right_scheduler": right[0],
                        "right_checkpoint_digest": right[1],
                        "right_training_seed": right[2],
                        "paired_target_count": len(targets),
                        "success_difference_mean": success_stats["mean"],
                        "success_difference_median": success_stats["median"],
                        "success_difference_std": success_stats["std"],
                        "success_difference_ci95_low": success_ci[0],
                        "success_difference_ci95_high": success_ci[1],
                        "paired_successful_expansion_target_count": len(
                            expansion_differences
                        ),
                        "conditional_expansion_difference_mean": expansion_stats["mean"],
                        "conditional_expansion_difference_median": expansion_stats["median"],
                        "conditional_expansion_difference_std": expansion_stats["std"],
                        "conditional_expansion_difference_ci95_low": expansion_ci[0],
                        "conditional_expansion_difference_ci95_high": expansion_ci[1],
                    }
                )
    return output, per_target_output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            _canonical_json(value)
                            if isinstance(value, (Mapping, list, tuple))
                            else ""
                            if value is None
                            else value
                        )
                        for key, value in row.items()
                    }
                )


def _portable_artifact_reference(path: Path) -> str:
    """Prefer stable POSIX paths for artifacts created below the working tree."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return relative.as_posix()


def _series_label(row: Mapping[str, Any]) -> str:
    prefix = f"{row.get('split', 'unspecified')}/{row.get('difficulty', 'unspecified')}:"
    checkpoint = str(row["checkpoint_digest"])
    if checkpoint == _NONE_CHECKPOINT:
        return prefix + str(row["scheduler"])
    short = checkpoint.split(":")[-1][:8]
    return f"{prefix}{row['scheduler']}@seed-{row.get('training_seed')}:{short}"


def _line_chart_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    title: str,
    y_label: str,
    fixed_y_max: float | None = None,
) -> str:
    width, height = 920, 520
    left, right, top, bottom = 82, 210, 58, 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get(value_field))
        budget = _as_int(row.get("expansion_budget"))
        if value is not None and budget is not None:
            grouped[_series_label(row)].append((budget, value))
    budgets = sorted({point[0] for points in grouped.values() for point in points})
    maximum_budget = max(budgets, default=1)
    maximum_value = fixed_y_max or max(
        (point[1] for points in grouped.values() for point in points),
        default=1.0,
    )
    maximum_value = max(maximum_value, 1e-12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" fill="#0f172a">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#0f172a" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#0f172a" stroke-width="2"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_height * (1.0 - fraction)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="#cbd5e1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{maximum_value * fraction:.3g}</text>'
        )
    for index, label in enumerate(sorted(grouped)):
        color = _COLORS[index % len(_COLORS)]
        points = sorted(grouped[label])
        coordinates = []
        for budget, value in points:
            x = left + chart_width * budget / maximum_budget
            y = top + chart_height * (1.0 - value / maximum_value)
            coordinates.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_y = top + 20 * index
        parts.append(
            f'<line x1="{left + chart_width + 20}" y1="{legend_y}" x2="{left + chart_width + 42}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{left + chart_width + 50}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    if not grouped:
        parts.append(
            f'<text x="{left + chart_width / 2}" y="{top + chart_height / 2}" text-anchor="middle" font-family="sans-serif">No raw records available</text>'
        )
    parts.extend(
        [
            f'<text x="{left + chart_width / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14">Expansion budget</text>',
            f'<text transform="translate(20 {top + chart_height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _summary_markdown(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# Article V1 reporting summary",
        "",
        f"- Raw completed runs: `{aggregate['raw_run_count']}`",
        f"- Statistical unit: `{aggregate['statistics']['unit']}`",
        f"- Bootstrap: `{aggregate['statistics']['bootstrap']}`",
        f"- Statistics seed / samples: `{aggregate['statistics']['stats_seed']}` / `{aggregate['statistics']['bootstrap_samples']}`",
        "",
        "## Budget-success and conditional effort",
        "",
        "| Split | Difficulty | Scheduler | Training seed | Checkpoint | Budget | Targets | Success | 95% CI | Successful-target expansion mean |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["success_curves"]:
        conditional = row["conditional_successful_expansions_mean"]
        conditional_text = "n/a" if conditional is None else f"{conditional:.4g}"
        lines.append(
            f"| {row['split']} | {row['difficulty']} | {row['scheduler']} | "
            f"{row['training_seed']} | {row['checkpoint_digest']} | "
            f"{row['expansion_budget']} | "
            f"{row['target_count']} | {row['success_rate']:.4g} | "
            f"[{row['success_rate_ci95_low']:.4g}, {row['success_rate_ci95_high']:.4g}] | "
            f"{conditional_text} |"
        )
    lines.extend(
        [
            "",
            "## Article SARSA across independent learner seeds",
            "",
            "| Split | Difficulty | Budget | Learners | Targets | Target-averaged success | Target std | Between-learner std | Target-bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["learner_seed_summary"]:
        lines.append(
            f"| {row['split']} | {row['difficulty']} | {row['expansion_budget']} | "
            f"{row['checkpoint_count']} | {row['target_count']} | "
            f"{row['success_rate_mean']:.4g} | {row['success_rate_std']:.4g} | "
            f"{row['learner_success_rate_std']:.4g} | "
            f"[{row['success_rate_ci95_low']:.4g}, {row['success_rate_ci95_high']:.4g}] |"
        )
    lines.extend(
        [
            "",
            "### Per-learner results",
            "",
            "| Split | Difficulty | Budget | Training seed | Checkpoint | Targets | Success | 95% CI |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in aggregate["learner_seed_results"]:
        lines.append(
            f"| {row['split']} | {row['difficulty']} | {row['expansion_budget']} | "
            f"{row['training_seed']} | {row['checkpoint_digest']} | "
            f"{row['target_count']} | {row['success_rate_mean']:.4g} | "
            f"[{row['success_rate_ci95_low']:.4g}, {row['success_rate_ci95_high']:.4g}] |"
        )
    lines.extend(
        [
            "",
            "## Required qualifications",
            "",
            "- Every aggregate above was regenerated from `raw_runs.jsonl`; no runner-side summary was accepted as evidence.",
            "- Failed and truncated runs remain in the raw store and in every success denominator.",
            "- Evaluation-seed repetitions, including repeated deterministic schedulers, are first averaged within each target. They are not counted as independent held-out targets.",
            "- Expansions are conditional on certification and are reported only beside the corresponding success rate.",
            "- Timing and circuit-resource summaries describe this recorded implementation and hardware context; they do not prove asymptotic scalability.",
            "- Paired differences are descriptive target-level comparisons. Confidence intervals alone are not a claim of statistical superiority.",
            "- These artifacts do not establish circuit optimality, universal RL superiority, unrestricted Toffoli discovery, or exact learned QFT synthesis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_article_v1_report(
    raw_jsonl: str | Path,
    output_dir: str | Path,
    *,
    budgets: Sequence[int] | None = None,
    stats_seed: int = DEFAULT_STATISTICS_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    expected_raw_sha256: str | None = None,
) -> dict[str, str]:
    """Rebuild all publication primitives from the raw JSONL store only."""

    reporting_started = time.perf_counter_ns()
    raw_path = Path(raw_jsonl)
    destination = Path(output_dir)
    store = AppendOnlyJSONLRunStore(raw_path)
    if expected_raw_sha256 is None:
        records = store.load_records(repair_partial=True)
    else:
        if (
            not isinstance(expected_raw_sha256, str)
            or len(expected_raw_sha256) != 71
            or not expected_raw_sha256.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in expected_raw_sha256.removeprefix("sha256:")
            )
        ):
            raise ValueError(
                "expected_raw_sha256 must use canonical sha256:<64 lowercase hex>"
            )

        def raw_digest() -> str:
            try:
                encoded = raw_path.read_bytes()
            except OSError as error:
                raise ValueError("raw JSONL is missing or unreadable") from error
            return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

        if raw_digest() != expected_raw_sha256:
            raise ValueError("raw JSONL digest differs before report loading")
        records = store.load_records(repair_partial=False)
        if raw_digest() != expected_raw_sha256:
            raise ValueError("raw JSONL digest changed during report loading")
    aggregate = aggregate_article_v1_runs(
        records,
        budgets=budgets,
        stats_seed=stats_seed,
        bootstrap_samples=bootstrap_samples,
    )
    paths = {
        "per_target": destination / "per_target.csv",
        "success_curves": destination / "success_curves.csv",
        "timing_breakdown": destination / "timing_breakdown.csv",
        "paired_differences": destination / "tables" / "paired_differences.csv",
        "paired_per_target": destination / "tables" / "paired_per_target.csv",
        "learner_seed_summary": destination / "tables" / "learner_seed_summary.csv",
        "learner_seed_results": destination / "tables" / "learner_seed_results.csv",
        "summary_table": destination / "tables" / "summary.csv",
        "summary_markdown": destination / "tables" / "summary.md",
        "success_figure": destination / "figures" / "success_curves.svg",
        "expansion_figure": destination / "figures" / "conditional_expansions.svg",
        "timing_figure": destination / "figures" / "wall_time.svg",
        "completion_summary": destination / "completion_summary.md",
        "report_metadata": destination / "report_metadata.json",
    }
    _write_csv(paths["per_target"], aggregate["per_target"])
    _write_csv(paths["success_curves"], aggregate["success_curves"])
    _write_csv(paths["timing_breakdown"], aggregate["timing_breakdown"])
    _write_csv(paths["paired_differences"], aggregate["paired_differences"])
    _write_csv(paths["learner_seed_summary"], aggregate["learner_seed_summary"])
    _write_csv(paths["learner_seed_results"], aggregate["learner_seed_results"])
    _write_csv(
        paths["paired_per_target"],
        aggregate["paired_per_target_differences"],
    )
    _write_csv(paths["summary_table"], aggregate["success_curves"])
    summary = _summary_markdown(aggregate)
    paths["summary_markdown"].write_text(summary, encoding="utf-8")
    paths["completion_summary"].write_text(summary, encoding="utf-8")
    paths["success_figure"].parent.mkdir(parents=True, exist_ok=True)
    paths["success_figure"].write_text(
        _line_chart_svg(
            aggregate["success_curves"],
            value_field="success_rate",
            title="Target-level budget-success curves",
            y_label="Success probability",
            fixed_y_max=1.0,
        ),
        encoding="utf-8",
    )
    paths["expansion_figure"].write_text(
        _line_chart_svg(
            aggregate["success_curves"],
            value_field="conditional_successful_expansions_mean",
            title="Conditional expansions on certified targets",
            y_label="Mean expansions (conditional on success)",
        ),
        encoding="utf-8",
    )
    paths["timing_figure"].write_text(
        _line_chart_svg(
            aggregate["timing_breakdown"],
            value_field="wall_time_seconds_mean",
            title="Target-level mean wall time",
            y_label="Wall time (seconds)",
        ),
        encoding="utf-8",
    )
    reporting_time_ns = time.perf_counter_ns() - reporting_started
    _atomic_write_bytes(
        paths["report_metadata"],
        (
            _canonical_json(
                {
                    "schema_version": ARTICLE_V1_REPORT_SCHEMA,
                    "raw_run_count": aggregate["raw_run_count"],
                    "statistics_seed": int(stats_seed),
                    "bootstrap_samples": int(bootstrap_samples),
                    "raw_ledger_sha256": expected_raw_sha256,
                    "raw_ledger_digest_bound": expected_raw_sha256 is not None,
                    "reporting_time_ns": int(reporting_time_ns),
                    "timing_categories_are_not_summed": True,
                }
            )
            + "\n"
        ).encode("utf-8"),
    )
    return {name: _portable_artifact_reference(path) for name, path in paths.items()}


__all__ = [
    "ARTICLE_V1_RAW_RUN_SCHEMA",
    "ARTICLE_V1_REPORT_SCHEMA",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_STATISTICS_SEED",
    "AppendOnlyJSONLRunStore",
    "ArticleV1RunStore",
    "aggregate_article_v1_runs",
    "bootstrap_mean_ci",
    "load_completed_run_keys",
    "run_identity_payload",
    "unique_run_key",
    "write_article_v1_report",
]
