"""Deterministic pilot-to-publication cost projection for Article V1.

The projection is deliberately downstream of the campaign integrity audit.  It
reads a byte-bound, passing pilot ledger, derives empirical per-run throughput
and artifact-size distributions, and combines those observations with the
mechanically enumerated publication campaign plan.  It never executes search.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import tempfile
from typing import Any

from reporting.article_v1 import AppendOnlyJSONLRunStore


ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA = "article-v1-campaign-audit-v1"
ARTICLE_V1_COST_PROJECTION_SCHEMA = "article-v1-pilot-cost-projection-v1"
SUPPORTED_SERIAL_MODE = "supported-serial"
IDEALIZED_PARALLEL_MODE = "idealized-parallel-estimate"
EXECUTION_MODES = (SUPPORTED_SERIAL_MODE, IDEALIZED_PARALLEL_MODE)
CURRENT_SUPPORTED_WORKER_COUNT = 1
RAW_RECORD_BYTES_PER_ABLATION_PLAN_ESTIMATE = 32_768
REQUIRED_CAMPAIGN_AUDIT_INTEGRITY_CHECKS = frozenset(
    {
        "terminal_newline",
        "no_blank_or_duplicate_records",
        "no_duplicate_json_members",
        "raw_schema_and_run_keys",
        "exact_expected_key_set",
        "target_ids_and_fingerprints",
        "scheduler_seed_budget_checkpoint_binding",
        "config_source_and_schema_binding",
        "type_strict_scientific_metadata",
        "no_reference_witness_fallback",
        "finite_counters_and_timings",
        "native_search_event_equations",
        "successes_independently_certified",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{name} is missing, unreadable, or invalid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return resolved


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic linearly interpolated empirical percentile."""

    if not values:
        raise ValueError("cannot summarize an empty observation sequence")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty observation sequence")
    resolved = [float(value) for value in values]
    if not all(math.isfinite(value) for value in resolved):
        raise ValueError("observations must be finite")
    return {
        "count": len(resolved),
        "minimum": min(resolved),
        "p05": _percentile(resolved, 0.05),
        "p25": _percentile(resolved, 0.25),
        "median": statistics.median(resolved),
        "mean": statistics.fmean(resolved),
        "p75": _percentile(resolved, 0.75),
        "p95": _percentile(resolved, 0.95),
        "maximum": max(resolved),
    }


def _scaled_range(
    count: int,
    distribution: Mapping[str, int | float],
    *,
    central_statistic: str,
) -> dict[str, float]:
    central = float(distribution[central_statistic])
    lower = min(float(distribution["p05"]), central)
    upper = max(float(distribution["p95"]), central)
    return {
        "lower": float(count) * lower,
        "central": float(count) * central,
        "upper": float(count) * upper,
    }


def _convert_range(
    values: Mapping[str, float], *, divisor: float
) -> dict[str, float]:
    return {
        bound: float(values[bound]) / divisor
        for bound in ("lower", "central", "upper")
    }


def _portable_relative(path: Path, root: Path, *, name: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{name} must resolve inside the pilot run directory"
        ) from error
    return relative.as_posix()


def _declared_raw_path(run_directory: Path, declared: object) -> Path:
    if not isinstance(declared, str) or not declared:
        raise ValueError("campaign audit must declare a portable raw_ledger_path")
    normalized = declared.replace("\\", "/")
    portable = PurePosixPath(normalized)
    if portable.is_absolute() or ".." in portable.parts or not portable.parts:
        raise ValueError("campaign audit raw_ledger_path must be repository-portable")
    return run_directory.joinpath(*portable.parts)


def _validate_audit(
    run_directory: Path,
    raw_path: Path,
) -> tuple[dict[str, Any], Path]:
    audit_path = run_directory / "campaign_audit.json"
    audit = _load_json_object(audit_path, name="campaign audit")
    if audit.get("schema_version") != ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA:
        raise ValueError("campaign audit uses an unsupported schema")
    if audit.get("passed") is not True:
        raise ValueError(
            "pilot cost projection requires campaign_audit.json with passed=true"
        )
    if audit.get("config_profile") != "pilot":
        raise ValueError("pilot cost projection requires an audited pilot profile")
    for field in ("config_digest", "code_version", "source_worktree_digest"):
        _nonempty_string(audit.get(field), name=f"campaign audit {field}")

    integrity_checks = audit.get("integrity_checks")
    if (
        not isinstance(integrity_checks, Mapping)
        or set(integrity_checks) != REQUIRED_CAMPAIGN_AUDIT_INTEGRITY_CHECKS
        or any(value is not True for value in integrity_checks.values())
    ):
        raise ValueError(
            "the exact campaign audit integrity checks must be present and true"
        )
    for field in ("missing_run_keys", "unexpected_run_keys", "duplicate_run_keys"):
        if audit.get(field) != []:
            raise ValueError(f"passing campaign audit must declare {field}=[]")

    expected = _positive_int(
        audit.get("expected_run_count"), name="audit expected_run_count"
    )
    observed = _positive_int(
        audit.get("observed_run_count"), name="audit observed_run_count"
    )
    if expected != observed:
        raise ValueError("passing campaign audit expected/observed run counts disagree")
    certified_success_count = _nonnegative_int(
        audit.get("independently_certified_success_count"),
        name="audit independently_certified_success_count",
    )
    if certified_success_count > observed:
        raise ValueError(
            "campaign audit independently certified more successes than records"
        )
    expected_by_split = audit.get("expected_by_split")
    observed_by_split = audit.get("observed_by_split")
    if not isinstance(expected_by_split, Mapping) or not isinstance(
        observed_by_split, Mapping
    ):
        raise ValueError("campaign audit split cardinalities are missing")
    if dict(expected_by_split) != dict(observed_by_split):
        raise ValueError("passing campaign audit split cardinalities disagree")
    if set(expected_by_split) != {"test", "ood_test"}:
        raise ValueError(
            "campaign audit split cardinalities must cover test and ood_test"
        )
    split_total = sum(
        _nonnegative_int(value, name=f"audit {split} run count")
        for split, value in expected_by_split.items()
    )
    if split_total != expected:
        raise ValueError("campaign audit split cardinalities do not sum to run count")

    declared_path = _declared_raw_path(run_directory, audit.get("raw_ledger_path"))
    if declared_path.resolve(strict=True) != raw_path.resolve(strict=True):
        raise ValueError(
            "selected raw ledger does not match the audited raw_ledger_path"
        )
    observed_digest = _sha256(raw_path)
    if audit.get("raw_ledger_sha256") != observed_digest:
        raise ValueError("raw ledger SHA-256 does not match the passing campaign audit")
    return audit, audit_path


def _load_audited_records(
    raw_path: Path,
    *,
    expected_count: int,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], list[int], int]:
    try:
        raw_bytes = raw_path.read_bytes()
    except OSError as error:
        raise ValueError(f"raw ledger is missing or unreadable: {raw_path}") from error
    if f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}" != expected_sha256:
        raise ValueError("raw ledger changed before audited record loading")
    if not raw_bytes or not raw_bytes.endswith(b"\n"):
        raise ValueError("audited raw ledger must be nonempty and newline-complete")
    records = AppendOnlyJSONLRunStore(raw_path).load_records(repair_partial=False)
    if _sha256(raw_path) != expected_sha256:
        raise ValueError("raw ledger changed during audited record loading")
    if len(records) != expected_count:
        raise ValueError("raw ledger record count no longer matches the passing audit")
    encoded_records = [
        line for line in raw_bytes.splitlines(keepends=True) if line.strip()
    ]
    if len(encoded_records) != len(records):
        raise ValueError("raw ledger physical records disagree with validated run keys")
    return records, [len(line) for line in encoded_records], len(raw_bytes)


def _observed_measurements(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runtimes: list[float] = []
    expansions: list[int] = []
    throughputs: list[float] = []
    seconds_per_expansion: list[float] = []
    by_scheduler: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    runtime_by_scheduler_difficulty: dict[str, dict[str, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    throughput_by_scheduler_difficulty: dict[str, dict[str, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for index, record in enumerate(records):
        expansion_count = _positive_int(
            record.get("expansions"), name=f"raw record {index} expansions"
        )
        runtime = _positive_float(
            record.get("runtime_seconds"), name=f"raw record {index} runtime_seconds"
        )
        scheduler = record.get("scheduler")
        if not isinstance(scheduler, str) or not scheduler:
            raise ValueError(f"raw record {index} has no scheduler label")
        difficulty = record.get("difficulty", record.get("stratum", "unknown"))
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError(f"raw record {index} has no difficulty label")

        throughput = float(expansion_count) / runtime
        seconds_per = runtime / float(expansion_count)
        runtimes.append(runtime)
        expansions.append(expansion_count)
        throughputs.append(throughput)
        seconds_per_expansion.append(seconds_per)
        by_scheduler[scheduler].append(throughput)
        by_difficulty[difficulty].append(throughput)
        runtime_by_scheduler_difficulty[scheduler][difficulty].append(runtime)
        throughput_by_scheduler_difficulty[scheduler][difficulty].append(throughput)

    return {
        "raw_record_count": len(records),
        "total_runtime_seconds": math.fsum(runtimes),
        "total_expansions": sum(expansions),
        "runtime_seconds_per_record": _distribution(runtimes),
        "expansions_per_record": _distribution(expansions),
        "expansions_per_second": _distribution(throughputs),
        "seconds_per_expansion": _distribution(seconds_per_expansion),
        "expansion_weighted_seconds_per_expansion": (
            math.fsum(runtimes) / float(sum(expansions))
        ),
        "maximum_expansion_throughput": max(throughputs),
        "expansions_per_second_by_scheduler": {
            name: _distribution(by_scheduler[name]) for name in sorted(by_scheduler)
        },
        "expansions_per_second_by_difficulty": {
            name: _distribution(by_difficulty[name]) for name in sorted(by_difficulty)
        },
        "runtime_seconds_by_scheduler_and_difficulty": {
            scheduler: {
                difficulty: _distribution(
                    runtime_by_scheduler_difficulty[scheduler][difficulty]
                )
                for difficulty in sorted(runtime_by_scheduler_difficulty[scheduler])
            }
            for scheduler in sorted(runtime_by_scheduler_difficulty)
        },
        "expansions_per_second_by_scheduler_and_difficulty": {
            scheduler: {
                difficulty: _distribution(
                    throughput_by_scheduler_difficulty[scheduler][difficulty]
                )
                for difficulty in sorted(throughput_by_scheduler_difficulty[scheduler])
            }
            for scheduler in sorted(throughput_by_scheduler_difficulty)
        },
    }


def _checkpoint_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    from experiments.article_v1_ablations import (
        ARTICLE_V1_ABLATION_REGISTRY,
        REQUIRED_ABLATION_IDS,
    )

    breakdown = plan["learner_checkpoint_breakdown"]
    if not isinstance(breakdown, Mapping):
        raise ValueError("campaign plan checkpoint breakdown is invalid")
    standard = _positive_int(breakdown.get("standard"), name="standard checkpoints")
    ood_length = _positive_int(breakdown.get("ood_length"), name="OOD checkpoints")
    if standard != ood_length:
        raise ValueError("standard and OOD checkpoint seed counts disagree")
    trained_ablation_ids = [
        ablation_id
        for ablation_id in REQUIRED_ABLATION_IDS
        if ARTICLE_V1_ABLATION_REGISTRY[ablation_id].checkpoint_mode == "train_variant"
    ]
    ablation_variants = standard * len(trained_ablation_ids)
    return {
        "standard": standard,
        "ood_length": ood_length,
        "trained_ablation_variants": ablation_variants,
        "trained_ablation_ids": trained_ablation_ids,
        "total": standard + ood_length + ablation_variants,
        "reused_primary_ablations_create_new_files": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def project_pilot_cost(
    pilot_run_directory: str | Path,
    publication_config: str | Path,
    *,
    observed_pilot_peak_rss_bytes: int,
    worker_count: int = CURRENT_SUPPORTED_WORKER_COUNT,
    execution_mode: str = SUPPORTED_SERIAL_MODE,
    raw_ledger: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and persist a deterministic, audit-gated cost projection."""

    workers = _positive_int(worker_count, name="worker_count")
    peak_rss = _positive_int(
        observed_pilot_peak_rss_bytes,
        name="observed_pilot_peak_rss_bytes",
    )
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {EXECUTION_MODES!r}")
    if (
        workers != CURRENT_SUPPORTED_WORKER_COUNT
        and execution_mode != IDEALIZED_PARALLEL_MODE
    ):
        raise ValueError(
            "worker_count > 1 is not a supported execution mode; label it "
            "idealized-parallel-estimate explicitly"
        )

    run_directory = Path(pilot_run_directory)
    if not run_directory.is_dir():
        raise ValueError("pilot_run_directory must be an existing directory")
    audit_preview = _load_json_object(
        run_directory / "campaign_audit.json", name="campaign audit"
    )
    audited_default = _declared_raw_path(
        run_directory, audit_preview.get("raw_ledger_path")
    )
    raw_path = audited_default if raw_ledger is None else Path(raw_ledger)
    audit, audit_path = _validate_audit(run_directory, raw_path)
    audited_raw_sha256 = _nonempty_string(
        audit.get("raw_ledger_sha256"), name="campaign audit raw_ledger_sha256"
    )
    records, raw_line_sizes, audited_raw_bytes = _load_audited_records(
        raw_path,
        expected_count=_positive_int(
            audit["observed_run_count"], name="audit observed_run_count"
        ),
        expected_sha256=audited_raw_sha256,
    )
    observed = _observed_measurements(records)

    checkpoint_paths = sorted(
        path
        for path in (run_directory / "checkpoints").rglob("*.json")
        if path.is_file()
    )
    if not checkpoint_paths:
        raise ValueError("audited pilot run has no checkpoint JSON files to project")
    checkpoint_sizes = [path.stat().st_size for path in checkpoint_paths]
    if any(size <= 0 for size in checkpoint_sizes):
        raise ValueError("audited pilot checkpoint files must be nonempty")

    # Imported lazily to keep this reporting module out of the runner's import
    # graph.  campaign_plan is no-execution and is the authoritative mechanical
    # source for publication cardinalities and worst-case expansion counts.
    from experiments.article_v1_runner import campaign_plan

    plan = campaign_plan(publication_config, worker_count=workers)
    if plan.get("executes_search") is not False:
        raise ValueError("publication campaign plan unexpectedly executes search")
    if plan.get("config_profile") != "publication":
        raise ValueError("publication_config must select the publication profile")

    worst_case_expansions = _positive_int(
        plan.get("worst_case_expansion_count"),
        name="publication worst_case_expansion_count",
    )
    raw_record_count = _positive_int(
        plan.get("expected_raw_ledger_keys"),
        name="publication expected_raw_ledger_keys",
    )
    ablation_record_count = _nonnegative_int(
        plan.get("ablation_run_count"), name="publication ablation_run_count"
    )
    checkpoint_projection = _checkpoint_projection(plan)

    seconds_per_expansion = observed["seconds_per_expansion"]
    cpu_seconds = _scaled_range(
        worst_case_expansions,
        seconds_per_expansion,
        central_statistic="median",
    )
    cpu_hours = _convert_range(cpu_seconds, divisor=3600.0)
    wall_seconds = _convert_range(cpu_seconds, divisor=float(workers))
    wall_hours = _convert_range(wall_seconds, divisor=3600.0)
    wall_days = _convert_range(wall_hours, divisor=24.0)

    raw_size_distribution = _distribution(raw_line_sizes)
    checkpoint_size_distribution = _distribution(checkpoint_sizes)
    raw_disk = _scaled_range(
        raw_record_count, raw_size_distribution, central_statistic="mean"
    )
    checkpoint_disk = _scaled_range(
        int(checkpoint_projection["total"]),
        checkpoint_size_distribution,
        central_statistic="mean",
    )
    ablation_disk = float(
        ablation_record_count * RAW_RECORD_BYTES_PER_ABLATION_PLAN_ESTIMATE
    )
    total_disk = {
        bound: raw_disk[bound] + checkpoint_disk[bound] + ablation_disk
        for bound in ("lower", "central", "upper")
    }

    raw_relative = _portable_relative(raw_path, run_directory, name="raw ledger")
    checkpoint_relative = [
        _portable_relative(path, run_directory, name="checkpoint")
        for path in checkpoint_paths
    ]
    result: dict[str, Any] = {
        "schema_version": ARTICLE_V1_COST_PROJECTION_SCHEMA,
        "executes_search": False,
        "source": {
            "campaign_audit_path": _portable_relative(
                audit_path, run_directory, name="campaign audit"
            ),
            "campaign_audit_schema": audit["schema_version"],
            "campaign_audit_sha256": _sha256(audit_path),
            "raw_ledger_path": raw_relative,
            "raw_ledger_sha256": audited_raw_sha256,
            "pilot_config_profile": audit["config_profile"],
            "pilot_config_digest": audit.get("config_digest"),
            "pilot_code_version": audit.get("code_version"),
            "pilot_source_worktree_digest": audit.get("source_worktree_digest"),
            "publication_config_profile": plan["config_profile"],
            "publication_config_digest": plan["config_digest"],
            "publication_campaign_plan_schema": plan["schema_version"],
        },
        "observed_pilot": {
            **observed,
            "raw_ledger_bytes": audited_raw_bytes,
            "raw_record_bytes": raw_size_distribution,
            "checkpoint_count": len(checkpoint_paths),
            "checkpoint_paths": checkpoint_relative,
            "checkpoint_bytes": checkpoint_size_distribution,
            "observed_peak_rss_bytes": peak_rss,
        },
        "publication_cardinalities": {
            "target_counts": plan["target_counts"],
            "training_episodes": plan["training_episodes"],
            "standard_test_run_count": plan["standard_test_run_count"],
            "ood_run_count": plan["ood_run_count"],
            "validation_run_count": plan["validation_run_count"],
            "ablation_run_count": ablation_record_count,
            "worst_case_expansion_count": worst_case_expansions,
            "worst_case_expansion_breakdown": plan[
                "worst_case_expansion_breakdown"
            ],
        },
        "execution": {
            "worker_count": workers,
            "execution_mode": execution_mode,
            "current_supported_worker_count": CURRENT_SUPPORTED_WORKER_COUNT,
            "parallel_efficiency_assumption": (
                "not-applicable-serial"
                if workers == CURRENT_SUPPORTED_WORKER_COUNT
                else "ideal-100-percent; estimate-only-not-supported-execution"
            ),
        },
        "projected_cpu_hours": {
            **cpu_hours,
            "basis": (
                "publication worst-case expansions multiplied by the empirical "
                "pilot per-record seconds-per-expansion distribution"
            ),
        },
        "projected_wall_time": {
            "seconds": wall_seconds,
            "hours": wall_hours,
            "days": wall_days,
            "worker_count": workers,
            "execution_mode": execution_mode,
        },
        "projected_disk_use": {
            "bytes": total_disk,
            "gibibytes": _convert_range(total_disk, divisor=float(1 << 30)),
            "components": {
                "primary_raw_ledger_bytes": raw_disk,
                "ablation_record_bytes": {
                    "lower": ablation_disk,
                    "central": ablation_disk,
                    "upper": ablation_disk,
                },
                "checkpoint_bytes": checkpoint_disk,
            },
            "scope": (
                "primary raw JSONL, required ablation records, and checkpoint JSON; "
                "tables, figures, manifests, logs, and temporary files excluded"
            ),
        },
        "projected_maximum_per_process_ram": {
            "bytes": peak_rss,
            "gibibytes": float(peak_rss) / float(1 << 30),
            "method": "carry forward the externally observed pilot peak RSS",
            "interpretation": "empirical lower-bound estimate without a scaling model",
        },
        "projected_raw_record_count": raw_record_count,
        "projected_ablation_record_count": ablation_record_count,
        "projected_checkpoint_count": int(checkpoint_projection["total"]),
        "projected_checkpoint_count_breakdown": checkpoint_projection,
        "assumptions": [
            "campaign_audit.json passed and byte-binds the exact raw ledger used",
            (
                "the publication campaign uses the mechanically enumerated "
                "checked-in plan"
            ),
            (
                "pilot evaluation seconds per expansion approximate training, "
                "validation, test, OOD, and ablation expansion cost"
            ),
            (
                "CPU projection uses worst-case configured expansion caps; early "
                "success or frontier exhaustion can reduce actual cost"
            ),
            (
                "the central runtime estimate uses the per-record median seconds "
                "per expansion"
            ),
            (
                "disk scaling uses observed pilot JSONL/checkpoint bytes and the "
                "campaign-plan ablation-record allowance"
            ),
            (
                "peak RSS is carried forward per process without claiming a "
                "publication-scale memory upper bound"
            ),
        ],
        "uncertainty": {
            "runtime_interval": (
                "empirical per-record p05-p95 seconds-per-expansion range; not a "
                "confidence or prediction interval"
            ),
            "disk_interval": (
                "empirical p05-p95 raw-record and checkpoint byte sizes; excludes "
                "nonenumerated artifacts"
            ),
            "parallel_wall_time": (
                "worker counts above one assume perfect scaling and are explicitly "
                "labelled estimate-only"
            ),
            "memory": (
                "one externally supplied pilot peak RSS observation; publication "
                "targets may require more memory"
            ),
        },
    }

    destination = (
        run_directory / "pilot_cost_projection.json"
        if output_path is None
        else Path(output_path)
    )
    protected = {audit_path.resolve(), raw_path.resolve()}
    if destination.resolve() in protected:
        raise ValueError("projection output may not overwrite the audit or raw ledger")
    _atomic_json(destination, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run-dir", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path)
    parser.add_argument("--publication-config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=CURRENT_SUPPORTED_WORKER_COUNT)
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default=SUPPORTED_SERIAL_MODE,
    )
    parser.add_argument(
        "--observed-pilot-peak-rss-bytes", type=int, required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = project_pilot_cost(
        args.pilot_run_dir,
        args.publication_config,
        observed_pilot_peak_rss_bytes=args.observed_pilot_peak_rss_bytes,
        worker_count=args.workers,
        execution_mode=args.execution_mode,
        raw_ledger=args.raw_ledger,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "ARTICLE_V1_COST_PROJECTION_SCHEMA",
    "CURRENT_SUPPORTED_WORKER_COUNT",
    "IDEALIZED_PARALLEL_MODE",
    "REQUIRED_CAMPAIGN_AUDIT_INTEGRITY_CHECKS",
    "SUPPORTED_SERIAL_MODE",
    "main",
    "project_pilot_cost",
]
