"""Qualification harness for the Article V1 feature evaluator.

This module deliberately does not own a command-line entry point.  The search
runner supplies measurements through :class:`FeatureBenchmarkAdapter`; this
module validates those measurements, applies the preregistered engineering
gates, fits clearly labelled extrapolations, and writes the performance
artifact bundle.

The outputs are engineering diagnostics.  They are never scheduler-comparison
observations and must not be appended to the Article V1 scientific raw ledger.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import statistics
import sys
import tempfile
import time
from typing import Any, Protocol, runtime_checkable
import xml.etree.ElementTree as ElementTree

import numpy as np


ARTICLE_V1_FEATURE_BENCHMARK_SCHEMA = "article-v1-feature-benchmark-v1"
ARTICLE_V1_FEATURE_BASELINE_SCHEMA = "article-v1-feature-baseline-v1"
ARTICLE_V1_FEATURE_PROJECTION_SCHEMA = "article-v1-feature-cost-projection-v1"
ARTICLE_V1_FEATURE_EVALUATOR_SCHEMA = "article-v1-exact-incremental-v2"
ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA = "article-v1-reference-all-pairs-v1"

ENGINEERING_EVIDENCE_CLASS = "engineering-performance-diagnostic"
DEFAULT_FRONTIER_SIZES = (32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_STAGED_EXPANSION_CAPS = (32, 64, 128, 256, 512, 1024)
DEFAULT_REFERENCE_SAFE_FRONTIER_SIZE = 1024
DEFAULT_REFERENCE_TRACE_CAPS = (32, 64)
DEFAULT_HARD_EXPANSION_CAP = 8192
PRODUCTION_DOMINANCE_IMPLEMENTATION_CHECK = (
    "no_python_nested_frontier_record_loop"
)
PILOT_HARD_3Q_TARGET_ID = (
    "sha256:dfd960b7be1309661b720bb31eaf4fd97589b52fd3b11c7f25eb68dada3dafbf"
)

MINIMUM_SPEEDUP_AT_1024 = 10.0
MINIMUM_END_TO_END_SPEEDUP_AT_REFERENCE_CAP = 2.0
MAXIMUM_512_TO_1024_COMPACT_SCORE_RATIO = 2.5
MAXIMUM_MEMORY_SCALING_EXPONENT = 1.25

REQUIRED_CORRECTNESS_CHECKS = (
    "snapshot_equivalence",
    "score_equivalence",
    "selected_record_equivalence",
    "sarsa_trace_equivalence",
    "final_weight_equivalence",
    "terminal_status_equivalence",
    "witness_certification_equivalence",
)

# These are deliberately exact, reviewable pytest selections rather than a
# blanket success bit copied onto every semantic check.  A command run parses
# JUnit cases back into the individual CorrectnessGate fields below.
DEFAULT_CORRECTNESS_TEST_NODE_IDS = (
    "tests/article_v1/test_incremental_feature_index.py",
    "tests/article_v1/test_compact_linear_scoring.py",
    "tests/article_v1/test_search_trace_equivalence.py::"
    "test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64",
    "tests/article_v1/test_feature_benchmark_gate.py::"
    "test_reference_and_optimized_success_witnesses_certify_identically",
)

_CORRECTNESS_TEST_IDENTITIES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "snapshot_equivalence": (
        (
            "tests.article_v1.test_incremental_feature_index",
            "test_incremental_snapshot_is_exactly_equal_to_all_pairs_oracle",
        ),
    ),
    "score_equivalence": (
        (
            "tests.article_v1.test_compact_linear_scoring",
            "test_compact_effective_weight_scores_equal_explicit_full_feature_dot_products",
        ),
    ),
    "selected_record_equivalence": (
        (
            "tests.article_v1.test_compact_linear_scoring",
            "test_selected_row_materialization_matches_reference_31d_exactly",
        ),
    ),
    "sarsa_trace_equivalence": (
        (
            "tests.article_v1.test_search_trace_equivalence",
            "test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64",
        ),
    ),
    "final_weight_equivalence": (
        (
            "tests.article_v1.test_search_trace_equivalence",
            "test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64",
        ),
    ),
    "terminal_status_equivalence": (
        (
            "tests.article_v1.test_search_trace_equivalence",
            "test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64",
        ),
    ),
    "witness_certification_equivalence": (
        (
            "tests.article_v1.test_feature_benchmark_gate",
            "test_reference_and_optimized_success_witnesses_certify_identically",
        ),
    ),
}

REQUIRED_PILOT_RELAUNCH_CHECKS = (
    "source_revision_committed_and_clean",
    "structured_progress_verified",
    "checkpoint_recovery_verified",
    "new_schema_mini_ci_passed_twice_byte_stable",
    "no_held_out_publication_test_outcomes_inspected",
)

MICROBENCHMARK_COLUMNS = (
    "frontier_size",
    "repetitions",
    "reference_expected",
    "reference_measured",
    "reference_total_seconds",
    "reference_basis",
    "optimized_synchronization_seconds",
    "optimized_compact_batch_seconds",
    "optimized_scoring_seconds",
    "optimized_selected_row_seconds",
    "optimized_compact_plus_score_seconds",
    "optimized_total_decision_seconds",
    "effective_reference_seconds",
    "speedup",
    "feature_index_memory_bytes",
    "process_peak_rss_bytes",
    "unique_resource_groups",
    "correctness_gate_passed",
    "reference_safe_frontier_size",
    "notes",
)

END_TO_END_COLUMNS = (
    "expansion_cap",
    "expansions_completed",
    "runtime_seconds",
    "feature_time_seconds",
    "feature_time_share",
    "peak_frontier",
    "peak_unique_resource_groups",
    "peak_feature_index_memory_bytes",
    "terminal_status",
    "reference_expected",
    "reference_runtime_seconds",
    "reference_feature_time_seconds",
    "reference_feature_time_share",
    "end_to_end_speedup",
    "trace_equivalent",
    "final_weights_equivalent",
    "terminal_status_equivalent",
    "deterministic_counters_equivalent",
    "reference_parity_passed",
    "notes",
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return resolved


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return resolved


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return resolved


def _positive_float(value: object, *, name: str) -> float:
    resolved = _nonnegative_float(value, name=name)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return resolved


def _optional_positive_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _optional_nonnegative_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_float(value, name=name)


def _optional_nonnegative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name=name)


def _optional_bool(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool or None")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _json_compatible_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible_copy(item) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_compatible_copy(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_compatible_copy(item())
    return value


def _validated_axis(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    resolved = tuple(_positive_int(value, name=f"{name} entry") for value in values)
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return resolved


def _benchmark_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "bound": False,
            "note": "integration caller did not supply source provenance",
        }
    if not isinstance(value, Mapping):
        raise ValueError("benchmark_provenance must be a mapping or None")
    copied = dict(value)
    _nonempty_string(copied.get("code_version"), name="provenance code_version")
    _nonempty_string(
        copied.get("source_worktree_digest"),
        name="provenance source_worktree_digest",
    )
    _strict_bool(copied.get("worktree_clean"), name="provenance worktree_clean")
    copied["bound"] = True
    return copied


@dataclass(frozen=True, slots=True)
class CorrectnessGate:
    """Reference-equivalence evidence that gates performance qualification."""

    checks: Mapping[str, bool]
    command: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.checks, Mapping):
            raise ValueError("correctness checks must be a mapping")
        copied: dict[str, bool] = {}
        for name, value in self.checks.items():
            key = _nonempty_string(name, name="correctness check name")
            copied[key] = _strict_bool(value, name=f"correctness check {key}")
        object.__setattr__(self, "checks", copied)
        _nonempty_string(self.command, name="correctness command")
        if any(not isinstance(item, str) or not item for item in self.evidence):
            raise ValueError("correctness evidence entries must be nonempty strings")

    @property
    def missing_checks(self) -> tuple[str, ...]:
        return tuple(name for name in REQUIRED_CORRECTNESS_CHECKS if name not in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in REQUIRED_CORRECTNESS_CHECKS
            if self.checks.get(name) is not True
        )

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "required_checks": list(REQUIRED_CORRECTNESS_CHECKS),
            "checks": {name: self.checks[name] for name in sorted(self.checks)},
            "missing_checks": list(self.missing_checks),
            "failed_checks": list(self.failed_checks),
            "command": self.command,
            "evidence": list(self.evidence),
        }


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _pytest_junit_cases(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Parse the small, inspectable subset of JUnit needed by the gate."""

    if not path.is_file() or path.stat().st_size == 0:
        return [], "pytest did not produce a nonempty JUnit document"
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError) as exc:
        return [], f"could not parse pytest JUnit evidence: {exc}"
    cases: list[dict[str, Any]] = []
    for testcase in root.iter():
        if _xml_local_name(testcase.tag) != "testcase":
            continue
        status = "passed"
        detail = ""
        for child in testcase:
            child_name = _xml_local_name(child.tag)
            if child_name in {"failure", "error", "skipped"}:
                status = child_name
                detail = str(child.attrib.get("message", ""))
                break
        raw_time = testcase.attrib.get("time")
        try:
            duration_seconds = None if raw_time is None else float(raw_time)
        except ValueError:
            duration_seconds = None
        cases.append(
            {
                "classname": str(testcase.attrib.get("classname", "")),
                "name": str(testcase.attrib.get("name", "")),
                "status": status,
                "duration_seconds": duration_seconds,
                "detail": detail,
            }
        )
    if not cases:
        return [], "pytest JUnit evidence contained no test cases"
    return cases, None


def _matching_junit_cases(
    cases: Sequence[Mapping[str, Any]],
    identity: tuple[str, str],
) -> list[Mapping[str, Any]]:
    classname, function_name = identity
    return [
        case
        for case in cases
        if case.get("classname") == classname
        and (
            case.get("name") == function_name
            or str(case.get("name", "")).startswith(f"{function_name}[")
        )
    ]


def run_focused_correctness_gate(
    output_directory: str | Path,
    *,
    repository_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    test_node_ids: Sequence[str] = DEFAULT_CORRECTNESS_TEST_NODE_IDS,
    timeout_seconds: float = 300.0,
    evidence_binding: Mapping[str, Any] | None = None,
) -> tuple[CorrectnessGate, dict[str, Any]]:
    """Run and parse the exact reference-equivalence pytest gate.

    Every gate field comes from its mapped JUnit case(s).  A nonzero pytest
    return code also fails every field, so an unrelated failure in this exact
    focused invocation can never be hidden by the mapped cases passing.
    Evidence is retained under ``profiles/`` even when the gate fails.
    """

    output = Path(output_directory).resolve()
    profiles = output / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    executable = str(
        Path(sys.executable if python_executable is None else python_executable)
    )
    selected = tuple(
        _nonempty_string(item, name="correctness pytest node ID")
        for item in test_node_ids
    )
    if not selected:
        raise ValueError("correctness pytest node IDs must not be empty")
    timeout = _positive_float(timeout_seconds, name="correctness timeout_seconds")

    junit_path = profiles / "correctness_gate.junit.xml"
    stdout_path = profiles / "correctness_gate.stdout.txt"
    stderr_path = profiles / "correctness_gate.stderr.txt"
    report_path = profiles / "correctness_gate.json"
    # Never parse a stale success document after pytest failed before writing.
    junit_path.unlink(missing_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".correctness-gate-",
        suffix=".xml",
        dir=profiles,
    )
    os.close(descriptor)
    temporary_junit = Path(temporary_name)
    command = (
        executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "--tb=short",
        f"--junitxml={temporary_junit}",
        *selected,
    )
    command_text = subprocess.list2cmdline(list(command))
    started = time.perf_counter()
    returncode: int | None = None
    timed_out = False
    execution_error: str | None = None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        execution_error = f"focused correctness pytest timed out after {timeout:g}s"
        stdout = "" if exc.stdout is None else str(exc.stdout)
        stderr = "" if exc.stderr is None else str(exc.stderr)
    except OSError as exc:
        execution_error = f"could not execute focused correctness pytest: {exc}"
    duration_seconds = float(time.perf_counter() - started)
    if temporary_junit.is_file() and temporary_junit.stat().st_size > 0:
        os.replace(temporary_junit, junit_path)
    else:
        temporary_junit.unlink(missing_ok=True)
    _atomic_write(stdout_path, stdout.encode("utf-8", errors="replace"))
    _atomic_write(stderr_path, stderr.encode("utf-8", errors="replace"))

    cases, parse_error = _pytest_junit_cases(junit_path)
    if parse_error is not None and execution_error is None:
        execution_error = parse_error
    suite_passed = returncode == 0 and not timed_out and execution_error is None
    checks: dict[str, bool] = {}
    check_evidence: dict[str, Any] = {}
    for check_name in REQUIRED_CORRECTNESS_CHECKS:
        identities = _CORRECTNESS_TEST_IDENTITIES.get(check_name, ())
        identity_evidence: list[dict[str, Any]] = []
        identity_passes: list[bool] = []
        for identity in identities:
            matches = _matching_junit_cases(cases, identity)
            identity_passed = bool(matches) and all(
                match.get("status") == "passed" for match in matches
            )
            identity_passes.append(identity_passed)
            identity_evidence.append(
                {
                    "classname": identity[0],
                    "test_name_prefix": identity[1],
                    "matched_cases": [dict(match) for match in matches],
                    "passed": identity_passed,
                }
            )
        checks[check_name] = bool(
            suite_passed and identities and all(identity_passes)
        )
        check_evidence[check_name] = identity_evidence

    evidence_paths = [stdout_path, stderr_path, report_path]
    if junit_path.is_file():
        evidence_paths.insert(0, junit_path)
    gate = CorrectnessGate(
        checks=checks,
        command=command_text,
        evidence=tuple(str(path) for path in evidence_paths),
    )
    junit_digest = (
        f"sha256:{sha256(junit_path.read_bytes()).hexdigest()}"
        if junit_path.is_file()
        else None
    )
    report: dict[str, Any] = {
        "schema_version": "article-v1-feature-correctness-pytest-v1",
        "passed": gate.passed,
        "command": command_text,
        "argv": list(command),
        "repository_root": str(root),
        "evidence_binding": _json_compatible_copy(evidence_binding or {}),
        "returncode": returncode,
        "timed_out": timed_out,
        "execution_error": execution_error,
        "duration_seconds": duration_seconds,
        "junit": {
            "path": str(junit_path),
            "sha256": junit_digest,
            "case_count": len(cases),
        },
        "cases": cases,
        "check_evidence": check_evidence,
        "correctness_gate": gate.as_dict(),
    }
    _atomic_write(report_path, _json_bytes(report))
    return gate, report


_PYTHON_LOOP_NODE_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _call_attribute(node: ast.Call) -> tuple[str | None, str | None]:
    function = node.func
    if not isinstance(function, ast.Attribute):
        return None, None
    owner = function.value
    if isinstance(owner, ast.Name):
        return owner.id, function.attr
    if isinstance(owner, ast.Attribute):
        return owner.attr, function.attr
    return None, function.attr


def _nested_python_loop_lines(method: ast.AST) -> list[tuple[int, int]]:
    nested: list[tuple[int, int]] = []
    loops = [node for node in ast.walk(method) if isinstance(node, _PYTHON_LOOP_NODE_TYPES)]
    for outer in loops:
        for descendant in ast.walk(outer):
            if descendant is outer:
                continue
            if isinstance(descendant, _PYTHON_LOOP_NODE_TYPES):
                nested.append(
                    (int(getattr(outer, "lineno", 0)), int(getattr(descendant, "lineno", 0)))
                )
    return sorted(set(nested))


def _same_class_method_calls(method: ast.AST) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "self":
            calls.add(node.func.attr)
    return calls


def inspect_production_dominance_update(
    source_path: str | Path | None = None,
    *,
    evidence_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservatively verify vectorized production dominance updates by AST.

    Only ``_insert_resource`` and ``_remove_resource`` are qualified.  The
    deliberately quadratic reference evaluator and debug reconciliation
    oracle are outside this production check.  The check is intentionally
    stricter than the plan: any Python loop/comprehension in either update
    fails, not merely a recognizable nested all-record loop.
    """

    path = (
        Path(__file__).resolve().parents[1] / "rl" / "article_frontier_index.py"
        if source_path is None
        else Path(source_path).resolve()
    )
    required_methods = ("_insert_resource", "_remove_resource")
    maximum_call_depth = 8
    report: dict[str, Any] = {
        "schema_version": "article-v1-dominance-source-check-v1",
        "source_path": str(path),
        "source_sha256": None,
        "qualified_class": "ExactArticleFrontierFeatureIndex",
        "qualified_methods": list(required_methods),
        "maximum_same_class_call_depth": maximum_call_depth,
        "debug_or_reference_oracles_excluded": True,
        "evidence_binding": _json_compatible_copy(evidence_binding or {}),
        "methods": {},
        "checks": {},
        "passed": False,
        "error": None,
    }
    try:
        source = path.read_text(encoding="utf-8")
        report["source_sha256"] = f"sha256:{sha256(source.encode('utf-8')).hexdigest()}"
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        report["error"] = f"could not inspect dominance source: {exc}"
        return report

    class_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ExactArticleFrontierFeatureIndex"
    ]
    all_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    if len(class_nodes) == 1:
        all_methods = {
            node.name: node
            for node in class_nodes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    roots_found = {
        name: all_methods[name] for name in required_methods if name in all_methods
    }
    reachable_depths: dict[str, int] = {}
    call_graph: dict[str, list[str]] = {}
    queue: list[tuple[str, int]] = [(name, 0) for name in required_methods]
    call_graph_truncated = False
    while queue:
        name, depth = queue.pop(0)
        if name not in all_methods:
            continue
        prior_depth = reachable_depths.get(name)
        if prior_depth is not None and prior_depth <= depth:
            continue
        reachable_depths[name] = depth
        callees = sorted(
            candidate
            for candidate in _same_class_method_calls(all_methods[name])
            if candidate in all_methods
        )
        call_graph[name] = callees
        if depth >= maximum_call_depth:
            if any(candidate not in reachable_depths for candidate in callees):
                call_graph_truncated = True
            continue
        queue.extend((candidate, depth + 1) for candidate in callees)

    root_loop_count = 0
    reachable_nested_count = 0
    method_reports: dict[str, Any] = {}
    for name in sorted(reachable_depths, key=lambda item: (reachable_depths[item], item)):
        method = all_methods[name]
        loop_nodes = [
            node
            for node in ast.walk(method)
            if isinstance(node, _PYTHON_LOOP_NODE_TYPES)
        ]
        nested_lines = _nested_python_loop_lines(method)
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        numpy_all_lines = [
            int(getattr(call, "lineno", 0))
            for call in calls
            if _call_attribute(call) == ("np", "all")
        ]
        active_group_lines = [
            int(getattr(call, "lineno", 0))
            for call in calls
            if _call_attribute(call)[1] == "_active_group_indices"
        ]
        if name in required_methods:
            root_loop_count += len(loop_nodes)
        reachable_nested_count += len(nested_lines)
        method_reports[name] = {
            "found": True,
            "call_depth": reachable_depths[name],
            "same_class_callees": call_graph.get(name, []),
            "line_start": int(method.lineno),
            "line_end": int(getattr(method, "end_lineno", method.lineno)),
            "python_loop_lines": sorted(
                int(getattr(node, "lineno", 0)) for node in loop_nodes
            ),
            "nested_python_loop_lines": [list(item) for item in nested_lines],
            "numpy_all_lines": numpy_all_lines,
            "active_group_index_lines": active_group_lines,
    }
    report["methods"] = method_reports
    report["same_class_call_graph"] = call_graph
    report["reachable_methods"] = [
        name
        for name in sorted(
            reachable_depths, key=lambda item: (reachable_depths[item], item)
        )
    ]
    report["call_graph_truncated"] = call_graph_truncated
    checks = {
        "single_production_index_class_found": len(class_nodes) == 1,
        "production_update_methods_found": set(roots_found) == set(required_methods),
        "same_class_call_graph_fully_inspected": not call_graph_truncated,
        "active_resource_group_vectorization_present": bool(roots_found)
        and all(
            method_reports.get(name, {}).get("active_group_index_lines")
            for name in required_methods
        ),
        "numpy_all_vectorization_present": bool(roots_found)
        and all(
            method_reports.get(name, {}).get("numpy_all_lines")
            for name in required_methods
        ),
        "no_python_loops_in_production_dominance_updates": root_loop_count == 0,
        "no_reachable_python_nested_loop": reachable_nested_count == 0,
        "no_python_all_open_record_nested_loop": reachable_nested_count == 0,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def write_implementation_check_evidence(
    output_directory: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Persist the source/AST preflight alongside profiles and pytest evidence."""

    if not isinstance(report, Mapping):
        raise ValueError("implementation report must be a mapping")
    path = Path(output_directory).resolve() / "profiles" / "implementation_check.json"
    _atomic_write(path, _json_bytes(dict(report)))
    return path


def _require_pre_timing_gates(
    correctness_gate: CorrectnessGate,
    implementation_checks: Mapping[str, bool],
    *,
    require_correctness: bool = True,
) -> dict[str, bool]:
    if not isinstance(correctness_gate, CorrectnessGate):
        raise TypeError("correctness_gate must be a CorrectnessGate")
    if not isinstance(require_correctness, bool):
        raise ValueError("require_correctness must be a bool")
    if require_correctness and not correctness_gate.passed:
        raise ValueError("reference-equivalence correctness gate must pass before timing")
    if not isinstance(implementation_checks, Mapping):
        raise ValueError("implementation_checks must be a mapping")
    normalized: dict[str, bool] = {}
    for name, value in implementation_checks.items():
        key = _nonempty_string(name, name="implementation check name")
        normalized[key] = _strict_bool(value, name=f"implementation check {key}")
    if normalized.get(PRODUCTION_DOMINANCE_IMPLEMENTATION_CHECK) is not True:
        raise ValueError(
            "production dominance implementation check must pass before timing"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class MicrobenchmarkMeasurement:
    """One isolated evaluator measurement for a fixed frontier size."""

    frontier_size: int
    repetitions: int
    reference_total_seconds: float | None
    optimized_synchronization_seconds: float
    optimized_compact_batch_seconds: float
    optimized_scoring_seconds: float
    optimized_selected_row_seconds: float
    feature_index_memory_bytes: int
    unique_resource_groups: int
    process_peak_rss_bytes: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        _positive_int(self.frontier_size, name="frontier_size")
        _positive_int(self.repetitions, name="repetitions")
        _optional_positive_float(
            self.reference_total_seconds, name="reference_total_seconds"
        )
        for field in (
            "optimized_synchronization_seconds",
            "optimized_compact_batch_seconds",
            "optimized_scoring_seconds",
            "optimized_selected_row_seconds",
        ):
            _nonnegative_float(getattr(self, field), name=field)
        _positive_int(self.feature_index_memory_bytes, name="feature_index_memory_bytes")
        groups = _positive_int(self.unique_resource_groups, name="unique_resource_groups")
        if groups > int(self.frontier_size):
            raise ValueError("unique_resource_groups cannot exceed frontier_size")
        peak_rss = _optional_nonnegative_int(
            self.process_peak_rss_bytes, name="process_peak_rss_bytes"
        )
        if peak_rss is not None and peak_rss < int(self.feature_index_memory_bytes):
            raise ValueError("process_peak_rss_bytes cannot be smaller than index memory")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if self.optimized_total_decision_seconds <= 0.0:
            raise ValueError("optimized total decision time must be positive")

    @property
    def optimized_compact_plus_score_seconds(self) -> float:
        return float(self.optimized_compact_batch_seconds + self.optimized_scoring_seconds)

    @property
    def optimized_total_decision_seconds(self) -> float:
        return float(
            self.optimized_synchronization_seconds
            + self.optimized_compact_batch_seconds
            + self.optimized_scoring_seconds
            + self.optimized_selected_row_seconds
        )

    @classmethod
    def from_value(
        cls, value: "MicrobenchmarkMeasurement | Mapping[str, Any]"
    ) -> "MicrobenchmarkMeasurement":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("microbenchmark adapter must return a measurement or mapping")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class EndToEndMeasurement:
    """One hard-target episode measurement at a staged expansion cap."""

    expansion_cap: int
    expansions_completed: int
    runtime_seconds: float
    feature_time_seconds: float
    peak_frontier: int
    peak_unique_resource_groups: int
    peak_feature_index_memory_bytes: int
    terminal_status: str
    reference_runtime_seconds: float | None = None
    reference_feature_time_seconds: float | None = None
    trace_equivalent: bool | None = None
    final_weights_equivalent: bool | None = None
    terminal_status_equivalent: bool | None = None
    deterministic_counters_equivalent: bool | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        cap = _positive_int(self.expansion_cap, name="expansion_cap")
        completed = _nonnegative_int(self.expansions_completed, name="expansions_completed")
        if completed > cap:
            raise ValueError("expansions_completed cannot exceed expansion_cap")
        runtime = _positive_float(self.runtime_seconds, name="runtime_seconds")
        feature_time = _nonnegative_float(
            self.feature_time_seconds, name="feature_time_seconds"
        )
        if feature_time > runtime:
            raise ValueError("feature_time_seconds cannot exceed runtime_seconds")
        peak_frontier = _positive_int(self.peak_frontier, name="peak_frontier")
        groups = _positive_int(
            self.peak_unique_resource_groups, name="peak_unique_resource_groups"
        )
        if groups > peak_frontier:
            raise ValueError("peak unique resource groups cannot exceed peak frontier")
        _positive_int(
            self.peak_feature_index_memory_bytes,
            name="peak_feature_index_memory_bytes",
        )
        _nonempty_string(self.terminal_status, name="terminal_status")
        _optional_positive_float(
            self.reference_runtime_seconds, name="reference_runtime_seconds"
        )
        reference_feature_time = _optional_nonnegative_float(
            self.reference_feature_time_seconds,
            name="reference_feature_time_seconds",
        )
        if (
            reference_feature_time is not None
            and self.reference_runtime_seconds is None
        ):
            raise ValueError(
                "reference feature time requires reference runtime seconds"
            )
        if (
            reference_feature_time is not None
            and reference_feature_time > float(self.reference_runtime_seconds)
        ):
            raise ValueError(
                "reference_feature_time_seconds cannot exceed reference runtime"
            )
        for field in (
            "trace_equivalent",
            "final_weights_equivalent",
            "terminal_status_equivalent",
            "deterministic_counters_equivalent",
        ):
            _optional_bool(getattr(self, field), name=field)
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")

    @property
    def feature_time_share(self) -> float:
        return float(self.feature_time_seconds / self.runtime_seconds)

    @property
    def reference_feature_time_share(self) -> float | None:
        if (
            self.reference_runtime_seconds is None
            or self.reference_feature_time_seconds is None
        ):
            return None
        return float(
            self.reference_feature_time_seconds / self.reference_runtime_seconds
        )

    @property
    def end_to_end_speedup(self) -> float | None:
        if self.reference_runtime_seconds is None:
            return None
        return float(self.reference_runtime_seconds / self.runtime_seconds)

    @property
    def reference_parity_passed(self) -> bool | None:
        values = (
            self.trace_equivalent,
            self.final_weights_equivalent,
            self.terminal_status_equivalent,
            self.deterministic_counters_equivalent,
        )
        if all(value is None for value in values):
            return None
        return all(value is True for value in values)

    @classmethod
    def from_value(
        cls, value: "EndToEndMeasurement | Mapping[str, Any]"
    ) -> "EndToEndMeasurement":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("end-to-end adapter must return a measurement or mapping")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PilotFeasibilityCriteria:
    """Explicit engineering acceptance bounds; never inferred from outcomes."""

    maximum_hard_episode_seconds: float
    maximum_peak_index_memory_bytes: int

    def __post_init__(self) -> None:
        _positive_float(
            self.maximum_hard_episode_seconds,
            name="maximum_hard_episode_seconds",
        )
        _positive_int(
            self.maximum_peak_index_memory_bytes,
            name="maximum_peak_index_memory_bytes",
        )


@runtime_checkable
class FeatureBenchmarkAdapter(Protocol):
    """Integration boundary implemented by the evaluator/search runner."""

    def measure_microbenchmark(
        self,
        frontier_size: int,
        *,
        include_reference: bool,
    ) -> MicrobenchmarkMeasurement | Mapping[str, Any]:
        """Measure one isolated frontier size after any adapter-owned warmup."""

    def measure_end_to_end(
        self,
        expansion_cap: int,
        *,
        include_reference: bool,
    ) -> EndToEndMeasurement | Mapping[str, Any]:
        """Measure the fixed hard target at one staged expansion cap."""


@dataclass(slots=True)
class _HardEpisodeResult:
    runtime_seconds: float
    feature_time_seconds: float
    peak_frontier: int
    peak_unique_resource_groups: int
    feature_index_memory_bytes: int
    terminal_status: str
    expansions_completed: int
    selected_record_ids: tuple[int, ...]
    rewards: tuple[float, ...]
    final_weights: np.ndarray
    deterministic_search_metrics: Mapping[str, Any]
    provider: object
    captured_records: tuple[object, ...] = ()
    captured_generation_counts: Mapping[object, int] | None = None
    capture_expansion: int | None = None


class RepositoryArticleV1FeatureBenchmarkAdapter:
    """Real fixed-workload adapter for the repository's Article V1 runner.

    Imports of the corpus, search environment, providers, and trainer are kept
    inside methods so importing the reporting harness cannot create a runner
    cycle.  The workload is the unique pilot ``train/hard/3q`` target.  Its
    scientific horizon remains 8,192 while engineering runs force a clean
    truncation after the staged cap.
    """

    def __init__(
        self,
        config: str | Path = "pilot",
        *,
        target_id: str = PILOT_HARD_3Q_TARGET_ID,
        microbenchmark_repetitions: int = 3,
        microbenchmark_warmups: int = 1,
        reference_safe_frontier_size: int = DEFAULT_REFERENCE_SAFE_FRONTIER_SIZE,
        frontier_capture_expansion_limit: int = 512,
        profile_caps: Sequence[int] = (32, 64),
        profile_frontier_sizes: Sequence[int] = (1024,),
    ) -> None:
        self.config_source = config
        self.target_id = _nonempty_string(target_id, name="target_id")
        self.microbenchmark_repetitions = _positive_int(
            microbenchmark_repetitions, name="microbenchmark_repetitions"
        )
        self.microbenchmark_warmups = _nonnegative_int(
            microbenchmark_warmups, name="microbenchmark_warmups"
        )
        self.reference_safe_frontier_size = _positive_int(
            reference_safe_frontier_size,
            name="reference_safe_frontier_size",
        )
        self.frontier_capture_expansion_limit = _positive_int(
            frontier_capture_expansion_limit,
            name="frontier_capture_expansion_limit",
        )
        self.profile_caps = _validated_axis(profile_caps, name="profile caps")
        self.profile_frontier_sizes = _validated_axis(
            profile_frontier_sizes, name="profile frontier sizes"
        )
        self._representative_records: tuple[object, ...] | None = None
        self._representative_generation_counts: Mapping[object, int] | None = None
        self._representative_capture_expansion: int | None = None
        self._load_workload()

    def _load_workload(self) -> None:
        from benchmarks.article_native_corpus import build_article_v1_corpus

        corpus = build_article_v1_corpus(self.config_source)
        training_targets = corpus.evaluation_targets(split="train")
        matches = [
            (index, case)
            for index, case in enumerate(training_targets)
            if case.target_id == self.target_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "feature benchmark target must identify exactly one train target"
            )
        index, case = matches[0]
        if case.difficulty != "hard" or case.num_qubits != 3:
            raise ValueError("feature benchmark target must be the hard three-qubit case")
        experiment = dict(corpus.config.experiment)
        training_seeds = experiment.get("training_seeds")
        if not isinstance(training_seeds, Sequence) or not training_seeds:
            raise ValueError("Article V1 config has no training seeds")
        self.corpus = corpus
        self.case = case
        self.transfer_target_index = int(index)
        self.effective_seed = int(training_seeds[0]) + int(index)
        self.experiment = experiment
        self.scientific_horizon = int(case.budget.expansion_budget)
        if self.scientific_horizon != DEFAULT_HARD_EXPANSION_CAP:
            raise ValueError(
                "fixed hard benchmark requires the configured 8,192-expansion horizon"
            )

    def metadata(self) -> dict[str, Any]:
        experiment_fields = (
            "profile_name",
            "feature_schema",
            "reward_schema",
            "target_metric_schema",
            "certification_schema",
            "beta",
            "gamma",
            "learning_rate",
            "epsilon",
            "training_episodes_per_target",
            "training_seeds",
            "random_scheduler_seeds",
            "validation_seeds",
            "statistics_seed",
            "certification_tolerance",
            "canonicalization_enabled",
            "pareto_dominance_enabled",
            "absorb_clifford_angles",
            "canonicalization_mode",
        )
        return {
            "config_profile": self.corpus.config.profile,
            "config_digest": self.corpus.config.digest,
            "resolved_config": self.corpus.config.to_dict(),
            "target_id": self.case.target_id,
            "split": self.case.split,
            "difficulty": self.case.difficulty,
            "num_qubits": self.case.num_qubits,
            "generator_length": self.case.generator_length,
            "transfer_target_index": self.transfer_target_index,
            "effective_seed": self.effective_seed,
            "scientific_horizon": self.scientific_horizon,
            "budget": self.case.budget.metadata(),
            "experiment": {
                name: _json_compatible_copy(self.experiment[name])
                for name in experiment_fields
            },
            "microbenchmark_repetitions": self.microbenchmark_repetitions,
            "microbenchmark_warmups": self.microbenchmark_warmups,
            "reference_safe_frontier_size": self.reference_safe_frontier_size,
            "frontier_capture_expansion_limit": (
                self.frontier_capture_expansion_limit
            ),
            "evidence_class": ENGINEERING_EVIDENCE_CLASS,
            "scientific_scheduler_evidence": False,
            "generator_witness_exposed_to_search": False,
        }

    def _provider(self, evaluator: str):
        from rl.article_features import (
            ArticleTargetContext,
            ArticleV1FeatureProvider,
            ArticleV1ReferenceFeatureProvider,
        )

        context = ArticleTargetContext(self.case.target)
        common = {
            "search_horizon": self.scientific_horizon,
        }
        if evaluator == "optimized":
            return ArticleV1FeatureProvider(context, **common), context
        if evaluator == "reference":
            return (
                ArticleV1ReferenceFeatureProvider(
                    context,
                    reference_safe_frontier_size=max(
                        self.reference_safe_frontier_size,
                        self.scientific_horizon,
                    ),
                    warn_above_safe_size=False,
                    **common,
                ),
                context,
            )
        raise ValueError("evaluator must be 'optimized' or 'reference'")

    @staticmethod
    def _deterministic_search_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in values.items():
            if name.endswith("_time_ns"):
                continue
            if name.startswith("feature_") or name.startswith("target_metric_"):
                continue
            if name == "ranking_time_ns":
                continue
            result[name] = value
        return result

    def _run_hard_episode(
        self,
        evaluator: str,
        stop_after: int,
        *,
        capture_frontier_size: int | None = None,
    ) -> _HardEpisodeResult:
        from certification.article_v1 import ArticleV1CertificationEngine
        from config import Config
        from env.rl_env import CircuitSynthesisEnv
        from rl.policy import LinearQPolicy
        from train import Trainer

        cap = _positive_int(stop_after, name="stop_after")
        if cap > self.scientific_horizon:
            raise ValueError("engineering stop cannot exceed the scientific horizon")
        if capture_frontier_size is not None:
            capture_frontier_size = _positive_int(
                capture_frontier_size, name="capture_frontier_size"
            )
        provider, context = self._provider(evaluator)
        policy = LinearQPolicy(
            feature_provider=provider,
            lr=float(self.experiment["learning_rate"]),
            gamma=1.0,
            seed=self.effective_seed,
        )
        environment = CircuitSynthesisEnv(
            Config(
                num_qubits=self.case.num_qubits,
                budget=self.case.budget.resource_budget(),
                max_steps=self.scientific_horizon,
                max_frontier=64,
                discount=1.0,
                seed=self.effective_seed,
                fairness_interval=0,
                canonicalization_enabled=bool(
                    self.experiment["canonicalization_enabled"]
                ),
                pareto_dominance_enabled=bool(
                    self.experiment["pareto_dominance_enabled"]
                ),
                absorb_clifford_angles=bool(
                    self.experiment["absorb_clifford_angles"]
                ),
                canonicalization_mode=str(
                    self.experiment["canonicalization_mode"]
                ),
                reward_mode="article_v1_expansion_potential",
                article_v1_beta=float(self.experiment["beta"]),
            ),
            ArticleV1CertificationEngine(
                self.case.target,
                tau_cert=float(self.experiment["certification_tolerance"]),
            ),
            feature_provider=provider,
            target_metric=context,
            instrumentation_enabled=True,
            observation_features=False,
        )
        trainer = Trainer(environment, policy=policy)
        epsilon = self.experiment["epsilon"]
        trainer.epsilon = float(epsilon["start"])
        trainer.min_epsilon = float(epsilon["minimum"])
        trainer.epsilon_decay = float(epsilon["decay"])

        selected_ids: list[int] = []
        rewards: list[float] = []
        captured_records: tuple[object, ...] = ()
        captured_counts: Mapping[object, int] | None = None
        capture_expansion: int | None = None
        original_select_record = environment.select_record

        def bounded_select_record(record_id: int):
            nonlocal captured_records, captured_counts, capture_expansion
            observation, reward, terminated, truncated, info = original_select_record(
                record_id
            )
            selected = info.get("selected_record_id", record_id)
            selected_ids.append(int(selected))
            rewards.append(float(reward))
            if capture_frontier_size is not None and not captured_records:
                records = tuple(environment.current_records())
                if len(records) >= capture_frontier_size:
                    captured_records = records
                    captured_counts = dict(environment.generation_counts)
                    capture_expansion = int(environment.steps)
                    if not terminated:
                        truncated = True
            if environment.steps >= cap and not terminated:
                truncated = True
            return observation, reward, terminated, truncated, info

        environment.select_record = bounded_select_record  # type: ignore[method-assign]
        started = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            history = trainer.train(1)
        runtime = float(time.perf_counter() - started)
        episode = history[0]
        metrics = dict(episode["search_metrics"])
        provider_metrics = dict(getattr(provider, "instrumentation", lambda: {})())
        terminal_status = (
            "certified"
            if bool(episode["certified"])
            else "truncated"
            if bool(episode["truncated"])
            else "frontier_exhausted"
        )
        return _HardEpisodeResult(
            runtime_seconds=runtime,
            feature_time_seconds=float(policy.feature_time_ns) / 1e9,
            peak_frontier=int(metrics["frontier_peak"]),
            peak_unique_resource_groups=int(
                provider_metrics.get("resource_group_peak", 0)
            ),
            feature_index_memory_bytes=int(
                provider_metrics.get("feature_index_memory_bytes", 0)
            ),
            terminal_status=terminal_status,
            expansions_completed=int(metrics["expanded"]),
            selected_record_ids=tuple(selected_ids),
            rewards=tuple(rewards),
            final_weights=np.array(policy.theta, dtype=np.float64, copy=True),
            deterministic_search_metrics=self._deterministic_search_metrics(metrics),
            provider=provider,
            captured_records=captured_records,
            captured_generation_counts=captured_counts,
            capture_expansion=capture_expansion,
        )

    def _ensure_representative_frontier(self, minimum_size: int) -> None:
        required = _positive_int(minimum_size, name="minimum_size")
        if (
            self._representative_records is not None
            and len(self._representative_records) >= required
        ):
            return
        result = self._run_hard_episode(
            "optimized",
            self.frontier_capture_expansion_limit,
            capture_frontier_size=required,
        )
        if len(result.captured_records) < required:
            raise RuntimeError(
                "fixed hard-target rollout did not reach the requested frontier "
                f"size {required} within {self.frontier_capture_expansion_limit} "
                f"expansions (peak={result.peak_frontier})"
            )
        if result.captured_generation_counts is None or result.capture_expansion is None:
            raise AssertionError("frontier capture omitted generation-count context")
        self._representative_records = result.captured_records
        self._representative_generation_counts = dict(
            result.captured_generation_counts
        )
        self._representative_capture_expansion = int(result.capture_expansion)

    def prepare_microbenchmarks(self, frontier_sizes: Sequence[int]) -> None:
        """Capture one shared representative frontier for every requested size."""

        sizes = _validated_axis(frontier_sizes, name="frontier sizes")
        self._ensure_representative_frontier(max(sizes))

    @staticmethod
    def _median_seconds(values_ns: Sequence[int]) -> float:
        if not values_ns:
            raise ValueError("timing sequence must not be empty")
        return float(statistics.median(values_ns)) / 1e9

    def measure_microbenchmark(
        self,
        frontier_size: int,
        *,
        include_reference: bool,
    ) -> MicrobenchmarkMeasurement:
        size = _positive_int(frontier_size, name="frontier_size")
        _strict_bool(include_reference, name="include_reference")
        if include_reference and size > self.reference_safe_frontier_size:
            raise ValueError(
                "reference measurement requested above the configured safe size"
            )
        self._ensure_representative_frontier(size)
        assert self._representative_records is not None
        assert self._representative_generation_counts is not None
        assert self._representative_capture_expansion is not None
        records = tuple(self._representative_records[:size])
        counts = self._representative_generation_counts
        completed = self._representative_capture_expansion

        reference_seconds: float | None = None
        if include_reference:
            reference, _reference_context = self._provider("reference")
            reference.bind_generation_counts(counts)
            reference_records = tuple(
                getattr(record, "node", record) for record in records
            )
            for _ in range(self.microbenchmark_warmups):
                reference.build_snapshot(
                    reference_records,
                    expansions_completed=completed,
                    expansion_budget=self.scientific_horizon,
                )
            reference_times: list[int] = []
            for _ in range(self.microbenchmark_repetitions):
                started = time.perf_counter_ns()
                reference.build_snapshot(
                    reference_records,
                    expansions_completed=completed,
                    expansion_budget=self.scientific_horizon,
                )
                reference_times.append(time.perf_counter_ns() - started)
            reference_seconds = self._median_seconds(reference_times)

        optimized, _optimized_context = self._provider("optimized")
        optimized.bind_generation_counts(counts)
        theta = np.linspace(-0.5, 0.5, int(optimized.dimension), dtype=np.float64)
        # Cold construction is intentionally outside the steady-state decision
        # timings. It populates immutable static rows exactly once.
        optimized.build_compact_batch(
            records,
            theta=theta,
            expansions_completed=completed,
            expansion_budget=self.scientific_horizon,
        )
        for _ in range(self.microbenchmark_warmups):
            optimized.synchronize_frontier(records)
            batch = optimized.feature_index.build_decision_batch(
                theta=theta,
                expansions_completed=completed,
                expansion_budget=self.scientific_horizon,
            )
            batch.scores(theta)
            batch.features_for_record(int(batch.record_ids[0]))

        synchronize_times: list[int] = []
        compact_times: list[int] = []
        scoring_times: list[int] = []
        selected_row_times: list[int] = []
        for _ in range(self.microbenchmark_repetitions):
            started = time.perf_counter_ns()
            optimized.synchronize_frontier(records)
            synchronize_times.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            batch = optimized.feature_index.build_decision_batch(
                theta=theta,
                expansions_completed=completed,
                expansion_budget=self.scientific_horizon,
            )
            compact_times.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            batch.scores(theta)
            scoring_times.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            batch.features_for_record(int(batch.record_ids[0]))
            selected_row_times.append(time.perf_counter_ns() - started)

        metrics = dict(optimized.instrumentation())
        return MicrobenchmarkMeasurement(
            frontier_size=size,
            repetitions=self.microbenchmark_repetitions,
            reference_total_seconds=reference_seconds,
            optimized_synchronization_seconds=self._median_seconds(
                synchronize_times
            ),
            optimized_compact_batch_seconds=self._median_seconds(compact_times),
            optimized_scoring_seconds=self._median_seconds(scoring_times),
            optimized_selected_row_seconds=self._median_seconds(selected_row_times),
            feature_index_memory_bytes=int(metrics["feature_index_memory_bytes"]),
            unique_resource_groups=int(metrics["unique_resource_group_count"]),
            process_peak_rss_bytes=None,
            notes=(
                "median steady-state decision components after static-cache warmup; "
                f"hard-3q target={self.case.target_id}; effective_seed={self.effective_seed}; "
                f"frontier captured at expansion={completed}; reference_repetitions="
                f"{self.microbenchmark_repetitions if include_reference else 0}"
            ),
        )

    def measure_end_to_end(
        self,
        expansion_cap: int,
        *,
        include_reference: bool,
    ) -> EndToEndMeasurement:
        cap = _positive_int(expansion_cap, name="expansion_cap")
        _strict_bool(include_reference, name="include_reference")
        optimized = self._run_hard_episode("optimized", cap)
        reference: _HardEpisodeResult | None = None
        if include_reference:
            reference = self._run_hard_episode("reference", cap)
        trace_equivalent = None
        final_weights_equivalent = None
        terminal_status_equivalent = None
        deterministic_counters_equivalent = None
        reference_runtime = None
        reference_feature_time = None
        if reference is not None:
            reference_runtime = reference.runtime_seconds
            reference_feature_time = reference.feature_time_seconds
            trace_equivalent = (
                optimized.selected_record_ids == reference.selected_record_ids
                and optimized.rewards == reference.rewards
            )
            final_weights_equivalent = bool(
                np.array_equal(optimized.final_weights, reference.final_weights)
            )
            terminal_status_equivalent = (
                optimized.terminal_status == reference.terminal_status
                and optimized.expansions_completed == reference.expansions_completed
            )
            deterministic_counters_equivalent = (
                dict(optimized.deterministic_search_metrics)
                == dict(reference.deterministic_search_metrics)
            )
        return EndToEndMeasurement(
            expansion_cap=cap,
            expansions_completed=optimized.expansions_completed,
            runtime_seconds=optimized.runtime_seconds,
            feature_time_seconds=optimized.feature_time_seconds,
            peak_frontier=optimized.peak_frontier,
            peak_unique_resource_groups=optimized.peak_unique_resource_groups,
            peak_feature_index_memory_bytes=optimized.feature_index_memory_bytes,
            terminal_status=optimized.terminal_status,
            reference_runtime_seconds=reference_runtime,
            reference_feature_time_seconds=reference_feature_time,
            trace_equivalent=trace_equivalent,
            final_weights_equivalent=final_weights_equivalent,
            terminal_status_equivalent=terminal_status_equivalent,
            deterministic_counters_equivalent=deterministic_counters_equivalent,
            notes=(
                f"fixed pilot train/hard/3q target; seed={self.effective_seed}; "
                f"scientific_horizon={self.scientific_horizon}; engineering_stop={cap}; "
                "forced stop changes neither horizon-dependent features nor config"
            ),
        )

    def write_profiles(self, directory: Path) -> None:
        """Write standard-library cProfile data outside timed CSV samples."""

        import cProfile
        import pstats

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for cap in self.profile_caps:
            profiler = cProfile.Profile()
            profiler.runcall(self._run_hard_episode, "optimized", cap)
            stem = f"hard-3q-cap-{cap}-optimized"
            temporary = destination / f".{stem}.prof.tmp"
            profiler.dump_stats(str(temporary))
            os.replace(temporary, destination / f"{stem}.prof")
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(60)
            _atomic_write(
                destination / f"{stem}.txt", stream.getvalue().encode("utf-8")
            )
        for size in self.profile_frontier_sizes:
            profiler = cProfile.Profile()
            profiler.runcall(
                self.measure_microbenchmark,
                size,
                # This artifact is explicitly named "optimized". Reference
                # all-pairs work is measured separately in the timed CSV and
                # must not dominate or contaminate this diagnostic profile.
                include_reference=False,
            )
            stem = f"frontier-F{size}-optimized"
            temporary = destination / f".{stem}.prof.tmp"
            profiler.dump_stats(str(temporary))
            os.replace(temporary, destination / f"{stem}.prof")
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(60)
            _atomic_write(
                destination / f"{stem}.txt", stream.getvalue().encode("utf-8")
            )


def create_repository_feature_benchmark_adapter(
    config: str | Path = "pilot",
    **kwargs: Any,
) -> RepositoryArticleV1FeatureBenchmarkAdapter:
    """Factory kept small enough for lazy use by ``article_benchmark.py``."""

    return RepositoryArticleV1FeatureBenchmarkAdapter(config, **kwargs)


def run_repository_feature_benchmark(
    output_directory: str | Path,
    *,
    correctness_gate: CorrectnessGate,
    implementation_checks: Mapping[str, bool],
    config: str | Path = "pilot",
    frontier_sizes: Sequence[int] = DEFAULT_FRONTIER_SIZES,
    staged_caps: Sequence[int] = DEFAULT_STAGED_EXPANSION_CAPS,
    baseline: Mapping[str, Any] | None = None,
    feasibility_criteria: PilotFeasibilityCriteria | None = None,
    pilot_relaunch_checks: Mapping[str, bool] | None = None,
    benchmark_provenance: Mapping[str, Any] | None = None,
    hard_training_episode_count: int | None = None,
    write_profiles: bool = True,
    adapter_kwargs: Mapping[str, Any] | None = None,
) -> FeatureBenchmarkArtifacts:
    """Run the real fixed workload; suitable for a thin CLI dispatch.

    The caller remains responsible for constructing byte-bound correctness and
    source-provenance evidence.  This function never changes the corpus config,
    expansion horizon, or held-out experiment state.
    """

    if not isinstance(write_profiles, bool):
        raise ValueError("write_profiles must be a bool")
    if adapter_kwargs is not None and not isinstance(adapter_kwargs, Mapping):
        raise ValueError("adapter_kwargs must be a mapping or None")
    _require_pre_timing_gates(correctness_gate, implementation_checks)
    adapter = create_repository_feature_benchmark_adapter(
        config, **dict(adapter_kwargs or {})
    )
    baseline_payload = dict(baseline or baseline_f653193())
    baseline_payload["benchmark_workload"] = adapter.metadata()
    return benchmark_feature_evaluator(
        adapter,
        output_directory,
        correctness_gate=correctness_gate,
        implementation_checks=implementation_checks,
        frontier_sizes=frontier_sizes,
        staged_caps=staged_caps,
        reference_safe_frontier_size=adapter.reference_safe_frontier_size,
        reference_trace_caps=DEFAULT_REFERENCE_TRACE_CAPS,
        baseline=baseline_payload,
        feasibility_criteria=feasibility_criteria,
        hard_expansion_cap=adapter.scientific_horizon,
        hard_training_episode_count=hard_training_episode_count,
        pilot_relaunch_checks=pilot_relaunch_checks,
        benchmark_provenance=benchmark_provenance,
        require_correctness_before_timing=True,
        profile_writer=adapter.write_profiles if write_profiles else None,
    )


def capture_benchmark_environment() -> dict[str, Any]:
    """Capture machine/runtime metadata without assigning scientific meaning."""

    configuration: Mapping[str, Any] = {}
    try:
        candidate = np.__config__.show(mode="dicts")
        if isinstance(candidate, Mapping):
            configuration = candidate
    except (TypeError, AttributeError):  # pragma: no cover - NumPy compatibility
        configuration = {}
    build_dependencies = configuration.get("Build Dependencies", {})
    blas = build_dependencies.get("blas", {}) if isinstance(build_dependencies, Mapping) else {}
    if not isinstance(blas, Mapping):
        blas = {}
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "configuration": blas.get("openblas configuration"),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "evidence_class": ENGINEERING_EVIDENCE_CLASS,
        "scientific_scheduler_evidence": False,
    }


def baseline_f653193(
    *, environment: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the immutable baseline record supplied for the optimization."""

    historical_environment = (
        {
            "captured": False,
            "status": "unknown-not-captured-with-historical-baseline",
            "note": (
                "The f653193 timing environment was not preserved; current-host "
                "metadata must not be retroactively attached to historical timings."
            ),
        }
        if environment is None
        else dict(environment)
    )

    return {
        "schema_version": ARTICLE_V1_FEATURE_BASELINE_SCHEMA,
        "evidence_class": ENGINEERING_EVIDENCE_CLASS,
        "scientific_scheduler_evidence": False,
        "source": {
            "branch": "frontier-rl",
            "commit": "f653193ec1fd15b17b948a476a0e89a343cbf062",
            "worktree_clean": True,
            "pilot_process_running": False,
            "pilot_checkpoint_exists": False,
            "pilot_scientific_raw_results_exist": False,
        },
        "preflight": {
            "compileall_passed": True,
            "article_v1_tests_passed": 240,
            "full_repository_tests_passed": 408,
            "calibration_passed": True,
            "mini_ci_passed_twice_with_byte_stable_resume": True,
        },
        "hard_three_qubit_episode": [
            {
                "expansion_cap": 32,
                "runtime_seconds": 8.589,
                "reported_runtime_range_seconds": [8.589, 11.8],
                "peak_frontier": 543,
                "feature_time_share": 0.847,
            },
            {
                "expansion_cap": 64,
                "runtime_seconds": 61.837,
                "reported_runtime_range_seconds": [59.9, 61.837],
                "peak_frontier": 1039,
                "feature_time_share": 0.945,
            },
        ],
        "isolated_reference_feature_batches": [
            {"frontier_size": 534, "seconds": 0.790},
            {"frontier_size": 1021, "seconds": 2.330},
        ],
        "profiling": {
            "confirmed_primary_component": "frontier-wide feature evaluation",
            "cap_32_feature_time_share": 0.847,
            "cap_64_feature_time_share": 0.945,
            "raw_cprofile_top_function_table_available": False,
            "note": (
                "The supplied baseline preserved component shares but not a raw "
                "cProfile table; new profile files belong under profiles/."
            ),
        },
        "environment": historical_environment,
    }


def _fit_power_law(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    """Fit y = coefficient * x**exponent in log space."""

    if len(x_values) != len(y_values) or len(x_values) < 2:
        return {"available": False, "reason": "at least two paired observations required"}
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x <= 0) or np.any(y <= 0):
        return {"available": False, "reason": "observations must be finite and positive"}
    log_x = np.log(x)
    log_y = np.log(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = intercept + exponent * log_x
    residual = float(np.sum((log_y - predicted) ** 2, dtype=np.float64))
    centered = float(np.sum((log_y - np.mean(log_y)) ** 2, dtype=np.float64))
    r_squared = 1.0 if centered == 0.0 and residual == 0.0 else (
        0.0 if centered == 0.0 else 1.0 - residual / centered
    )
    return {
        "available": True,
        "model": "power-law-log-least-squares",
        "coefficient": float(math.exp(float(intercept))),
        "exponent": float(exponent),
        "r_squared_log_space": float(r_squared),
        "observation_count": int(len(x)),
        "minimum_x": float(np.min(x)),
        "maximum_x": float(np.max(x)),
    }


def _predict(model: Mapping[str, Any], x_value: float) -> float | None:
    if model.get("available") is not True:
        return None
    return float(model["coefficient"]) * float(x_value) ** float(model["exponent"])


def _baseline_reference_near(
    baseline: Mapping[str, Any], target_size: int
) -> tuple[float | None, str]:
    values = baseline.get("isolated_reference_feature_batches", ())
    candidates: list[tuple[int, float]] = []
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for record in values:
            if not isinstance(record, Mapping):
                continue
            try:
                size = _positive_int(record.get("frontier_size"), name="baseline frontier size")
                seconds = _positive_float(record.get("seconds"), name="baseline seconds")
            except ValueError:
                continue
            candidates.append((size, seconds))
    if not candidates:
        return None, "unavailable"
    size, seconds = min(candidates, key=lambda item: (abs(item[0] - target_size), item[0]))
    if abs(size - target_size) > max(8, int(math.ceil(target_size * 0.05))):
        return None, "unavailable"
    return seconds, f"baseline-nearest-F{size}"


def _micro_rows(
    measurements: Sequence[MicrobenchmarkMeasurement],
    *,
    correctness_gate: CorrectnessGate,
    baseline: Mapping[str, Any],
    reference_safe_frontier_size: int,
) -> list[dict[str, Any]]:
    del baseline  # historical timings are context only, never a current-host gate
    rows: list[dict[str, Any]] = []
    for item in measurements:
        reference_expected = item.frontier_size <= reference_safe_frontier_size
        effective_reference = item.reference_total_seconds
        reference_basis = "measured-current-reference" if effective_reference is not None else ""
        speedup = (
            None
            if effective_reference is None
            else effective_reference / item.optimized_total_decision_seconds
        )
        rows.append(
            {
                "frontier_size": item.frontier_size,
                "repetitions": item.repetitions,
                "reference_expected": reference_expected,
                "reference_measured": item.reference_total_seconds is not None,
                "reference_total_seconds": item.reference_total_seconds,
                "reference_basis": reference_basis,
                "optimized_synchronization_seconds": item.optimized_synchronization_seconds,
                "optimized_compact_batch_seconds": item.optimized_compact_batch_seconds,
                "optimized_scoring_seconds": item.optimized_scoring_seconds,
                "optimized_selected_row_seconds": item.optimized_selected_row_seconds,
                "optimized_compact_plus_score_seconds": item.optimized_compact_plus_score_seconds,
                "optimized_total_decision_seconds": item.optimized_total_decision_seconds,
                "effective_reference_seconds": effective_reference,
                "speedup": speedup,
                "feature_index_memory_bytes": item.feature_index_memory_bytes,
                "process_peak_rss_bytes": item.process_peak_rss_bytes,
                "unique_resource_groups": item.unique_resource_groups,
                "correctness_gate_passed": correctness_gate.passed,
                "reference_safe_frontier_size": reference_safe_frontier_size,
                "notes": item.notes,
            }
        )
    return rows


def _end_to_end_rows(
    measurements: Sequence[EndToEndMeasurement],
    *,
    reference_trace_caps: Sequence[int],
) -> list[dict[str, Any]]:
    trace_caps = set(reference_trace_caps)
    return [
        {
            "expansion_cap": item.expansion_cap,
            "expansions_completed": item.expansions_completed,
            "runtime_seconds": item.runtime_seconds,
            "feature_time_seconds": item.feature_time_seconds,
            "feature_time_share": item.feature_time_share,
            "peak_frontier": item.peak_frontier,
            "peak_unique_resource_groups": item.peak_unique_resource_groups,
            "peak_feature_index_memory_bytes": item.peak_feature_index_memory_bytes,
            "terminal_status": item.terminal_status,
            "reference_expected": item.expansion_cap in trace_caps,
            "reference_runtime_seconds": item.reference_runtime_seconds,
            "reference_feature_time_seconds": item.reference_feature_time_seconds,
            "reference_feature_time_share": item.reference_feature_time_share,
            "end_to_end_speedup": item.end_to_end_speedup,
            "trace_equivalent": item.trace_equivalent,
            "final_weights_equivalent": item.final_weights_equivalent,
            "terminal_status_equivalent": item.terminal_status_equivalent,
            "deterministic_counters_equivalent": item.deterministic_counters_equivalent,
            "reference_parity_passed": item.reference_parity_passed,
            "notes": item.notes,
        }
        for item in measurements
    ]


def qualify_feature_benchmark(
    micro_rows: Sequence[Mapping[str, Any]],
    end_to_end_rows: Sequence[Mapping[str, Any]],
    *,
    correctness_gate: CorrectnessGate,
    implementation_checks: Mapping[str, bool],
    expected_frontier_sizes: Sequence[int] = DEFAULT_FRONTIER_SIZES,
    expected_staged_caps: Sequence[int] = DEFAULT_STAGED_EXPANSION_CAPS,
    reference_safe_frontier_size: int = DEFAULT_REFERENCE_SAFE_FRONTIER_SIZE,
    reference_trace_caps: Sequence[int] = DEFAULT_REFERENCE_TRACE_CAPS,
) -> dict[str, Any]:
    """Apply the explicit correctness, scaling, memory, and trace gates."""

    expected_sizes = _validated_axis(expected_frontier_sizes, name="frontier sizes")
    expected_caps = _validated_axis(expected_staged_caps, name="staged caps")
    safe_size = _positive_int(
        reference_safe_frontier_size, name="reference_safe_frontier_size"
    )
    trace_caps = _validated_axis(reference_trace_caps, name="reference trace caps")
    checks: dict[str, bool] = {}
    checks["correctness_gate"] = correctness_gate.passed
    checks["microbenchmark_size_coverage"] = tuple(
        sorted(int(row["frontier_size"]) for row in micro_rows)
    ) == expected_sizes
    checks["staged_cap_coverage"] = tuple(
        sorted(int(row["expansion_cap"]) for row in end_to_end_rows)
    ) == expected_caps

    by_size = {int(row["frontier_size"]): row for row in micro_rows}
    reference_coverage = all(
        size not in by_size
        or size > safe_size
        or by_size[size].get("reference_total_seconds") is not None
        for size in expected_sizes
    )
    checks["reference_measured_through_safe_size"] = reference_coverage

    row_1024 = by_size.get(1024)
    speedup_1024 = None if row_1024 is None else row_1024.get("speedup")
    checks["current_host_reference_measured_at_1024"] = bool(
        row_1024 is not None
        and row_1024.get("reference_measured") is True
        and row_1024.get("reference_basis") == "measured-current-reference"
    )
    checks["speedup_at_approximately_1024_at_least_10x"] = (
        isinstance(speedup_1024, (int, float))
        and not isinstance(speedup_1024, bool)
        and math.isfinite(float(speedup_1024))
        and float(speedup_1024) >= MINIMUM_SPEEDUP_AT_1024
    )

    row_512 = by_size.get(512)
    compact_score_ratio = None
    if row_512 is not None and row_1024 is not None:
        denominator = float(row_512["optimized_compact_plus_score_seconds"])
        if denominator > 0.0:
            compact_score_ratio = float(
                row_1024["optimized_compact_plus_score_seconds"]
            ) / denominator
    checks["compact_score_512_to_1024_ratio_at_most_2_5x"] = (
        compact_score_ratio is not None
        and math.isfinite(compact_score_ratio)
        and compact_score_ratio <= MAXIMUM_512_TO_1024_COMPACT_SCORE_RATIO
    )

    if not isinstance(implementation_checks, Mapping):
        raise ValueError("implementation_checks must be a mapping")
    normalized_implementation_checks: dict[str, bool] = {}
    for name, value in implementation_checks.items():
        key = _nonempty_string(name, name="implementation check name")
        normalized_implementation_checks[key] = _strict_bool(
            value, name=f"implementation check {key}"
        )
    checks["no_python_nested_frontier_record_loop"] = (
        normalized_implementation_checks.get("no_python_nested_frontier_record_loop")
        is True
    )

    memory_x = [
        float(int(row["frontier_size"]) + int(row["unique_resource_groups"]))
        for row in micro_rows
    ]
    memory_y = [float(row["feature_index_memory_bytes"]) for row in micro_rows]
    memory_model = _fit_power_law(memory_x, memory_y)
    memory_exponent = (
        float(memory_model["exponent"])
        if memory_model.get("available") is True
        else None
    )
    checks["feature_index_memory_approximately_linear"] = (
        memory_exponent is not None
        and math.isfinite(memory_exponent)
        and memory_exponent <= MAXIMUM_MEMORY_SCALING_EXPONENT
    )

    by_cap = {int(row["expansion_cap"]): row for row in end_to_end_rows}
    parity_by_cap = {
        cap: bool(by_cap.get(cap, {}).get("reference_parity_passed") is True)
        for cap in trace_caps
    }
    checks["reference_trace_parity_at_32_and_64"] = all(parity_by_cap.values())
    end_to_end_speedup_by_cap: dict[int, float | None] = {}
    optimized_feature_share_by_cap: dict[int, float | None] = {}
    reference_feature_share_by_cap: dict[int, float | None] = {}
    for cap in trace_caps:
        row = by_cap.get(cap, {})
        raw_speedup = row.get("end_to_end_speedup")
        end_to_end_speedup_by_cap[cap] = (
            float(raw_speedup)
            if isinstance(raw_speedup, (int, float))
            and not isinstance(raw_speedup, bool)
            and math.isfinite(float(raw_speedup))
            else None
        )
        optimized_feature_share_by_cap[cap] = row.get("feature_time_share")
        reference_feature_share_by_cap[cap] = row.get(
            "reference_feature_time_share"
        )
    checks["end_to_end_speedup_at_32_and_64_at_least_2x"] = all(
        speedup is not None
        and speedup >= MINIMUM_END_TO_END_SPEEDUP_AT_REFERENCE_CAP
        for speedup in end_to_end_speedup_by_cap.values()
    )
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "correctness": correctness_gate.as_dict(),
        "implementation_checks": normalized_implementation_checks,
        "thresholds": {
            "minimum_speedup_at_frontier_1024": MINIMUM_SPEEDUP_AT_1024,
            "maximum_compact_score_ratio_512_to_1024": (
                MAXIMUM_512_TO_1024_COMPACT_SCORE_RATIO
            ),
            "maximum_memory_scaling_exponent": MAXIMUM_MEMORY_SCALING_EXPONENT,
            "minimum_end_to_end_speedup_at_reference_caps": (
                MINIMUM_END_TO_END_SPEEDUP_AT_REFERENCE_CAP
            ),
            "reference_safe_frontier_size": safe_size,
        },
        "observed": {
            "speedup_at_frontier_1024": speedup_1024,
            "compact_score_ratio_512_to_1024": compact_score_ratio,
            "memory_scaling_exponent_vs_frontier_plus_groups": memory_exponent,
            "reference_trace_parity_by_cap": parity_by_cap,
            "end_to_end_speedup_by_cap": end_to_end_speedup_by_cap,
            "optimized_feature_time_share_by_cap": optimized_feature_share_by_cap,
            "reference_feature_time_share_by_cap": reference_feature_share_by_cap,
        },
        "policy": (
            "A failed or incomplete correctness gate invalidates performance "
            "qualification even when timings are faster."
        ),
    }


def project_pilot_cost_from_scaling(
    micro_rows: Sequence[Mapping[str, Any]],
    end_to_end_rows: Sequence[Mapping[str, Any]],
    *,
    qualification: Mapping[str, Any],
    hard_expansion_cap: int = DEFAULT_HARD_EXPANSION_CAP,
    feasibility_criteria: PilotFeasibilityCriteria | None = None,
    hard_training_episode_count: int | None = None,
    pilot_relaunch_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Fit diagnostic scaling models and project one configured hard episode."""

    hard_cap = _positive_int(hard_expansion_cap, name="hard_expansion_cap")
    if hard_training_episode_count is not None:
        hard_training_episode_count = _positive_int(
            hard_training_episode_count, name="hard_training_episode_count"
        )
    feature_model = _fit_power_law(
        [float(row["frontier_size"]) for row in micro_rows],
        [float(row["optimized_total_decision_seconds"]) for row in micro_rows],
    )
    memory_model = _fit_power_law(
        [
            float(int(row["frontier_size"]) + int(row["unique_resource_groups"]))
            for row in micro_rows
        ],
        [float(row["feature_index_memory_bytes"]) for row in micro_rows],
    )
    frontier_model = _fit_power_law(
        [float(row["expansion_cap"]) for row in end_to_end_rows],
        [float(row["peak_frontier"]) for row in end_to_end_rows],
    )
    runtime_model = _fit_power_law(
        [float(row["expansion_cap"]) for row in end_to_end_rows],
        [float(row["runtime_seconds"]) for row in end_to_end_rows],
    )

    projected_frontier = _predict(frontier_model, hard_cap)
    projected_runtime = _predict(runtime_model, hard_cap)
    projected_feature_decision = (
        None if projected_frontier is None else _predict(feature_model, projected_frontier)
    )
    projected_index_memory = (
        None
        if projected_frontier is None
        else _predict(memory_model, 2.0 * projected_frontier)
    )
    # The memory model's independent variable is F + G.  G <= F, so 2F is a
    # conservative structural input, not a confidence bound.
    criteria_payload = None
    within_criteria = None
    if feasibility_criteria is not None:
        criteria_payload = {
            "maximum_hard_episode_seconds": (
                feasibility_criteria.maximum_hard_episode_seconds
            ),
            "maximum_peak_index_memory_bytes": (
                feasibility_criteria.maximum_peak_index_memory_bytes
            ),
        }
        within_criteria = (
            projected_runtime is not None
            and projected_index_memory is not None
            and projected_runtime <= feasibility_criteria.maximum_hard_episode_seconds
            and projected_index_memory
            <= feasibility_criteria.maximum_peak_index_memory_bytes
        )

    normalized_relaunch_checks: dict[str, bool] = {}
    if pilot_relaunch_checks is not None:
        if not isinstance(pilot_relaunch_checks, Mapping):
            raise ValueError("pilot_relaunch_checks must be a mapping or None")
        for name, value in pilot_relaunch_checks.items():
            key = _nonempty_string(name, name="pilot relaunch check name")
            normalized_relaunch_checks[key] = _strict_bool(
                value, name=f"pilot relaunch check {key}"
            )
    missing_relaunch_checks = [
        name
        for name in REQUIRED_PILOT_RELAUNCH_CHECKS
        if name not in normalized_relaunch_checks
    ]
    failed_relaunch_checks = [
        name
        for name in REQUIRED_PILOT_RELAUNCH_CHECKS
        if normalized_relaunch_checks.get(name) is not True
    ]
    relaunch_checks_passed = not failed_relaunch_checks
    qualification_passed = qualification.get("passed") is True
    if not qualification_passed:
        decision = "insufficient performance evidence to relaunch"
    elif feasibility_criteria is None or within_criteria is None:
        decision = "insufficient performance evidence to relaunch"
    elif not within_criteria:
        decision = "configured pilot requires preregistered cap/stratum amendment"
    elif relaunch_checks_passed:
        decision = "configured pilot is feasible unchanged"
    else:
        decision = "insufficient performance evidence to relaunch"

    max_measured_cap = max(
        (int(row["expansion_cap"]) for row in end_to_end_rows), default=0
    )
    campaign_runtime = (
        None
        if projected_runtime is None or hard_training_episode_count is None
        else projected_runtime * hard_training_episode_count
    )
    return {
        "schema_version": ARTICLE_V1_FEATURE_PROJECTION_SCHEMA,
        "executes_search": False,
        "evidence_class": ENGINEERING_EVIDENCE_CLASS,
        "scientific_scheduler_evidence": False,
        "qualification_passed": qualification_passed,
        "pilot_decision": decision,
        "hard_expansion_cap": hard_cap,
        "maximum_measured_expansion_cap": max_measured_cap,
        "extrapolation_factor_beyond_largest_cap": (
            None if max_measured_cap <= 0 else hard_cap / max_measured_cap
        ),
        "models": {
            "feature_decision_seconds_vs_frontier": feature_model,
            "peak_index_memory_bytes_vs_frontier_plus_groups": memory_model,
            "peak_frontier_vs_expansion_cap": frontier_model,
            "episode_runtime_seconds_vs_expansion_cap": runtime_model,
        },
        "projected_hard_episode": {
            "runtime_seconds": projected_runtime,
            "peak_frontier": projected_frontier,
            "feature_decision_seconds_at_projected_peak": projected_feature_decision,
            "peak_feature_index_memory_bytes_conservative_structural_input": (
                projected_index_memory
            ),
        },
        "configured_hard_training_episodes": hard_training_episode_count,
        "projected_hard_training_runtime_seconds": campaign_runtime,
        "feasibility_criteria": criteria_payload,
        "within_feasibility_criteria": within_criteria,
        "pilot_relaunch_gate": {
            "passed": relaunch_checks_passed,
            "required_checks": list(REQUIRED_PILOT_RELAUNCH_CHECKS),
            "checks": {
                name: normalized_relaunch_checks[name]
                for name in sorted(normalized_relaunch_checks)
            },
            "missing_checks": missing_relaunch_checks,
            "failed_checks": failed_relaunch_checks,
        },
        "interpretation": [
            "Models are engineering extrapolations fitted in log space, not confidence intervals.",
            "The runtime projection applies to the fixed staged hard-target workload only.",
            "Peak-memory projection evaluates the F+G model at conservative G=F.",
            "Early certification or frontier exhaustion can reduce actual runtime.",
            "Exact synthesis remains exponential even after evaluator optimization.",
            "No held-out scheduler outcome is read or inferred by this projection.",
        ],
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in columns})
    return handle.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _format_cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _scaling_report(
    *,
    baseline: Mapping[str, Any],
    correctness_gate: CorrectnessGate,
    micro_rows: Sequence[Mapping[str, Any]],
    end_rows: Sequence[Mapping[str, Any]],
    qualification: Mapping[str, Any],
    projection: Mapping[str, Any],
    profile_inventory: Sequence[str],
) -> str:
    lines = [
        "# Article V1 feature-evaluator scaling report",
        "",
        "> Engineering diagnostic only. This report is not scientific scheduler evidence and must not enter the Article V1 raw run ledger.",
        "",
        "## Qualification outcome",
        "",
        f"Overall performance qualification: **{'PASS' if qualification['passed'] else 'FAIL'}**.",
        "",
        f"Pilot decision: **{projection['pilot_decision']}**.",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for name, value in qualification["checks"].items():
        lines.append(f"| `{name}` | {_format_cell(value)} |")
    lines.extend(
        [
            "",
            "The correctness gate dominates all timing results: faster measurements do not qualify when any required parity check is absent or false.",
            "",
            "## Correctness gate",
            "",
            f"Command: `{correctness_gate.command}`",
            "",
            "| Required check | Result |",
            "|---|---:|",
        ]
    )
    for name in REQUIRED_CORRECTNESS_CHECKS:
        lines.append(f"| `{name}` | {_format_cell(correctness_gate.checks.get(name, False))} |")
    lines.extend(
        [
            "",
            "## Isolated feature measurements",
            "",
            "| F | Reference (s) | Sync (s) | Compact batch (s) | Score (s) | Selected row (s) | Total optimized (s) | Speedup | Groups | Index bytes |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in micro_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_cell(row[name])
                for name in (
                    "frontier_size",
                    "effective_reference_seconds",
                    "optimized_synchronization_seconds",
                    "optimized_compact_batch_seconds",
                    "optimized_scoring_seconds",
                    "optimized_selected_row_seconds",
                    "optimized_total_decision_seconds",
                    "speedup",
                    "unique_resource_groups",
                    "feature_index_memory_bytes",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Staged hard-target measurements",
            "",
            "| Cap | Expansions | Optimized runtime (s) | Reference runtime (s) | End-to-end speedup | Optimized feature share | Reference feature share | Peak F | Peak groups | Trace parity |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in end_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_cell(row[name])
                for name in (
                    "expansion_cap",
                    "expansions_completed",
                    "runtime_seconds",
                    "reference_runtime_seconds",
                    "end_to_end_speedup",
                    "feature_time_share",
                    "reference_feature_time_share",
                    "peak_frontier",
                    "peak_unique_resource_groups",
                    "reference_parity_passed",
                )
            )
            + " |"
        )
    source = baseline.get("source", {})
    lines.extend(
        [
            "",
            "## Baseline and projection",
            "",
            f"Baseline commit: `{source.get('commit', 'unknown')}`.",
            "",
            f"Configured hard-cap projection: {_format_cell(projection['projected_hard_episode']['runtime_seconds'])} seconds for cap {projection['hard_expansion_cap']}.",
            "",
            "This is a fitted extrapolation, not a confidence interval. It does not make exact synthesis polynomial and does not justify a pilot relaunch unless every gate and explicit feasibility criterion passes.",
            "",
            "## Profile artifacts",
            "",
        ]
    )
    if profile_inventory:
        lines.extend(f"- `{path}`" for path in profile_inventory)
    else:
        lines.append("No profile files were supplied by the integration adapter.")
    lines.extend(
        [
            "",
            "## Research boundary",
            "",
            "The true frontier remains unbounded, record selection and exhaustive expansion are unchanged, and the 10-D/31-D equations remain exact. These artifacts characterize evaluator implementation cost only.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FeatureBenchmarkArtifacts:
    output_directory: Path
    baseline_json: Path
    microbenchmarks_csv: Path
    end_to_end_scaling_csv: Path
    profiles_directory: Path
    scaling_report_md: Path
    projected_pilot_cost_json: Path
    qualification: Mapping[str, Any]
    projection: Mapping[str, Any]


def write_feature_benchmark_artifacts(
    output_directory: str | Path,
    *,
    correctness_gate: CorrectnessGate,
    microbenchmarks: Sequence[MicrobenchmarkMeasurement | Mapping[str, Any]],
    end_to_end: Sequence[EndToEndMeasurement | Mapping[str, Any]],
    implementation_checks: Mapping[str, bool],
    baseline: Mapping[str, Any] | None = None,
    expected_frontier_sizes: Sequence[int] = DEFAULT_FRONTIER_SIZES,
    expected_staged_caps: Sequence[int] = DEFAULT_STAGED_EXPANSION_CAPS,
    reference_safe_frontier_size: int = DEFAULT_REFERENCE_SAFE_FRONTIER_SIZE,
    reference_trace_caps: Sequence[int] = DEFAULT_REFERENCE_TRACE_CAPS,
    hard_expansion_cap: int = DEFAULT_HARD_EXPANSION_CAP,
    feasibility_criteria: PilotFeasibilityCriteria | None = None,
    hard_training_episode_count: int | None = None,
    pilot_relaunch_checks: Mapping[str, bool] | None = None,
    benchmark_provenance: Mapping[str, Any] | None = None,
) -> FeatureBenchmarkArtifacts:
    """Validate and atomically write the complete performance artifact set."""

    expected_sizes = _validated_axis(expected_frontier_sizes, name="frontier sizes")
    expected_caps = _validated_axis(expected_staged_caps, name="staged caps")
    safe_size = _positive_int(
        reference_safe_frontier_size, name="reference_safe_frontier_size"
    )
    trace_caps = _validated_axis(reference_trace_caps, name="reference trace caps")
    normalized_micro = sorted(
        (MicrobenchmarkMeasurement.from_value(value) for value in microbenchmarks),
        key=lambda value: value.frontier_size,
    )
    normalized_end = sorted(
        (EndToEndMeasurement.from_value(value) for value in end_to_end),
        key=lambda value: value.expansion_cap,
    )
    if len({value.frontier_size for value in normalized_micro}) != len(normalized_micro):
        raise ValueError("microbenchmark frontier sizes must be unique")
    if len({value.expansion_cap for value in normalized_end}) != len(normalized_end):
        raise ValueError("end-to-end expansion caps must be unique")
    baseline_payload = dict(baseline or baseline_f653193())
    if baseline_payload.get("scientific_scheduler_evidence") is not False:
        raise ValueError("baseline must be explicitly marked non-scientific")
    micro_rows = _micro_rows(
        normalized_micro,
        correctness_gate=correctness_gate,
        baseline=baseline_payload,
        reference_safe_frontier_size=safe_size,
    )
    end_rows = _end_to_end_rows(normalized_end, reference_trace_caps=trace_caps)
    qualification = qualify_feature_benchmark(
        micro_rows,
        end_rows,
        correctness_gate=correctness_gate,
        implementation_checks=implementation_checks,
        expected_frontier_sizes=expected_sizes,
        expected_staged_caps=expected_caps,
        reference_safe_frontier_size=safe_size,
        reference_trace_caps=trace_caps,
    )
    projection = project_pilot_cost_from_scaling(
        micro_rows,
        end_rows,
        qualification=qualification,
        hard_expansion_cap=hard_expansion_cap,
        feasibility_criteria=feasibility_criteria,
        hard_training_episode_count=hard_training_episode_count,
        pilot_relaunch_checks=pilot_relaunch_checks,
    )

    output = Path(output_directory)
    baseline_path = output / "baseline.json"
    micro_path = output / "microbenchmarks.csv"
    end_path = output / "end_to_end_scaling.csv"
    profiles_path = output / "profiles"
    report_path = output / "scaling_report.md"
    projection_path = output / "projected_pilot_cost.json"
    profiles_path.mkdir(parents=True, exist_ok=True)
    profile_inventory = sorted(
        path.relative_to(output).as_posix()
        for path in profiles_path.rglob("*")
        if path.is_file()
    )
    baseline_document = {
        **baseline_payload,
        "benchmark_schema_version": ARTICLE_V1_FEATURE_BENCHMARK_SCHEMA,
        "feature_schema_version": "article-v1-31d",
        "feature_evaluator_schema_version": ARTICLE_V1_FEATURE_EVALUATOR_SCHEMA,
        "reference_feature_evaluator_schema_version": (
            ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA
        ),
        "benchmark_provenance": _benchmark_provenance(benchmark_provenance),
        "benchmark_environment": capture_benchmark_environment(),
        "correctness_gate": correctness_gate.as_dict(),
        "reference_safe_frontier_size": safe_size,
        "default_frontier_sizes": list(expected_sizes),
        "default_staged_expansion_caps": list(expected_caps),
    }
    report = _scaling_report(
        baseline=baseline_payload,
        correctness_gate=correctness_gate,
        micro_rows=micro_rows,
        end_rows=end_rows,
        qualification=qualification,
        projection=projection,
        profile_inventory=profile_inventory,
    )
    # All validation and model fitting precede writes, so invalid inputs cannot
    # partially replace an earlier qualification bundle.
    _atomic_write(baseline_path, _json_bytes(baseline_document))
    _atomic_write(micro_path, _csv_bytes(micro_rows, MICROBENCHMARK_COLUMNS))
    _atomic_write(end_path, _csv_bytes(end_rows, END_TO_END_COLUMNS))
    _atomic_write(report_path, report.encode("utf-8"))
    _atomic_write(projection_path, _json_bytes(projection))
    return FeatureBenchmarkArtifacts(
        output_directory=output,
        baseline_json=baseline_path,
        microbenchmarks_csv=micro_path,
        end_to_end_scaling_csv=end_path,
        profiles_directory=profiles_path,
        scaling_report_md=report_path,
        projected_pilot_cost_json=projection_path,
        qualification=qualification,
        projection=projection,
    )


def benchmark_feature_evaluator(
    adapter: FeatureBenchmarkAdapter,
    output_directory: str | Path,
    *,
    correctness_gate: CorrectnessGate,
    implementation_checks: Mapping[str, bool],
    frontier_sizes: Sequence[int] = DEFAULT_FRONTIER_SIZES,
    staged_caps: Sequence[int] = DEFAULT_STAGED_EXPANSION_CAPS,
    reference_safe_frontier_size: int = DEFAULT_REFERENCE_SAFE_FRONTIER_SIZE,
    reference_trace_caps: Sequence[int] = DEFAULT_REFERENCE_TRACE_CAPS,
    baseline: Mapping[str, Any] | None = None,
    feasibility_criteria: PilotFeasibilityCriteria | None = None,
    hard_expansion_cap: int = DEFAULT_HARD_EXPANSION_CAP,
    hard_training_episode_count: int | None = None,
    pilot_relaunch_checks: Mapping[str, bool] | None = None,
    benchmark_provenance: Mapping[str, Any] | None = None,
    require_correctness_before_timing: bool = True,
    profile_writer: Callable[[Path], object] | None = None,
) -> FeatureBenchmarkArtifacts:
    """Run the staged adapter API and write its qualification artifacts.

    ``measure_microbenchmark`` receives ``include_reference=True`` only through
    the configured reference-safe frontier size. ``measure_end_to_end``
    receives it only for the exact bounded trace-parity caps (32 and 64 by
    default).  This function does not mutate configs or launch a pilot.
    """

    _require_pre_timing_gates(
        correctness_gate,
        implementation_checks,
        require_correctness=require_correctness_before_timing,
    )
    sizes = _validated_axis(frontier_sizes, name="frontier sizes")
    caps = _validated_axis(staged_caps, name="staged caps")
    safe_size = _positive_int(
        reference_safe_frontier_size, name="reference_safe_frontier_size"
    )
    trace_caps = _validated_axis(reference_trace_caps, name="reference trace caps")
    prepare_microbenchmarks = getattr(adapter, "prepare_microbenchmarks", None)
    if callable(prepare_microbenchmarks):
        prepare_microbenchmarks(sizes)
    micro = [
        MicrobenchmarkMeasurement.from_value(
            adapter.measure_microbenchmark(
                size, include_reference=size <= safe_size
            )
        )
        for size in sizes
    ]
    end = [
        EndToEndMeasurement.from_value(
            adapter.measure_end_to_end(
                cap, include_reference=cap in set(trace_caps)
            )
        )
        for cap in caps
    ]
    output = Path(output_directory)
    profiles = output / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    if profile_writer is not None:
        profile_writer(profiles)
    return write_feature_benchmark_artifacts(
        output,
        correctness_gate=correctness_gate,
        microbenchmarks=micro,
        end_to_end=end,
        implementation_checks=implementation_checks,
        baseline=baseline,
        expected_frontier_sizes=sizes,
        expected_staged_caps=caps,
        reference_safe_frontier_size=safe_size,
        reference_trace_caps=trace_caps,
        hard_expansion_cap=hard_expansion_cap,
        feasibility_criteria=feasibility_criteria,
        hard_training_episode_count=hard_training_episode_count,
        pilot_relaunch_checks=pilot_relaunch_checks,
        benchmark_provenance=benchmark_provenance,
    )


__all__ = [
    "ARTICLE_V1_FEATURE_BASELINE_SCHEMA",
    "ARTICLE_V1_FEATURE_BENCHMARK_SCHEMA",
    "ARTICLE_V1_FEATURE_EVALUATOR_SCHEMA",
    "ARTICLE_V1_FEATURE_PROJECTION_SCHEMA",
    "ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA",
    "CorrectnessGate",
    "DEFAULT_CORRECTNESS_TEST_NODE_IDS",
    "DEFAULT_FRONTIER_SIZES",
    "DEFAULT_HARD_EXPANSION_CAP",
    "DEFAULT_REFERENCE_SAFE_FRONTIER_SIZE",
    "DEFAULT_REFERENCE_TRACE_CAPS",
    "DEFAULT_STAGED_EXPANSION_CAPS",
    "END_TO_END_COLUMNS",
    "ENGINEERING_EVIDENCE_CLASS",
    "EndToEndMeasurement",
    "FeatureBenchmarkAdapter",
    "FeatureBenchmarkArtifacts",
    "MICROBENCHMARK_COLUMNS",
    "MicrobenchmarkMeasurement",
    "PilotFeasibilityCriteria",
    "PILOT_HARD_3Q_TARGET_ID",
    "PRODUCTION_DOMINANCE_IMPLEMENTATION_CHECK",
    "REQUIRED_CORRECTNESS_CHECKS",
    "REQUIRED_PILOT_RELAUNCH_CHECKS",
    "RepositoryArticleV1FeatureBenchmarkAdapter",
    "baseline_f653193",
    "benchmark_feature_evaluator",
    "capture_benchmark_environment",
    "create_repository_feature_benchmark_adapter",
    "inspect_production_dominance_update",
    "project_pilot_cost_from_scaling",
    "qualify_feature_benchmark",
    "run_repository_feature_benchmark",
    "run_focused_correctness_gate",
    "write_implementation_check_evidence",
    "write_feature_benchmark_artifacts",
]
