"""Article V1 training/evaluation runner for native frontier-record ranking.

The generator witness is never passed to an environment.  Every evaluation
starts with a fresh frontier, target-metric cache, and independent Article V1
certifier.  Full publication commands are intentionally explicit and are not
part of the ordinary unit-test suite; ``mini-ci`` is the bounded smoke path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import redirect_stdout
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from io import StringIO
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from benchmarks.article_native_corpus import (
    ARTICLE_V1_TRAINING_BUDGET_POLICY,
    CHECKPOINT_FAMILIES,
    COMPLETE_TRAINING_SCOPE,
    NATIVE_GATE_NAMES,
    OOD_LENGTH_CHECKPOINT_FAMILY,
    PARTIAL_SMOKE_TRAINING_SCOPE,
    STANDARD_CHECKPOINT_FAMILY,
    TRAINING_SCOPE_MODES,
    ArticleV1CheckpointScope,
    ArticleV1Corpus,
    ArticleV1CorpusConfig,
    ArticleV1EvaluationTarget,
    ArticleV1TargetCase,
    article_delta_phi,
    build_article_v1_corpus,
    dense_target_digest,
    load_article_v1_config,
)
from certification.article_v1 import ArticleV1CertificationEngine
from certification.base import CertStatus
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from config import Config
from env.rl_env import CircuitSynthesisEnv
from enums import GateType
from evaluate import evaluate
from experiments.profiles import ARTICLE_V1_PROFILE
from experiments.article_v1_progress import (
    ArticleV1ProgressEvent,
    ArticleV1ProgressReporter,
    ProgressCadence,
    utc_timestamp,
)
from experiments.article_v1_replay_timing import (
    REPLAY_TIMING_EXPECTED_EXPANSIONS,
    ArticleV1ReplayTimingEvidence,
    ReplayValidationResult,
    checkpoint_file_sha256,
    measure_replay_timing,
    write_replay_timing,
)
from experiments.article_v1_runtime_snapshot import (
    ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA,
    DEFAULT_RUNTIME_SNAPSHOT_INTERVAL,
    ArticleV1RuntimeSnapshotStore,
    ArticleV1RuntimeState,
    LoadedRuntimeSnapshot,
)
from experiments.article_v1_training_checkpoint import (
    ArticleV1CheckpointProvenance,
    ArticleV1EventJournal,
    ArticleV1JournalEntry,
    ArticleV1TrainingCheckpointStore,
    CheckpointCadence,
    CheckpointCadenceGate,
    CheckpointCompatibilityError,
    CheckpointFormatError,
    MidEpisodeCheckpoint,
    ReplayObservation,
    ResumeExpectation,
    TrainingProgressCheckpoint,
    feature_row_digest,
    policy_weight_digest,
    portable_digest,
    replay_and_validate,
    validate_pending_resume_state,
    validate_resume_compatibility,
)
from reporting.article_v1 import (
    ARTICLE_V1_RAW_RUN_SCHEMA,
    ArticleV1RunStore,
    run_identity_payload,
    unique_run_key,
    write_article_v1_report,
)
from search.action_space import generate_actions
from rl.article_features import (
    ARTICLE_V1_FEATURE_NAMES,
    ARTICLE_V1_FEATURE_SCHEMA_VERSION,
    ArticleTargetContext,
    ArticleV1FeatureProvider,
    ArticleV1NoTargetFeatureProvider,
    ArticleV1NoZFeatureProvider,
)
from rl.policy import LinearQPolicy
from train import Trainer, TrainerBoundaryEvent, TrainerEpisodeResume


ARTICLE_V1_RUNNER_SCHEMA = "article-v1-publication-runner-v4"
ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA = "article-v1-campaign-audit-v1"
ARTICLE_V1_REPLAY_CAPTURE_SCHEMA = "article-v1-replay-checkpoint-capture-v1"
PRIMARY_SCHEDULERS = (
    "fifo",
    "lifo",
    "uniform_cost",
    "seeded_random",
    "zero_weight_linear",
    "article_target_distance",
    "article_sarsa",
)
FEATURE_VARIANTS = {
    "article-v1-31d": ArticleV1FeatureProvider,
    "article-v1-no-target-28d": ArticleV1NoTargetFeatureProvider,
    "article-v1-no-z-21d": ArticleV1NoZFeatureProvider,
}
FEATURE_NAME_VARIANTS = {
    schema: tuple(provider_type.feature_names)
    for schema, provider_type in FEATURE_VARIANTS.items()
}

_GIT_PROVENANCE_CACHE: dict[str, object] | None = None
_RELEVANT_UNTRACKED_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}


class _ProgressFeatureTimingWindow:
    """Track exact new compact-batch durations without boundary double counts."""

    def __init__(self, *, initial_count: int = 0, window_size: int = 25) -> None:
        if isinstance(initial_count, bool) or int(initial_count) < 0:
            raise ValueError("initial compact-batch count must be nonnegative")
        if isinstance(window_size, bool) or int(window_size) < 1:
            raise ValueError("feature timing window size must be positive")
        self._observed_count = int(initial_count)
        self._samples_ns: deque[int] = deque(maxlen=int(window_size))

    def observe(
        self,
        instrumentation: Mapping[str, object],
        *,
        recent_batch_times_ns: Sequence[int],
    ) -> tuple[float, float]:
        count = int(instrumentation.get("compact_batch_count", 0))
        last_ns = int(instrumentation.get("last_compact_batch_time_ns", 0))
        if count < 0 or last_ns < 0:
            raise RuntimeError("feature batch timing instrumentation is negative")
        if count < self._observed_count:
            # The provider owns a fresh per-episode index after environment reset.
            self._observed_count = 0
            self._samples_ns.clear()
        new_count = count - self._observed_count
        recent = tuple(int(value) for value in recent_batch_times_ns)
        if any(value < 0 for value in recent):
            raise RuntimeError("recent feature batch timings are negative")
        if new_count > len(recent):
            # A callback may be attached after more than the provider's retained
            # 25 samples.  Reloading that exact suffix still yields the exact
            # requested rolling-25 statistic.
            self._samples_ns.clear()
            self._samples_ns.extend(recent)
        elif new_count:
            self._samples_ns.extend(recent[-new_count:])
        self._observed_count = count
        last_seconds = float(last_ns / 1_000_000_000) if count else 0.0
        rolling_seconds = (
            float(sum(self._samples_ns) / len(self._samples_ns) / 1_000_000_000)
            if self._samples_ns
            else 0.0
        )
        return last_seconds, rolling_seconds

    def reset_episode(self) -> None:
        self._observed_count = 0
        self._samples_ns.clear()


class _EpisodeProgressClock:
    """Episode-local elapsed/rate clock whose reset follows the final event."""

    def __init__(
        self,
        *,
        started: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not math.isfinite(float(started)):
            raise ValueError("episode progress start must be finite")
        self._started = float(started)
        self._clock = clock

    def measure(self, expansion: int) -> tuple[float, float]:
        if isinstance(expansion, bool) or not isinstance(expansion, int) or expansion < 0:
            raise ValueError("episode progress expansion must be nonnegative")
        elapsed = float(self._clock()) - self._started
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("episode progress clock regressed or is non-finite")
        return elapsed, (float(expansion / elapsed) if elapsed else 0.0)

    def reset(self, *, now: float | None = None) -> None:
        current = float(self._clock() if now is None else now)
        if not math.isfinite(current):
            raise RuntimeError("episode progress reset time is non-finite")
        self._started = current


_TIMING_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    return str(value)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_relevant_untracked_source(path: str) -> bool:
    normalized = path.replace("\\", "/")
    components = tuple(component for component in normalized.split("/") if component)
    if any(
        component in {".git", ".venv", "__pycache__", "outputs"}
        or component.startswith((".pytest-", ".stage"))
        for component in components
    ):
        return False
    return Path(normalized).suffix.lower() in _RELEVANT_UNTRACKED_SUFFIXES


def git_provenance(*, refresh: bool = False) -> dict[str, object]:
    """Return commit plus a content digest for the executable source tree.

    Ordinary callers retain the process cache. Qualification commands may set
    ``refresh=True`` to take an uncached before/after integrity snapshot.
    """

    global _GIT_PROVENANCE_CACHE
    if not isinstance(refresh, bool):
        raise ValueError("git provenance refresh must be a bool")
    if _GIT_PROVENANCE_CACHE is not None and not refresh:
        return {
            **_GIT_PROVENANCE_CACHE,
            "relevant_untracked_files": list(
                _GIT_PROVENANCE_CACHE["relevant_untracked_files"]
            ),
        }

    def text_command(*arguments: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            ("git", *arguments),
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.stdout.strip()

    try:
        repository_root = Path(text_command("rev-parse", "--show-toplevel"))
        commit_sha = text_command("rev-parse", "HEAD", cwd=repository_root)
        branch = text_command("branch", "--show-current", cwd=repository_root)
        tracked_diff = subprocess.run(
            ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"),
            check=True,
            capture_output=True,
            cwd=repository_root,
        ).stdout
        untracked_output = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            check=True,
            capture_output=True,
            cwd=repository_root,
        ).stdout
        untracked = sorted(
            os.fsdecode(value)
            for value in untracked_output.split(b"\0")
            if value and _is_relevant_untracked_source(os.fsdecode(value))
        )
        digest = sha256()
        digest.update(b"article-v1-source-worktree-v1\0tracked-diff\0")
        digest.update(tracked_diff)
        for relative in untracked:
            encoded_path = relative.replace("\\", "/").encode(
                "utf-8", errors="surrogateescape"
            )
            digest.update(b"\0untracked\0")
            digest.update(encoded_path)
            digest.update(b"\0")
            digest.update((repository_root / relative).read_bytes())
        provenance: dict[str, object] = {
            "commit_sha": commit_sha,
            "branch": branch,
            "dirty_worktree": bool(tracked_diff or untracked),
            "source_worktree_digest": f"sha256:{digest.hexdigest()}",
            "relevant_untracked_files": untracked,
        }
    except (OSError, subprocess.CalledProcessError):
        provenance = {
            "commit_sha": "unknown",
            "branch": "unknown",
            "dirty_worktree": True,
            "source_worktree_digest": "unknown",
            "relevant_untracked_files": [],
        }
    _GIT_PROVENANCE_CACHE = provenance
    return {
        **provenance,
        "relevant_untracked_files": list(provenance["relevant_untracked_files"]),
    }


def _portable_environment_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def environment_metadata() -> dict[str, object]:
    packages: dict[str, str] = {}
    for distribution in ("numpy", "gymnasium", "pytest", "pytest-cov"):
        try:
            packages[distribution] = package_version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "packages": packages,
        "processor": platform.processor(),
        "executable": _portable_environment_path(sys.executable),
        "cwd": ".",
        "execution": {
            "concurrency_mode": "serial-single-process",
            "worker_count": 1,
            "thread_environment": {
                name: os.environ.get(name)
                for name in _TIMING_THREAD_ENVIRONMENT
            },
        },
        "git": git_provenance(),
    }


def _target(case: ArticleV1TargetCase | ArticleV1EvaluationTarget):
    if isinstance(case, ArticleV1TargetCase):
        return case.synthesis_target()
    return case.target


def _case_metadata(case: ArticleV1TargetCase | ArticleV1EvaluationTarget) -> dict[str, object]:
    return {
        "target_id": case.target_id,
        "split": case.split,
        "difficulty": case.difficulty,
        "num_qubits": case.num_qubits,
        "generator_length": case.generator_length,
        "budget": case.budget.metadata(),
        "target_specific_reachability_oracle": False,
    }


def _resource_budget_payload(case) -> dict[str, int]:
    values = case.budget.metadata()
    return {
        name: int(values[name])
        for name in (
            "max_t_count",
            "max_two_qubit_count",
            "max_gates",
            "max_depth",
        )
    }


ARTICLE_V1_CHECKPOINT_SCHEMA = "article-v1-transferable-linear-checkpoint-v4"


def _weights_digest(
    weights: Sequence[float],
    *,
    training_seed: int,
    feature_schema: str,
    feature_evaluator_schema: str,
    checkpoint_family: str,
    training_scope_mode: str,
    corpus_config_digest: str,
    training_target_ids: Sequence[str],
    learning_rate: float,
    epsilon_schedule: Sequence[tuple[str, float]],
    training_beta: float,
    certification_tolerance: float,
    episodes_per_target: int,
    expansion_cap: int | None,
    training_budget_policy: str,
    effective_training_expansion_budgets: Sequence[tuple[str, int]],
) -> str:
    digest = sha256()
    digest.update(b"article-v1-transferable-linear-checkpoint-digest-v3\0")
    digest.update(str(int(training_seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(feature_schema.encode("ascii"))
    digest.update(b"\0")
    digest.update(feature_evaluator_schema.encode("ascii"))
    digest.update(b"\0")
    digest.update(checkpoint_family.encode("ascii"))
    digest.update(b"\0")
    digest.update(training_scope_mode.encode("ascii"))
    digest.update(b"\0")
    digest.update(corpus_config_digest.encode("ascii"))
    digest.update(b"\0")
    for target_id in training_target_ids:
        digest.update(str(target_id).encode("ascii"))
        digest.update(b"\0")
    protocol = {
        "learning_rate": float(learning_rate),
        "epsilon_schedule": {
            str(name): float(value) for name, value in epsilon_schedule
        },
        "training_beta": float(training_beta),
        "certification_tolerance": float(certification_tolerance),
        "episodes_per_target": int(episodes_per_target),
        "expansion_cap": None if expansion_cap is None else int(expansion_cap),
        "training_budget_policy": str(training_budget_policy),
        "effective_training_expansion_budgets": [
            [str(target_id), int(budget)]
            for target_id, budget in effective_training_expansion_budgets
        ],
    }
    digest.update(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(weights, dtype="<f8").tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class ArticleV1Checkpoint:
    training_seed: int
    weights: tuple[float, ...]
    feature_schema_version: str
    ordered_feature_names: tuple[str, ...]
    reward_schema_version: str
    target_metric_schema_version: str
    certification_schema_version: str
    learning_rate: float
    discount: float
    epsilon_schedule: tuple[tuple[str, float], ...]
    checkpoint_family: str
    training_scope_mode: str
    training_beta: float
    training_certification_tolerance: float
    training_episodes_per_target: int
    training_expansion_cap: int | None
    training_budget_policy: str
    effective_training_expansion_budgets: tuple[tuple[str, int], ...]
    training_target_ids: tuple[str, ...]
    training_histories: tuple[Mapping[str, object], ...]
    corpus_config_digest: str
    feature_evaluator_schema_version: str = (
        ARTICLE_V1_PROFILE.feature_evaluator_schema
    )

    @property
    def weight_digest(self) -> str:
        return _weights_digest(
            self.weights,
            training_seed=self.training_seed,
            feature_schema=self.feature_schema_version,
            feature_evaluator_schema=self.feature_evaluator_schema_version,
            checkpoint_family=self.checkpoint_family,
            training_scope_mode=self.training_scope_mode,
            corpus_config_digest=self.corpus_config_digest,
            training_target_ids=self.training_target_ids,
            learning_rate=self.learning_rate,
            epsilon_schedule=self.epsilon_schedule,
            training_beta=self.training_beta,
            certification_tolerance=self.training_certification_tolerance,
            episodes_per_target=self.training_episodes_per_target,
            expansion_cap=self.training_expansion_cap,
            training_budget_policy=self.training_budget_policy,
            effective_training_expansion_budgets=(
                self.effective_training_expansion_budgets
            ),
        )

    def validate_contract(self, *, require_nonzero: bool = False) -> None:
        """Reject schema drift instead of resizing or reinterpreting weights."""

        expected_names = FEATURE_NAME_VARIANTS.get(self.feature_schema_version)
        if expected_names is None:
            raise ValueError(
                f"unsupported Article V1 checkpoint feature schema "
                f"{self.feature_schema_version!r}"
            )
        if self.ordered_feature_names != expected_names:
            raise ValueError("checkpoint ordered feature names do not match its schema")
        if len(self.weights) != len(expected_names):
            raise ValueError("checkpoint weight dimension does not match its schema")
        if (
            self.feature_evaluator_schema_version
            != ARTICLE_V1_PROFILE.feature_evaluator_schema
        ):
            raise ValueError("checkpoint feature evaluator schema is not Article V1")
        if self.reward_schema_version != ARTICLE_V1_PROFILE.reward_schema:
            raise ValueError("checkpoint reward schema is not Article V1")
        if self.target_metric_schema_version != ARTICLE_V1_PROFILE.target_metric_schema:
            raise ValueError("checkpoint target-metric schema is not Article V1")
        if self.certification_schema_version != ARTICLE_V1_PROFILE.certification_schema:
            raise ValueError("checkpoint certification schema is not Article V1")
        if self.discount != ARTICLE_V1_PROFILE.gamma:
            raise ValueError("checkpoint discount is not Article V1 gamma=1")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise ValueError("checkpoint training seed must be an integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("checkpoint learning rate must be finite and positive")
        if tuple(name for name, _value in self.epsilon_schedule) != (
            "decay",
            "minimum",
            "start",
        ):
            raise ValueError(
                "checkpoint epsilon schedule must define decay/minimum/start"
            )
        epsilon = dict(self.epsilon_schedule)
        if not (
            0.0 <= epsilon["minimum"] <= epsilon["start"] <= 1.0
            and 0.0 < epsilon["decay"] <= 1.0
            and all(np.isfinite(value) for value in epsilon.values())
        ):
            raise ValueError("checkpoint epsilon schedule is invalid")
        if self.checkpoint_family not in CHECKPOINT_FAMILIES:
            raise ValueError(
                f"checkpoint family must be one of {CHECKPOINT_FAMILIES!r}"
            )
        if self.training_scope_mode not in TRAINING_SCOPE_MODES:
            raise ValueError(
                f"checkpoint training scope mode must be one of {TRAINING_SCOPE_MODES!r}"
            )
        if not np.isfinite(self.training_beta) or self.training_beta < 0.0:
            raise ValueError("checkpoint training beta must be finite and non-negative")
        if (
            not np.isfinite(self.training_certification_tolerance)
            or self.training_certification_tolerance <= 0.0
        ):
            raise ValueError(
                "checkpoint training certification tolerance must be finite and positive"
            )
        if (
            isinstance(self.training_episodes_per_target, bool)
            or not isinstance(self.training_episodes_per_target, int)
            or self.training_episodes_per_target < 1
        ):
            raise ValueError("checkpoint training episodes per target must be positive")
        if self.training_expansion_cap is not None and (
            isinstance(self.training_expansion_cap, bool)
            or not isinstance(self.training_expansion_cap, int)
            or self.training_expansion_cap < 1
        ):
            raise ValueError("checkpoint training expansion cap must be positive or None")
        if self.training_budget_policy != ARTICLE_V1_TRAINING_BUDGET_POLICY:
            raise ValueError("unsupported checkpoint training budget policy")
        if not self.corpus_config_digest:
            raise ValueError("checkpoint corpus config digest is required")
        if not self.training_target_ids:
            raise ValueError("checkpoint training target IDs must be nonempty")
        if (
            any(not target_id for target_id in self.training_target_ids)
            or len(set(self.training_target_ids)) != len(self.training_target_ids)
        ):
            raise ValueError("checkpoint training target IDs must be nonempty and unique")
        budget_ids = tuple(
            target_id
            for target_id, _budget in self.effective_training_expansion_budgets
        )
        if budget_ids != self.training_target_ids:
            raise ValueError(
                "checkpoint effective training budgets do not match training target IDs"
            )
        for _target_id, budget in self.effective_training_expansion_budgets:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError(
                    "checkpoint effective training expansion budgets must be positive"
                )
            if (
                self.training_expansion_cap is not None
                and budget > self.training_expansion_cap
            ):
                raise ValueError(
                    "checkpoint effective training budget exceeds its expansion cap"
                )
        if self.training_histories:
            history_ids = tuple(
                str(history.get("target_id", ""))
                for history in self.training_histories
            )
            if history_ids != self.training_target_ids:
                raise ValueError(
                    "checkpoint training histories do not match training target IDs"
                )
            if any(
                str(history.get("split", "")) != "train"
                for history in self.training_histories
            ):
                raise ValueError("checkpoint training histories contain a held-out split")
        values = np.asarray(self.weights, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("checkpoint weights must be finite")
        if require_nonzero and not np.any(np.abs(values) > 0.0):
            raise ValueError("article_sarsa requires a nonzero trained checkpoint")

    def validate_against_scope(
        self,
        scope: ArticleV1CheckpointScope,
        *,
        require_nonzero: bool = True,
    ) -> None:
        """Bind checkpoint provenance to a frozen corpus/training scope."""

        self.validate_contract(require_nonzero=require_nonzero)
        if self.feature_schema_version != scope.expected_feature_schema_version:
            raise ValueError(
                "checkpoint feature schema does not match the expected evaluation schema"
            )
        if self.corpus_config_digest != scope.corpus_config_digest:
            raise ValueError("checkpoint corpus config digest does not match evaluation scope")
        if self.checkpoint_family != scope.checkpoint_family:
            raise ValueError(
                "checkpoint family does not match the standard/OOD evaluation scope"
            )
        if self.training_scope_mode != scope.training_scope_mode:
            raise ValueError(
                "checkpoint training scope mode does not match evaluation scope"
            )
        if self.training_seed not in scope.allowed_training_seeds:
            raise ValueError(
                "checkpoint training seed is outside the permitted evaluation scope"
            )
        if self.learning_rate != scope.expected_learning_rate:
            raise ValueError(
                "checkpoint learning rate does not match the evaluation scope"
            )
        if self.epsilon_schedule != scope.expected_epsilon_schedule:
            raise ValueError(
                "checkpoint epsilon schedule does not match the evaluation scope"
            )

        training_ids = set(self.training_target_ids)
        held_out_ids = set(scope.held_out_target_ids)
        leaked = training_ids & held_out_ids
        if leaked:
            raise ValueError(
                "checkpoint training IDs include held-out evaluation targets: "
                + ", ".join(sorted(leaked))
            )
        outside = training_ids - set(scope.allowed_training_target_ids)
        if outside:
            raise ValueError(
                "checkpoint training IDs fall outside the permitted training scope: "
                + ", ".join(sorted(outside))
            )
        if self.training_target_ids != scope.allowed_training_target_ids:
            raise ValueError(
                "checkpoint training IDs do not exactly match the complete evaluation scope"
            )
        if self.training_beta != scope.expected_training_beta:
            raise ValueError(
                "checkpoint training beta does not match the evaluation scope"
            )
        if (
            self.training_certification_tolerance
            != scope.expected_certification_tolerance
        ):
            raise ValueError(
                "checkpoint training certification tolerance does not match the "
                "evaluation scope"
            )
        if self.training_episodes_per_target != scope.expected_episodes_per_target:
            raise ValueError(
                "checkpoint training episodes per target do not match the evaluation scope"
            )
        if self.training_expansion_cap != scope.expected_expansion_cap:
            raise ValueError(
                "checkpoint training expansion cap does not match the evaluation scope"
            )
        if self.training_budget_policy != scope.training_budget_policy:
            raise ValueError(
                "checkpoint training budget policy does not match the evaluation scope"
            )
        if (
            self.effective_training_expansion_budgets
            != scope.expected_training_expansion_budgets
        ):
            raise ValueError(
                "checkpoint effective training budgets do not match the evaluation scope"
            )

    def validate_for_evaluation(
        self,
        scope: ArticleV1CheckpointScope,
        case: ArticleV1TargetCase | ArticleV1EvaluationTarget,
        *,
        require_nonzero: bool = True,
    ) -> None:
        """Bind provenance and one target before checkpoint evaluation."""

        scope.validate_evaluation_target(case)
        self.validate_against_scope(scope, require_nonzero=require_nonzero)

    def metadata(self, *, include_weights: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "checkpoint_schema": ARTICLE_V1_CHECKPOINT_SCHEMA,
            "profile_name": ARTICLE_V1_PROFILE.name,
            "algorithm": "linear-semi-gradient-sarsa(0)",
            "feature_schema_version": self.feature_schema_version,
            "feature_evaluator_schema_version": (
                self.feature_evaluator_schema_version
            ),
            "ordered_feature_names": list(self.ordered_feature_names),
            "feature_dimension": len(self.weights),
            "reward_schema_version": self.reward_schema_version,
            "target_metric_schema_version": self.target_metric_schema_version,
            "certification_schema_version": self.certification_schema_version,
            "target_fingerprint": f"training-corpus:{self.corpus_config_digest}",
            "target_context_binding_digest": self.corpus_config_digest,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon_schedule": dict(self.epsilon_schedule),
            "checkpoint_family": self.checkpoint_family,
            "training_protocol": {
                "training_scope_mode": self.training_scope_mode,
                "beta": self.training_beta,
                "certification_tolerance": self.training_certification_tolerance,
                "episodes_per_target": self.training_episodes_per_target,
                "expansion_cap": self.training_expansion_cap,
                "budget_policy": self.training_budget_policy,
                "effective_expansion_budgets": [
                    {"target_id": target_id, "expansion_budget": budget}
                    for target_id, budget in self.effective_training_expansion_budgets
                ],
            },
            "training_seed": self.training_seed,
            "weight_digest": self.weight_digest,
            "weight_norm": float(np.linalg.norm(self.weights)),
            "training_target_ids": list(self.training_target_ids),
            "training_histories": [_json_ready(value) for value in self.training_histories],
            "corpus_config_digest": self.corpus_config_digest,
        }
        if include_weights:
            result["weights"] = list(self.weights)
        return result

    def save(self, path: str | Path) -> None:
        payload = {
            **self.metadata(include_weights=True),
            "code": git_provenance(),
        }
        _atomic_json(Path(path), payload)

    @classmethod
    def load(cls, path: str | Path) -> "ArticleV1Checkpoint":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Article V1 checkpoint must contain a JSON object")
        if payload.get("checkpoint_schema") != ARTICLE_V1_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported Article V1 checkpoint schema")
        if payload.get("profile_name") != ARTICLE_V1_PROFILE.name:
            raise ValueError("checkpoint profile is not the frozen Article V1 profile")
        checkpoint_code = payload.get("code")
        if not isinstance(checkpoint_code, Mapping):
            raise ValueError("checkpoint code provenance is missing")
        if type(checkpoint_code.get("dirty_worktree")) is not bool:
            raise ValueError(
                "checkpoint code provenance dirty_worktree must be boolean"
            )
        current_code = git_provenance()
        for field_name in (
            "commit_sha",
            "source_worktree_digest",
            "dirty_worktree",
        ):
            if checkpoint_code.get(field_name) != current_code[field_name]:
                raise ValueError(
                    f"checkpoint code provenance {field_name} does not match "
                    "the current source tree"
                )
        feature_schema = str(payload["feature_schema_version"])
        names = tuple(str(value) for value in payload["ordered_feature_names"])
        weights = tuple(float(value) for value in payload["weights"])
        if len(names) != len(weights) or int(payload["feature_dimension"]) != len(weights):
            raise ValueError("checkpoint feature names/dimension/weights disagree")
        result = cls(
            training_seed=int(payload["training_seed"]),
            weights=weights,
            feature_schema_version=feature_schema,
            feature_evaluator_schema_version=str(
                payload["feature_evaluator_schema_version"]
            ),
            ordered_feature_names=names,
            reward_schema_version=str(payload["reward_schema_version"]),
            target_metric_schema_version=str(payload["target_metric_schema_version"]),
            certification_schema_version=str(payload["certification_schema_version"]),
            learning_rate=float(payload["learning_rate"]),
            discount=float(payload["discount"]),
            epsilon_schedule=tuple(
                (str(key), float(value))
                for key, value in sorted(payload["epsilon_schedule"].items())
            ),
            checkpoint_family=str(payload["checkpoint_family"]),
            training_scope_mode=str(
                payload["training_protocol"]["training_scope_mode"]
            ),
            training_beta=float(payload["training_protocol"]["beta"]),
            training_certification_tolerance=float(
                payload["training_protocol"]["certification_tolerance"]
            ),
            training_episodes_per_target=int(
                payload["training_protocol"]["episodes_per_target"]
            ),
            training_expansion_cap=(
                None
                if payload["training_protocol"]["expansion_cap"] is None
                else int(payload["training_protocol"]["expansion_cap"])
            ),
            training_budget_policy=str(
                payload["training_protocol"]["budget_policy"]
            ),
            effective_training_expansion_budgets=tuple(
                (
                    str(record["target_id"]),
                    int(record["expansion_budget"]),
                )
                for record in payload["training_protocol"][
                    "effective_expansion_budgets"
                ]
            ),
            training_target_ids=tuple(str(value) for value in payload["training_target_ids"]),
            training_histories=tuple(payload.get("training_histories", ())),
            corpus_config_digest=str(payload["corpus_config_digest"]),
        )
        if result.weight_digest != payload.get("weight_digest"):
            raise ValueError("checkpoint weight digest mismatch")
        result.validate_contract()
        return result


def _validate_checkpoint_campaign(
    checkpoints: Sequence[ArticleV1Checkpoint],
    scope: ArticleV1CheckpointScope,
) -> None:
    actual_seeds = tuple(checkpoint.training_seed for checkpoint in checkpoints)
    if len(set(actual_seeds)) != len(actual_seeds):
        raise ValueError("checkpoint campaign contains duplicate training seeds")
    if set(actual_seeds) != set(scope.allowed_training_seeds):
        raise ValueError(
            "checkpoint campaign seeds do not exactly match the required training seeds"
        )


def _load_or_train_article_v1_checkpoint(
    path: str | Path,
    *,
    scope: ArticleV1CheckpointScope,
    expected_training_seed: int,
    train_callback: Callable[[], ArticleV1Checkpoint],
    force_retrain: bool = False,
) -> tuple[ArticleV1Checkpoint, bool]:
    """Resume a compatible checkpoint byte-for-byte or train one missing file."""

    checkpoint_path = Path(path)
    if checkpoint_path.exists() and not force_retrain:
        try:
            checkpoint = ArticleV1Checkpoint.load(checkpoint_path)
            if checkpoint.training_seed != int(expected_training_seed):
                raise ValueError(
                    "checkpoint seed does not match the seed encoded by its path"
                )
            checkpoint.validate_against_scope(scope, require_nonzero=True)
        except (
            AttributeError,
            KeyError,
            OSError,
            OverflowError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"existing checkpoint {checkpoint_path} is corrupt or incompatible; "
                "use a new run ID or explicitly force retraining"
            ) from error
        return checkpoint, False

    checkpoint = train_callback()
    if checkpoint.training_seed != int(expected_training_seed):
        raise ValueError("trainer returned a checkpoint with the wrong training seed")
    checkpoint.validate_against_scope(scope, require_nonzero=True)
    checkpoint.save(checkpoint_path)
    return checkpoint, True


def _feature_provider(
    case: ArticleV1TargetCase | ArticleV1EvaluationTarget,
    *,
    expansion_budget: int,
    feature_schema: str = ARTICLE_V1_FEATURE_SCHEMA_VERSION,
):
    try:
        provider_type = FEATURE_VARIANTS[feature_schema]
    except KeyError as error:
        raise ValueError(f"unsupported Article V1 feature variant {feature_schema!r}") from error
    context = ArticleTargetContext(_target(case))
    # The no-target ablation accepts a context for interface consistency but
    # deliberately omits its coordinate and never evaluates it.
    return provider_type(context, search_horizon=expansion_budget), context


def _training_provenance(
    case: ArticleV1TargetCase | ArticleV1EvaluationTarget,
    *,
    corpus_config_digest: str,
    target_fingerprint: str,
    feature_schema: str,
    feature_evaluator_schema: str,
) -> ArticleV1CheckpointProvenance:
    code = git_provenance()
    source_digest = str(code["source_worktree_digest"])
    if not source_digest.startswith("sha256:"):
        raise ValueError("portable training recovery requires git source provenance")
    return ArticleV1CheckpointProvenance(
        source_commit_sha=str(code["commit_sha"]),
        source_worktree_digest=source_digest,
        config_digest=corpus_config_digest,
        corpus_digest=corpus_config_digest,
        profile_digest=portable_digest(
            ARTICLE_V1_PROFILE.metadata(), domain="article-v1-profile-v1"
        ),
        target_id=case.target_id,
        target_fingerprint=target_fingerprint,
        feature_schema_version=feature_schema,
        feature_evaluator_schema_version=feature_evaluator_schema,
        reward_schema_version=ARTICLE_V1_PROFILE.reward_schema,
        certifier_schema_version=ARTICLE_V1_PROFILE.certification_schema,
    )


def _training_state_digests(
    environment: CircuitSynthesisEnv,
) -> tuple[str, str, str]:
    frontier = environment.frontier
    if frontier is None:
        raise RuntimeError("training frontier is not initialized")
    frontier_ids = tuple(int(value) for value in frontier.active_record_ids())
    archive_payload = [
        {
            "record_id": int(record.record_id),
            "key": _json_ready(record.key),
            "resources": list(record.resources.as_tuple()),
            "expanded": bool(record.expanded),
            "active": bool(record.active),
            "queued": bool(record.queued),
            "tombstoned": bool(record.tombstoned),
        }
        for record in frontier.archive.all_records()
    ]
    generation_payload = sorted(
        (
            {
                "key": _json_ready(key),
                "count": int(count),
            }
            for key, count in environment.generation_counts.items()
        ),
        key=lambda value: json.dumps(value["key"], sort_keys=True, default=str),
    )
    return (
        portable_digest(frontier_ids, domain="article-v1-frontier-active-ids-v1"),
        portable_digest(archive_payload, domain="article-v1-archive-state-v1"),
        portable_digest(
            generation_payload, domain="article-v1-generation-count-state-v1"
        ),
    )


def _restore_rng_state(generator: object, state: Mapping[str, object]) -> None:
    bit_generator = getattr(generator, "bit_generator", None)
    if bit_generator is None:
        if state:
            raise CheckpointCompatibilityError("checkpoint RNG has no runtime owner")
        return
    bit_generator.state = _json_ready(state)


def _rng_state_digest(state: Mapping[str, object], *, owner: str) -> str:
    return portable_digest(
        _json_ready(state), domain=f"article-v1-{owner}-rng-state-v1"
    )


def _validate_loaded_runtime_snapshot(
    loaded: LoadedRuntimeSnapshot,
    *,
    checkpoint: MidEpisodeCheckpoint,
) -> tuple[CircuitSynthesisEnv, LinearQPolicy, int]:
    """Validate a trusted cache before it can replace portable root replay."""

    state = loaded.state
    manifest = loaded.manifest
    environment = state.environment
    if not isinstance(environment, CircuitSynthesisEnv):
        raise CheckpointCompatibilityError(
            "runtime snapshot does not contain a circuit-synthesis environment"
        )
    base = int(manifest["base_expansion"])
    if environment.steps != base or state.base_expansion != base:
        raise CheckpointCompatibilityError("runtime snapshot expansion mismatch")
    provider = environment.feature_provider
    policy = environment.policy
    if (
        provider is None
        or str(getattr(provider, "evaluator_schema_version", ""))
        != checkpoint.provenance.feature_evaluator_schema_version
        or not isinstance(policy, LinearQPolicy)
    ):
        raise CheckpointCompatibilityError(
            "runtime snapshot feature evaluator or policy mismatch"
        )
    frontier_digest, archive_digest, generation_digest = _training_state_digests(
        environment
    )
    observed_state = {
        "frontier_active_ids_digest": frontier_digest,
        "archive_digest": archive_digest,
        "generation_count_digest": generation_digest,
        "policy_weight_digest": policy_weight_digest(policy.theta),
    }
    for name, value in observed_state.items():
        if value != manifest[name]:
            raise CheckpointCompatibilityError(
                f"runtime snapshot loaded {name} mismatch"
            )
    if _rng_state_digest(
        _json_ready(policy.rng.bit_generator.state), owner="policy"
    ) != manifest["policy_rng_state_digest"]:
        raise CheckpointCompatibilityError("runtime snapshot policy RNG mismatch")
    environment_rng = getattr(environment, "__dict__", {}).get("_np_random")
    environment_rng_state = (
        {}
        if environment_rng is None
        else _json_ready(environment_rng.bit_generator.state)
    )
    if _rng_state_digest(
        environment_rng_state, owner="environment"
    ) != manifest["environment_rng_state_digest"]:
        raise CheckpointCompatibilityError("runtime snapshot environment RNG mismatch")
    pending_id = manifest["pending_record_id"]
    pending_node = next(
        (
            node
            for node in environment.current_nodes()
            if node.record_id == pending_id
        ),
        None,
    )
    if pending_node is None:
        raise CheckpointCompatibilityError(
            "runtime snapshot pending record is not open"
        )
    trainer = Trainer(environment, policy=policy)
    pending_batch = trainer._article_decision_batch()
    pending_digest = feature_row_digest(
        trainer._frozen_features(pending_batch, pending_node)
    )
    if pending_digest != manifest["pending_feature_digest"]:
        raise CheckpointCompatibilityError(
            "runtime snapshot pending feature mismatch"
        )
    return environment, policy, base


def _replay_training_checkpoint(
    checkpoint: MidEpisodeCheckpoint,
    *,
    environment: CircuitSynthesisEnv,
    policy: LinearQPolicy,
    trainer: Trainer,
    base_expansion: int = 0,
) -> TrainerEpisodeResume:
    """Rebuild one real Article V1 episode without policy ranking/RNG draws."""

    if (
        isinstance(base_expansion, bool)
        or not isinstance(base_expansion, int)
        or base_expansion < 0
        or base_expansion > checkpoint.expansion_count
    ):
        raise CheckpointCompatibilityError("runtime replay base expansion is invalid")
    recorded_td_errors = tuple(
        float(value)
        for value in checkpoint.training_aggregates.get("td_errors", ())
    )
    if len(recorded_td_errors) != checkpoint.expansion_count:
        raise CheckpointCompatibilityError(
            "checkpoint TD-error history length mismatch"
        )
    if base_expansion == 0:
        environment.reset(seed=getattr(environment.config, "seed", None))
        policy.theta[:] = np.asarray(
            checkpoint.episode_initial_theta, dtype=np.float64
        )
    elif environment.steps != base_expansion:
        raise CheckpointCompatibilityError(
            "runtime snapshot environment is at the wrong expansion"
        )
    replay_td_errors: list[float] = list(recorded_td_errors[:base_expansion])
    entry_index = base_expansion
    last_verified_state_digests: tuple[str, str, str] | None = None
    if base_expansion:
        base_entry = checkpoint.journal.entries[base_expansion - 1]
        if (
            not base_entry.state_digest_verified
            or base_entry.frontier_active_ids_digest is None
            or base_entry.archive_digest is None
            or base_entry.generation_count_digest is None
        ):
            raise CheckpointCompatibilityError(
                "runtime replay base is not a verified full-state boundary"
            )
        last_verified_state_digests = (
            base_entry.frontier_active_ids_digest,
            base_entry.archive_digest,
            base_entry.generation_count_digest,
        )

    def replay_step(selected_record_id: int) -> ReplayObservation:
        nonlocal entry_index, last_verified_state_digests
        recorded = checkpoint.journal.entries[entry_index]
        nodes_before = environment.current_nodes()
        selected = next(
            (node for node in nodes_before if node.record_id == selected_record_id),
            None,
        )
        if selected is None:
            raise CheckpointCompatibilityError(
                f"replay selected record {selected_record_id} is not open"
            )
        batch = trainer._article_decision_batch()
        selected_features = trainer._frozen_features(batch, selected)
        selected_digest = feature_row_digest(selected_features)
        _, reward, terminated, truncated, info = environment.select_record(
            selected_record_id
        )
        if info.get("selected_by_fairness", False):
            raise CheckpointCompatibilityError("replay unexpectedly invoked fairness")
        next_features = None
        pending = recorded.pending_next_record_id
        if not (terminated or truncated):
            next_nodes = environment.current_nodes()
            next_node = next(
                (node for node in next_nodes if node.record_id == pending), None
            )
            if next_node is None:
                raise CheckpointCompatibilityError(
                    "recorded pending action is not open during replay"
                )
            next_batch = trainer._article_decision_batch()
            next_features = trainer._frozen_features(next_batch, next_node)
        td_error = policy.update_from_features(
            current_features=selected_features,
            reward=float(reward),
            next_features=next_features,
            done=bool(terminated or truncated),
        )
        replay_td_errors.append(float(td_error))
        if recorded.state_digest_verified:
            frontier_digest, archive_digest, generation_digest = (
                _training_state_digests(environment)
            )
            last_verified_state_digests = (
                frontier_digest,
                archive_digest,
                generation_digest,
            )
        else:
            frontier_digest = archive_digest = generation_digest = None
        entry_index += 1
        return ReplayObservation(
            expansion_index=int(environment.steps),
            selected_record_id=selected_record_id,
            selected_feature_digest=selected_digest,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            frontier_revision=int(environment.frontier.revision),
            state_digest_verified=recorded.state_digest_verified,
            frontier_active_ids_digest=frontier_digest,
            archive_digest=archive_digest,
            generation_count_digest=generation_digest,
            policy_weight_digest_after_update=policy_weight_digest(policy.theta),
            pending_next_record_id=pending,
        )

    replay_journal = ArticleV1EventJournal(
        checkpoint.journal.entries[base_expansion:],
        base_expansion=base_expansion,
    )
    replay_and_validate(replay_journal, replay_step)
    if tuple(replay_td_errors) != recorded_td_errors:
        raise CheckpointCompatibilityError("replayed TD-error history mismatch")
    if last_verified_state_digests is None:
        raise CheckpointCompatibilityError(
            "replay journal has no verified final full-state boundary"
        )
    frontier_digest, archive_digest, generation_digest = last_verified_state_digests
    pending_node = next(
        (
            node
            for node in environment.current_nodes()
            if node.record_id == checkpoint.pending_next_record_id
        ),
        None,
    )
    if pending_node is None:
        raise CheckpointCompatibilityError("pending record is not open after replay")
    pending_batch = trainer._article_decision_batch()
    pending_features = trainer._frozen_features(pending_batch, pending_node)
    validate_pending_resume_state(
        checkpoint,
        active_record_ids=environment.frontier.active_record_ids(),
        recomputed_pending_feature_row=pending_features,
        frontier_revision=int(environment.frontier.revision),
        frontier_active_ids_digest=frontier_digest,
        archive_digest=archive_digest,
        generation_count_digest=generation_digest,
    )
    if policy_weight_digest(policy.theta) != checkpoint.weight_digest:
        raise CheckpointCompatibilityError("replayed final policy weights mismatch")
    policy.theta[:] = np.asarray(checkpoint.theta, dtype=np.float64)
    trainer.epsilon = float(checkpoint.epsilon)
    _restore_rng_state(policy.rng, checkpoint.policy_rng_state)
    environment_rng = getattr(environment, "__dict__", {}).get("_np_random")
    _restore_rng_state(environment_rng, checkpoint.environment_rng_state)
    return TrainerEpisodeResume(
        episode_index=checkpoint.episode_index,
        expansion=checkpoint.expansion_count,
        selected_record_id=checkpoint.pending_next_record_id,
        selected_features=checkpoint.pending_next_feature_row,
        total_reward=checkpoint.total_reward,
        td_errors=tuple(float(value) for value in replay_td_errors),
    )


def train_article_v1_checkpoint(
    cases: Sequence[ArticleV1TargetCase | ArticleV1EvaluationTarget],
    *,
    corpus_config_digest: str,
    training_seed: int,
    episodes_per_target: int,
    learning_rate: float,
    epsilon_start: float,
    epsilon_minimum: float,
    epsilon_decay: float,
    beta: float,
    feature_schema: str = ARTICLE_V1_FEATURE_SCHEMA_VERSION,
    checkpoint_family: str = STANDARD_CHECKPOINT_FAMILY,
    training_scope_mode: str = COMPLETE_TRAINING_SCOPE,
    expansion_cap: int | None = None,
    certification_tolerance: float = 1e-9,
    training_checkpoint_dir: str | Path | None = None,
    progress_reporter: ArticleV1ProgressReporter | None = None,
    checkpoint_cadence: CheckpointCadence | None = None,
    runtime_snapshot_every_expansions: int | None = (
        DEFAULT_RUNTIME_SNAPSHOT_INTERVAL
    ),
    resume_training: bool = True,
    run_id: str = "article-v1-training",
    interrupt_after_expansions: int | None = None,
) -> ArticleV1Checkpoint:
    """Train transferable weights without exposing generator witnesses."""

    if not cases:
        raise ValueError("training requires at least one target")
    if checkpoint_family not in CHECKPOINT_FAMILIES:
        raise ValueError(f"checkpoint family must be one of {CHECKPOINT_FAMILIES!r}")
    if training_scope_mode not in TRAINING_SCOPE_MODES:
        raise ValueError(
            f"training scope mode must be one of {TRAINING_SCOPE_MODES!r}"
        )
    if not corpus_config_digest:
        raise ValueError("training requires a corpus config digest")
    if any(case.split != "train" for case in cases):
        raise ValueError("Article V1 checkpoints may be trained only on the train split")
    training_target_ids = tuple(case.target_id for case in cases)
    if len(set(training_target_ids)) != len(training_target_ids):
        raise ValueError("training target IDs must be unique")
    if (
        isinstance(episodes_per_target, bool)
        or not isinstance(episodes_per_target, int)
        or episodes_per_target < 1
    ):
        raise ValueError("training episodes per target must be a positive integer")
    if not np.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("training beta must be finite and non-negative")
    if (
        not np.isfinite(float(certification_tolerance))
        or float(certification_tolerance) <= 0.0
    ):
        raise ValueError("training certification tolerance must be positive")
    if expansion_cap is not None and (
        isinstance(expansion_cap, bool)
        or not isinstance(expansion_cap, int)
        or expansion_cap < 1
    ):
        raise ValueError("training expansion cap must be a positive integer or None")
    if runtime_snapshot_every_expansions is not None and (
        isinstance(runtime_snapshot_every_expansions, bool)
        or not isinstance(runtime_snapshot_every_expansions, int)
        or runtime_snapshot_every_expansions < 1
    ):
        raise ValueError(
            "runtime snapshot interval must be a positive integer or None"
        )
    if interrupt_after_expansions is not None and (
        isinstance(interrupt_after_expansions, bool)
        or not isinstance(interrupt_after_expansions, int)
        or interrupt_after_expansions < 1
    ):
        raise ValueError("interrupt_after_expansions must be a positive integer")
    if interrupt_after_expansions is not None and training_checkpoint_dir is None:
        raise ValueError(
            "interrupt_after_expansions requires a training checkpoint directory"
        )
    cases = tuple(cases)
    effective_training_expansion_budgets = [
        (
            case.target_id,
            min(int(case.budget.expansion_budget), int(expansion_cap))
            if expansion_cap is not None
            else int(case.budget.expansion_budget),
        )
        for case in cases
    ]
    budget_mapping = dict(effective_training_expansion_budgets)
    evaluator_schema = str(FEATURE_VARIANTS[feature_schema].evaluator_schema_version)

    def provenance_for(case_index: int) -> ArticleV1CheckpointProvenance:
        target_context = ArticleTargetContext(_target(cases[case_index]))
        return _training_provenance(
            cases[case_index],
            corpus_config_digest=corpus_config_digest,
            target_fingerprint=target_context.fingerprint,
            feature_schema=feature_schema,
            feature_evaluator_schema=evaluator_schema,
        )

    store = (
        None
        if training_checkpoint_dir is None
        else ArticleV1TrainingCheckpointStore(training_checkpoint_dir)
    )
    runtime_snapshot_store = (
        None
        if store is None or runtime_snapshot_every_expansions is None
        else ArticleV1RuntimeSnapshotStore(training_checkpoint_dir)
    )
    loaded_recovery: MidEpisodeCheckpoint | TrainingProgressCheckpoint | None = None
    if store is not None and resume_training and (
        store.checkpoint_path("latest").exists()
        or store.manifest_path("latest").exists()
    ):
        loaded_recovery = store.load_latest_or_previous()

    weights: np.ndarray | None = None
    histories: list[Mapping[str, object]] = []
    start_target = 0
    partial_episode_results: list[Mapping[str, object]] = []
    if isinstance(loaded_recovery, TrainingProgressCheckpoint):
        if loaded_recovery.training_seed != int(training_seed):
            raise CheckpointCompatibilityError("training-progress seed mismatch")
        if loaded_recovery.target_count != len(cases):
            raise CheckpointCompatibilityError("training-progress target count mismatch")
        if loaded_recovery.episodes_per_target != int(episodes_per_target):
            raise CheckpointCompatibilityError("training-progress episode count mismatch")
        if dict(loaded_recovery.effective_budgets) != budget_mapping:
            raise CheckpointCompatibilityError("training-progress budget mismatch")
        expected_completed = training_target_ids[: loaded_recovery.target_cursor]
        if loaded_recovery.completed_target_ids != expected_completed:
            raise CheckpointCompatibilityError("training-progress target cursor mismatch")
        start_target = loaded_recovery.target_cursor
        serialized_history = list(_json_ready(loaded_recovery.training_history))
        histories = [dict(value) for value in serialized_history[:start_target]]
        if len(serialized_history) > start_target:
            incomplete = dict(serialized_history[start_target])
            if incomplete.get("target_id") != cases[start_target].target_id:
                raise CheckpointCompatibilityError(
                    "training-progress partial target mismatch"
                )
            partial_episode_results = [
                dict(value) for value in incomplete.get("episodes", ())
            ]
        weights = np.asarray(loaded_recovery.theta, dtype=np.float64).copy()
        if start_target < len(cases):
            expected = provenance_for(start_target)
            if loaded_recovery.provenance != expected:
                raise CheckpointCompatibilityError(
                    "training-progress provenance mismatch"
                )
    elif isinstance(loaded_recovery, MidEpisodeCheckpoint):
        matching = [
            index
            for index, case in enumerate(cases)
            if case.target_id == loaded_recovery.provenance.target_id
        ]
        if len(matching) != 1:
            raise CheckpointCompatibilityError("recovery target is outside training scope")
        start_target = matching[0]
        aggregates = _json_ready(loaded_recovery.training_aggregates)
        histories = [dict(value) for value in aggregates.get("completed_histories", ())]
        partial_episode_results = [
            dict(value) for value in aggregates.get("completed_episode_results", ())
        ]
        if tuple(record["target_id"] for record in histories) != training_target_ids[
            :start_target
        ]:
            raise CheckpointCompatibilityError("mid-episode completed history mismatch")
        weights = np.asarray(loaded_recovery.theta, dtype=np.float64).copy()

    if start_target > len(cases):
        raise CheckpointCompatibilityError("training recovery cursor exceeds corpus")

    provider = None
    for index in range(start_target, len(cases)):
        case = cases[index]
        maximum_steps = budget_mapping[case.target_id]
        provider, context = _feature_provider(
            case, expansion_budget=maximum_steps, feature_schema=feature_schema
        )
        policy = LinearQPolicy(
            feature_provider=provider,
            lr=learning_rate,
            gamma=1.0,
            seed=training_seed + index,
        )
        if weights is not None:
            if weights.shape != policy.theta.shape:
                raise ValueError("feature schema changed during transferable training")
            policy.theta[:] = weights
        target = _target(case)
        environment = CircuitSynthesisEnv(
            Config(
                num_qubits=case.num_qubits,
                budget=case.budget.resource_budget(),
                max_steps=maximum_steps,
                max_frontier=64,
                discount=1.0,
                seed=training_seed + index,
                fairness_interval=0,
                reward_mode="article_v1_expansion_potential",
                article_v1_beta=beta,
            ),
            ArticleV1CertificationEngine(
                target, tau_cert=float(certification_tolerance)
            ),
            feature_provider=provider,
            target_metric=context,
            observation_features=False,
        )
        trainer = Trainer(environment, policy=policy)
        trainer.epsilon = float(epsilon_start)
        trainer.min_epsilon = float(epsilon_minimum)
        trainer.epsilon_decay = float(epsilon_decay)
        current_recovery = loaded_recovery if index == start_target else None
        start_episode = 0
        resume_episode_state = None
        if isinstance(current_recovery, TrainingProgressCheckpoint):
            start_episode = current_recovery.episode_cursor
            trainer.epsilon = float(current_recovery.epsilon)
            _restore_rng_state(policy.rng, current_recovery.policy_rng_state)
        journal = ArticleV1EventJournal()
        episode_initial_theta = tuple(float(value) for value in policy.theta)
        td_errors: list[float] = []
        loaded_runtime_snapshot: LoadedRuntimeSnapshot | None = None
        runtime_snapshot_base_expansion = 0
        runtime_snapshot_restore_started_ns = 0
        runtime_snapshot_restore_time_ns = 0
        recovery_replay_time_ns = 0
        if isinstance(current_recovery, MidEpisodeCheckpoint):
            validate_expected = ResumeExpectation(
                provenance=provenance_for(index),
                training_seed=int(training_seed),
                episode_index=current_recovery.episode_index,
                episode_count=int(episodes_per_target),
                expansion_cap=maximum_steps,
                feature_dimension=len(policy.theta),
            )
            validate_resume_compatibility(current_recovery, validate_expected)
            episode_initial_theta = current_recovery.episode_initial_theta
            journal = ArticleV1EventJournal(current_recovery.journal.entries)
            td_errors = [
                float(value)
                for value in current_recovery.training_aggregates.get("td_errors", ())
            ]
            if runtime_snapshot_store is not None:
                runtime_snapshot_restore_started_ns = time.perf_counter_ns()
                loaded_runtime_snapshot = runtime_snapshot_store.load_compatible(
                    current_recovery
                )
                runtime_snapshot_restore_time_ns += (
                    time.perf_counter_ns() - runtime_snapshot_restore_started_ns
                )
            if loaded_runtime_snapshot is not None:
                runtime_snapshot_restore_started_ns = time.perf_counter_ns()
                environment, policy, runtime_snapshot_base_expansion = (
                    _validate_loaded_runtime_snapshot(
                        loaded_runtime_snapshot,
                        checkpoint=current_recovery,
                    )
                )
                provider = environment.feature_provider
                context = environment.article_target_metric
                if provider is None or context is None:
                    raise CheckpointCompatibilityError(
                        "runtime snapshot lost its Article V1 provider/context"
                    )
                trainer = Trainer(environment, policy=policy)
                trainer.visit_counts = defaultdict(
                    int, loaded_runtime_snapshot.state.visit_counts
                )
                runtime_snapshot_restore_time_ns += (
                    time.perf_counter_ns() - runtime_snapshot_restore_started_ns
                )
            replay_started_ns = time.perf_counter_ns()
            resume_episode_state = _replay_training_checkpoint(
                current_recovery,
                environment=environment,
                policy=policy,
                trainer=trainer,
                base_expansion=runtime_snapshot_base_expansion,
            )
            recovery_replay_time_ns = time.perf_counter_ns() - replay_started_ns
            start_episode = current_recovery.episode_index

        checkpoint_gate = CheckpointCadenceGate(
            checkpoint_cadence,
            initial_expansion=(
                0
                if resume_episode_state is None
                else int(resume_episode_state.expansion)
            ),
        )
        target_started = time.perf_counter()
        episode_progress_clock = _EpisodeProgressClock(started=target_started)
        initial_provider_metrics = dict(
            getattr(provider, "instrumentation", lambda: {})()
        )
        feature_timing_window = _ProgressFeatureTimingWindow(
            initial_count=int(
                initial_provider_metrics.get("compact_batch_count", 0)
            )
        )
        progress_start_ns = (
            0
            if progress_reporter is None
            else progress_reporter.progress_reporting_time_ns
        )
        checkpoint_start_ns = 0 if store is None else store.checkpoint_io_time_ns
        runtime_snapshot_start_ns = (
            0
            if runtime_snapshot_store is None
            else runtime_snapshot_store.snapshot_io_time_ns
        )
        runtime_snapshot_start_count = (
            0
            if runtime_snapshot_store is None
            else runtime_snapshot_store.snapshot_write_count
        )
        runtime_snapshot_start_bytes = (
            0
            if runtime_snapshot_store is None
            else runtime_snapshot_store.snapshot_bytes_written
        )
        checkpoint_state_digest_time_ns = 0
        last_checkpoint_path: str | None = None
        last_checkpoint_expansion: int | None = None
        last_safe_event: TrainerBoundaryEvent | None = None
        if progress_reporter is not None:
            progress_reporter.reset_cadence(expansion=0)

        def timed_training_state_digests() -> tuple[str, str, str]:
            nonlocal checkpoint_state_digest_time_ns
            started_ns = time.perf_counter_ns()
            try:
                return _training_state_digests(environment)
            finally:
                checkpoint_state_digest_time_ns += (
                    time.perf_counter_ns() - started_ns
                )

        def save_mid(
            event: TrainerBoundaryEvent,
            *,
            state_digests: tuple[str, str, str] | None = None,
        ) -> bool:
            nonlocal last_checkpoint_path, last_checkpoint_expansion
            if store is None or event.next_record_id is None or event.next_features is None:
                return False
            if journal.expansion_count != event.expansion or not journal.entries:
                return False
            if state_digests is None:
                state_digests = timed_training_state_digests()
            frontier_digest, archive_digest, generation_digest = state_digests
            journal.bind_latest_state_digests(
                frontier_active_ids_digest=frontier_digest,
                archive_digest=archive_digest,
                generation_count_digest=generation_digest,
            )
            checkpoint = MidEpisodeCheckpoint(
                provenance=provenance_for(index),
                training_seed=int(training_seed),
                episode_index=event.episode_index,
                episode_count=int(episodes_per_target),
                expansion_count=event.expansion,
                expansion_cap=maximum_steps,
                journal=journal,
                episode_initial_theta=episode_initial_theta,
                theta=event.policy_weights_after_update,
                epsilon=event.epsilon,
                policy_rng_state=event.policy_rng_state,
                environment_rng_state=event.environment_rng_state,
                pending_next_record_id=event.next_record_id,
                pending_next_feature_row=event.next_features,
                total_reward=event.total_reward,
                training_aggregates={
                    "td_errors": list(td_errors),
                    "completed_histories": histories,
                    "completed_episode_results": partial_episode_results,
                },
                search_metrics={
                    name: value
                    for name, value in event.search_metrics.items()
                    if not name.endswith("_time_ns")
                },
                frontier_revision=event.frontier_revision,
                frontier_active_ids_digest=frontier_digest,
                archive_digest=archive_digest,
                generation_count_digest=generation_digest,
            )
            last_checkpoint_path = str(store.save_latest(checkpoint).path)
            last_checkpoint_expansion = int(event.expansion)
            if (
                runtime_snapshot_store is not None
                and runtime_snapshot_every_expansions is not None
                and event.expansion % runtime_snapshot_every_expansions == 0
            ):
                environment_rng = getattr(environment, "__dict__", {}).get(
                    "_np_random"
                )
                environment_rng_state = (
                    {}
                    if environment_rng is None
                    else _json_ready(environment_rng.bit_generator.state)
                )
                runtime_snapshot_store.save_latest(
                    checkpoint,
                    ArticleV1RuntimeState(
                        base_expansion=event.expansion,
                        environment=environment,
                        visit_counts=dict(trainer.visit_counts),
                    ),
                    pending_feature_digest=feature_row_digest(event.next_features),
                    policy_rng_state_digest=_rng_state_digest(
                        event.policy_rng_state, owner="policy"
                    ),
                    environment_rng_state_digest=_rng_state_digest(
                        environment_rng_state, owner="environment"
                    ),
                )
            return True

        def checkpoint_callback(event: TrainerBoundaryEvent) -> None:
            nonlocal journal, episode_initial_theta, td_errors
            nonlocal partial_episode_results, last_safe_event, last_checkpoint_path
            last_safe_event = event
            if event.boundary == "expansion":
                assert event.selected_record_id is not None
                assert event.selected_features is not None
                assert event.reward is not None and event.td_error is not None
                nonterminal = not (event.terminated or event.truncated)
                cadence_due = bool(
                    store is not None
                    and nonterminal
                    and checkpoint_gate.due(event.expansion)
                )
                snapshot_due = bool(
                    runtime_snapshot_store is not None
                    and runtime_snapshot_every_expansions is not None
                    and nonterminal
                    and event.expansion % runtime_snapshot_every_expansions == 0
                )
                interrupt_due = bool(
                    interrupt_after_expansions is not None
                    and event.expansion == int(interrupt_after_expansions)
                    and nonterminal
                )
                state_digest_verified = cadence_due or interrupt_due or snapshot_due
                state_digests = (
                    timed_training_state_digests()
                    if state_digest_verified
                    else (None, None, None)
                )
                journal.append(
                    ArticleV1JournalEntry(
                        expansion_index=event.expansion,
                        selected_record_id=event.selected_record_id,
                        selected_feature_digest=feature_row_digest(
                            event.selected_features
                        ),
                        reward=event.reward,
                        terminated=event.terminated,
                        truncated=event.truncated,
                        frontier_revision=event.frontier_revision,
                        state_digest_verified=state_digest_verified,
                        frontier_active_ids_digest=state_digests[0],
                        archive_digest=state_digests[1],
                        generation_count_digest=state_digests[2],
                        policy_weight_digest_after_update=policy_weight_digest(
                            event.policy_weights_after_update
                        ),
                        pending_next_record_id=event.next_record_id,
                    )
                )
                td_errors.append(event.td_error)
                if cadence_due or snapshot_due:
                    assert all(value is not None for value in state_digests)
                    if not save_mid(
                        event, state_digests=state_digests  # type: ignore[arg-type]
                    ):
                        raise RuntimeError(
                            "checkpoint cadence reached without a safe pending state"
                        )
                    checkpoint_gate.mark_saved(event.expansion)
                if interrupt_due:
                    if not (cadence_due or snapshot_due):
                        assert all(value is not None for value in state_digests)
                        if not save_mid(
                            event,
                            state_digests=state_digests,  # type: ignore[arg-type]
                        ):
                            raise RuntimeError(
                                "controlled interrupt reached without a valid "
                                "mid-episode checkpoint"
                            )
                    raise KeyboardInterrupt
                return

            episode_result = dict(_json_ready(event.episode_result))
            partial_episode_results.append(episode_result)
            target_complete = event.episode_index + 1 == int(episodes_per_target)
            target_cursor = index + 1 if target_complete else index
            episode_cursor = 0 if target_complete else event.episode_index + 1
            completed_ids = training_target_ids[:target_cursor]
            target_history_record = {
                "target_id": case.target_id,
                "split": case.split,
                "difficulty": case.difficulty,
                "episodes": partial_episode_results,
                "training_runtime_seconds": float(time.perf_counter() - target_started),
                "target_metric": context.cache_metrics(),
                "policy_instrumentation": policy.instrumentation(),
                "progress_reporting_time_ns": (
                    0
                    if progress_reporter is None
                    else progress_reporter.progress_reporting_time_ns
                    - progress_start_ns
                ),
                "checkpoint_callback_time_ns": int(
                    trainer.checkpoint_callback_time_ns
                ),
                "checkpoint_state_digest_time_ns": int(
                    checkpoint_state_digest_time_ns
                ),
                "checkpoint_io_time_ns": (
                    0
                    if store is None
                    else store.checkpoint_io_time_ns - checkpoint_start_ns
                ),
                "runtime_snapshot_schema_version": (
                    None
                    if runtime_snapshot_store is None
                    else ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA
                ),
                "runtime_snapshot_restore_time_ns": int(
                    runtime_snapshot_restore_time_ns
                ),
                "recovery_replay_time_ns": int(recovery_replay_time_ns),
                "runtime_snapshot_io_time_ns": (
                    0
                    if runtime_snapshot_store is None
                    else runtime_snapshot_store.snapshot_io_time_ns
                    - runtime_snapshot_start_ns
                ),
                "runtime_snapshot_write_count": (
                    0
                    if runtime_snapshot_store is None
                    else runtime_snapshot_store.snapshot_write_count
                    - runtime_snapshot_start_count
                ),
                "runtime_snapshot_bytes_written": (
                    0
                    if runtime_snapshot_store is None
                    else runtime_snapshot_store.snapshot_bytes_written
                    - runtime_snapshot_start_bytes
                ),
                "runtime_snapshot_base_expansion": int(
                    runtime_snapshot_base_expansion
                ),
                "training_incomplete": not target_complete,
            }
            serialized_histories = [*histories, target_history_record]
            next_index = min(target_cursor, len(cases) - 1)
            if store is not None:
                progress_checkpoint = TrainingProgressCheckpoint(
                    provenance=provenance_for(next_index),
                    training_seed=int(training_seed),
                    target_cursor=target_cursor,
                    target_count=len(cases),
                    episode_cursor=episode_cursor,
                    episodes_per_target=int(episodes_per_target),
                    theta=event.policy_weights_after_update,
                    epsilon=event.epsilon,
                    policy_rng_state=event.policy_rng_state,
                    training_history=tuple(serialized_histories),
                    completed_target_ids=completed_ids,
                    effective_budgets=budget_mapping,
                )
                last_checkpoint_path = str(
                    store.save_episode_final(progress_checkpoint).path
                )
            journal = ArticleV1EventJournal()
            td_errors = []
            episode_initial_theta = event.policy_weights_after_update
            checkpoint_gate.reset(expansion=0)

        def emit_progress(
            event: TrainerBoundaryEvent, *, force: bool = False
        ) -> None:
            if progress_reporter is None:
                return
            frontier = environment.frontier
            assert frontier is not None
            archive = frontier.archive
            provider_metrics = dict(getattr(provider, "instrumentation", lambda: {})())
            recent_timings = getattr(
                provider, "recent_compact_batch_times_ns", lambda: ()
            )()
            last_feature_seconds, rolling_feature_seconds = (
                feature_timing_window.observe(
                    provider_metrics,
                    recent_batch_times_ns=recent_timings,
                )
            )
            elapsed, expansion_rate = episode_progress_clock.measure(event.expansion)
            progress_reporter.maybe_emit(
                ArticleV1ProgressEvent(
                    timestamp_utc=utc_timestamp(),
                    run_id=run_id,
                    phase=f"training:{checkpoint_family}",
                    feature_evaluator_schema_version=evaluator_schema,
                    training_seed=int(training_seed),
                    target_index=index,
                    target_count=len(cases),
                    target_id=case.target_id,
                    split=case.split,
                    stratum=case.difficulty,
                    num_qubits=case.num_qubits,
                    episode_index=event.episode_index,
                    episode_count=int(episodes_per_target),
                    expansion=event.expansion,
                    expansion_cap=maximum_steps,
                    frontier_size=len(event.frontier_active_record_ids),
                    frontier_peak=max(
                        len(event.frontier_active_record_ids),
                        int(event.search_metrics.get("frontier_peak", 0)),
                    ),
                    archive_records=int(archive.archive_record_count),
                    active_archive_records=int(archive.active_record_count),
                    unique_resource_groups=int(
                        provider_metrics.get("unique_resource_group_count", 0)
                    ),
                    last_feature_batch_seconds=last_feature_seconds,
                    rolling_feature_batch_seconds=rolling_feature_seconds,
                    elapsed_seconds=elapsed,
                    expansions_per_second=expansion_rate,
                    checkpoint_path=last_checkpoint_path,
                ),
                force=force,
            )
            if event.boundary == "episode_end":
                progress_reporter.reset_cadence(expansion=0)
                feature_timing_window.reset_episode()
                episode_progress_clock.reset()

        def progress_callback(event: TrainerBoundaryEvent) -> None:
            emit_progress(event, force=event.boundary == "episode_end")

        recovery_callbacks_enabled = bool(
            store is not None or interrupt_after_expansions is not None
        )
        trainer.checkpoint_callback = (
            checkpoint_callback if recovery_callbacks_enabled else None
        )
        trainer.progress_callback = progress_callback if progress_reporter else None
        try:
            with redirect_stdout(StringIO()):
                trainer.train(
                    int(episodes_per_target),
                    start_episode=start_episode,
                    resume_episode=resume_episode_state,
                )
        except KeyboardInterrupt:
            if (
                store is not None
                and last_safe_event is not None
                and not (last_safe_event.terminated or last_safe_event.truncated)
                and last_checkpoint_expansion != last_safe_event.expansion
            ):
                save_mid(last_safe_event)
            if progress_reporter is not None and last_safe_event is not None:
                emit_progress(last_safe_event, force=True)
            raise

        target_history = list(partial_episode_results)
        history_record = {
            "target_id": case.target_id,
            "split": case.split,
            "difficulty": case.difficulty,
            "episodes": target_history,
            "training_runtime_seconds": float(trainer.last_training_runtime_seconds),
            "target_metric": context.cache_metrics(),
            "policy_instrumentation": policy.instrumentation(),
            "progress_reporting_time_ns": (
                0
                if progress_reporter is None
                else progress_reporter.progress_reporting_time_ns - progress_start_ns
            ),
            "checkpoint_callback_time_ns": int(
                trainer.checkpoint_callback_time_ns
            ),
            "checkpoint_state_digest_time_ns": int(
                checkpoint_state_digest_time_ns
            ),
            "checkpoint_io_time_ns": (
                0 if store is None else store.checkpoint_io_time_ns - checkpoint_start_ns
            ),
            "runtime_snapshot_schema_version": (
                None
                if runtime_snapshot_store is None
                else ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA
            ),
            "runtime_snapshot_restore_time_ns": int(
                runtime_snapshot_restore_time_ns
            ),
            "recovery_replay_time_ns": int(recovery_replay_time_ns),
            "runtime_snapshot_io_time_ns": (
                0
                if runtime_snapshot_store is None
                else runtime_snapshot_store.snapshot_io_time_ns
                - runtime_snapshot_start_ns
            ),
            "runtime_snapshot_write_count": (
                0
                if runtime_snapshot_store is None
                else runtime_snapshot_store.snapshot_write_count
                - runtime_snapshot_start_count
            ),
            "runtime_snapshot_bytes_written": (
                0
                if runtime_snapshot_store is None
                else runtime_snapshot_store.snapshot_bytes_written
                - runtime_snapshot_start_bytes
            ),
            "runtime_snapshot_base_expansion": int(
                runtime_snapshot_base_expansion
            ),
        }
        histories.append(history_record)
        weights = np.array(policy.theta, dtype=np.float64, copy=True)
        partial_episode_results = []
        loaded_recovery = None

        if store is not None:
            target_cursor = index + 1
            next_index = min(target_cursor, len(cases) - 1)
            final_progress = TrainingProgressCheckpoint(
                provenance=provenance_for(next_index),
                training_seed=int(training_seed),
                target_cursor=target_cursor,
                target_count=len(cases),
                episode_cursor=0,
                episodes_per_target=int(episodes_per_target),
                theta=tuple(float(value) for value in weights),
                epsilon=float(trainer.epsilon),
                policy_rng_state=getattr(last_safe_event, "policy_rng_state", {}),
                training_history=tuple(histories),
                completed_target_ids=training_target_ids[:target_cursor],
                effective_budgets=budget_mapping,
            )
            store.save_latest(final_progress)

    if weights is None:
        if not isinstance(loaded_recovery, TrainingProgressCheckpoint):
            raise RuntimeError("training produced no recoverable weights")
        weights = np.asarray(loaded_recovery.theta, dtype=np.float64)
        provider, _ = _feature_provider(
            cases[-1],
            expansion_budget=budget_mapping[cases[-1].target_id],
            feature_schema=feature_schema,
        )
        histories = [dict(value) for value in _json_ready(loaded_recovery.training_history)]
    assert provider is not None
    provider_names = tuple(provider.names)
    return ArticleV1Checkpoint(
        training_seed=int(training_seed),
        weights=tuple(float(value) for value in weights),
        feature_schema_version=str(provider.schema_version),
        feature_evaluator_schema_version=str(
            getattr(
                provider,
                "evaluator_schema_version",
                ARTICLE_V1_PROFILE.feature_evaluator_schema,
            )
        ),
        ordered_feature_names=provider_names,
        reward_schema_version=ARTICLE_V1_PROFILE.reward_schema,
        target_metric_schema_version=ARTICLE_V1_PROFILE.target_metric_schema,
        certification_schema_version=ARTICLE_V1_PROFILE.certification_schema,
        learning_rate=float(learning_rate),
        discount=1.0,
        epsilon_schedule=(
            ("decay", float(epsilon_decay)),
            ("minimum", float(epsilon_minimum)),
            ("start", float(epsilon_start)),
        ),
        checkpoint_family=checkpoint_family,
        training_scope_mode=training_scope_mode,
        training_beta=float(beta),
        training_certification_tolerance=float(certification_tolerance),
        training_episodes_per_target=int(episodes_per_target),
        training_expansion_cap=(
            None if expansion_cap is None else int(expansion_cap)
        ),
        training_budget_policy=ARTICLE_V1_TRAINING_BUDGET_POLICY,
        effective_training_expansion_budgets=tuple(
            effective_training_expansion_budgets
        ),
        training_target_ids=training_target_ids,
        training_histories=tuple(histories),
        corpus_config_digest=corpus_config_digest,
    )


def _fixed_pilot_replay_workload() -> dict[str, object]:
    """Resolve the one preregistered pilot hard-target replay workload."""

    from experiments.article_v1_feature_benchmark import (
        DEFAULT_HARD_EXPANSION_CAP,
        PILOT_HARD_3Q_TARGET_ID,
    )

    config = load_article_v1_config("pilot")
    corpus = build_article_v1_corpus(config)
    training_cases = corpus.evaluation_targets(split="train")
    matches = [
        (case_index, case)
        for case_index, case in enumerate(training_cases)
        if case.target_id == PILOT_HARD_3Q_TARGET_ID
    ]
    if len(matches) != 1:
        raise ValueError(
            "frozen pilot replay target must identify exactly one train target"
        )
    case_index, case = matches[0]
    if case.difficulty != "hard" or case.num_qubits != 3:
        raise ValueError("frozen pilot replay target must be hard and three-qubit")
    horizon = int(case.budget.expansion_budget)
    if horizon != int(DEFAULT_HARD_EXPANSION_CAP):
        raise ValueError("frozen pilot replay target must retain the 8192 horizon")
    experiment = dict(config.experiment)
    feature_schema = str(experiment["feature_schema"])
    if feature_schema != ARTICLE_V1_FEATURE_SCHEMA_VERSION:
        raise ValueError("frozen pilot replay requires the production feature schema")
    seeds = experiment.get("training_seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds:
        raise ValueError("frozen pilot config has no training seed")
    effective_seed = int(seeds[0]) + int(case_index)
    provider, context = _feature_provider(
        case,
        expansion_budget=horizon,
        feature_schema=feature_schema,
    )
    return {
        "config": config,
        "case": case,
        "case_index": int(case_index),
        "experiment": experiment,
        "feature_schema": feature_schema,
        "feature_evaluator_schema": str(provider.evaluator_schema_version),
        "feature_dimension": int(provider.dimension),
        "target_fingerprint": str(context.fingerprint),
        "effective_seed": effective_seed,
        "horizon": horizon,
    }


def _require_unchanged_source(
    before: Mapping[str, object], after: Mapping[str, object]
) -> None:
    for name in ("commit_sha", "source_worktree_digest", "dirty_worktree"):
        if before.get(name) != after.get(name):
            raise RuntimeError(
                f"source provenance changed during replay operation ({name})"
            )


def _validate_fixed_pilot_replay_checkpoint(
    checkpoint: object,
    *,
    workload: Mapping[str, object],
    source: Mapping[str, object],
) -> MidEpisodeCheckpoint:
    if not isinstance(checkpoint, MidEpisodeCheckpoint):
        raise CheckpointCompatibilityError(
            "fixed replay capture must be an internal mid-episode checkpoint"
        )
    config = workload["config"]
    case = workload["case"]
    assert isinstance(config, ArticleV1CorpusConfig)
    assert isinstance(case, ArticleV1EvaluationTarget)
    expected_provenance = _training_provenance(
        case,
        corpus_config_digest=config.digest,
        target_fingerprint=str(workload["target_fingerprint"]),
        feature_schema=str(workload["feature_schema"]),
        feature_evaluator_schema=str(workload["feature_evaluator_schema"]),
    )
    if checkpoint.provenance != expected_provenance:
        raise CheckpointCompatibilityError(
            "fixed replay checkpoint provenance does not match the canonical workload"
        )
    if (
        checkpoint.provenance.source_commit_sha != source.get("commit_sha")
        or checkpoint.provenance.source_worktree_digest
        != source.get("source_worktree_digest")
    ):
        raise CheckpointCompatibilityError(
            "fixed replay checkpoint does not bind the current source snapshot"
        )
    validate_resume_compatibility(
        checkpoint,
        ResumeExpectation(
            provenance=expected_provenance,
            training_seed=int(workload["effective_seed"]),
            episode_index=0,
            episode_count=int(
                dict(workload["experiment"])["training_episodes_per_target"]
            ),
            expansion_cap=int(workload["horizon"]),
            feature_dimension=int(workload["feature_dimension"]),
        ),
    )
    if checkpoint.expansion_count != REPLAY_TIMING_EXPECTED_EXPANSIONS:
        raise CheckpointCompatibilityError(
            "fixed replay checkpoint must end at expansion 1024"
        )
    if checkpoint.journal.base_expansion != 0 or len(checkpoint.journal.entries) != (
        REPLAY_TIMING_EXPECTED_EXPANSIONS
    ):
        raise CheckpointCompatibilityError(
            "fixed replay checkpoint must contain the exact 1024-entry journal"
        )
    final = checkpoint.journal.entries[-1]
    if not final.state_digest_verified:
        raise CheckpointCompatibilityError(
            "fixed replay checkpoint final entry lacks full-state verification"
        )
    return checkpoint


def capture_article_v1_replay_checkpoint(
    output_directory: str | Path,
    *,
    quiet: bool = False,
    checkpoint_cadence: CheckpointCadence | None = None,
) -> dict[str, object]:
    """Capture the canonical pilot hard target at the safe expansion-1024 boundary.

    A controlled ``KeyboardInterrupt`` is the expected stop mechanism.  It is
    converted to successful capture only after the manifest, portable schema,
    provenance, exact journal length, and final full-state digests validate.
    """

    if not isinstance(quiet, bool):
        raise ValueError("quiet must be a bool")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "replay_checkpoint_capture.json"
    if status_path.exists():
        raise ValueError(
            "replay checkpoint capture status already exists; use a new run ID"
        )
    source_before = git_provenance(refresh=True)
    source_digest = str(source_before.get("source_worktree_digest"))
    if not source_digest.startswith("sha256:"):
        raise ValueError("fixed replay capture requires portable Git provenance")
    workload = _fixed_pilot_replay_workload()
    config = workload["config"]
    case = workload["case"]
    experiment = dict(workload["experiment"])
    assert isinstance(config, ArticleV1CorpusConfig)
    assert isinstance(case, ArticleV1EvaluationTarget)
    state_directory = destination / "training_state"
    store = ArticleV1TrainingCheckpointStore(state_directory)
    runtime_store = ArticleV1RuntimeSnapshotStore(state_directory)

    existing_exact = False
    try:
        existing = store.load("latest")
    except (FileNotFoundError, CheckpointFormatError):
        existing = None
    if isinstance(existing, MidEpisodeCheckpoint) and (
        existing.expansion_count == REPLAY_TIMING_EXPECTED_EXPANSIONS
    ):
        _validate_fixed_pilot_replay_checkpoint(
            existing, workload=workload, source=source_before
        )
        existing_runtime = runtime_store.load_compatible(existing)
        if (
            existing_runtime is None
            or existing_runtime.state.base_expansion
            != REPLAY_TIMING_EXPECTED_EXPANSIONS
        ):
            raise ValueError(
                "existing expansion-1024 checkpoint has no compatible compact "
                "runtime snapshot; use a new run ID"
            )
        existing_exact = True

    expected_interrupt_observed = False
    if not existing_exact:
        reporter = ArticleV1ProgressReporter(
            destination,
            cadence=ProgressCadence(every_expansions=25, every_seconds=10.0),
            quiet=quiet,
        )
        try:
            train_article_v1_checkpoint(
                (case,),
                corpus_config_digest=config.digest,
                training_seed=int(workload["effective_seed"]),
                episodes_per_target=int(experiment["training_episodes_per_target"]),
                learning_rate=float(experiment["learning_rate"]),
                epsilon_start=float(experiment["epsilon"]["start"]),
                epsilon_minimum=float(experiment["epsilon"]["minimum"]),
                epsilon_decay=float(experiment["epsilon"]["decay"]),
                beta=float(experiment["beta"]),
                feature_schema=str(workload["feature_schema"]),
                expansion_cap=int(workload["horizon"]),
                certification_tolerance=float(
                    experiment["certification_tolerance"]
                ),
                training_checkpoint_dir=state_directory,
                progress_reporter=reporter,
                checkpoint_cadence=checkpoint_cadence or CheckpointCadence(),
                resume_training=True,
                run_id=destination.name,
                interrupt_after_expansions=REPLAY_TIMING_EXPECTED_EXPANSIONS,
            )
        except KeyboardInterrupt:
            expected_interrupt_observed = True
        else:
            raise RuntimeError(
                "fixed replay workload ended without the expected expansion-1024 "
                "checkpoint interrupt"
            )

    checkpoint = _validate_fixed_pilot_replay_checkpoint(
        store.load("latest"), workload=workload, source=source_before
    )
    runtime_snapshot = runtime_store.load_compatible(checkpoint)
    if (
        runtime_snapshot is None
        or runtime_snapshot.state.base_expansion
        != REPLAY_TIMING_EXPECTED_EXPANSIONS
    ):
        raise RuntimeError(
            "validated capture did not publish an exact expansion-1024 runtime snapshot"
        )
    source_after = git_provenance(refresh=True)
    _require_unchanged_source(source_before, source_after)
    if load_article_v1_config("pilot").digest != config.digest:
        raise RuntimeError("frozen pilot config changed during checkpoint capture")
    checkpoint_path = store.checkpoint_path("latest")
    payload: dict[str, object] = {
        "capture_schema": ARTICLE_V1_REPLAY_CAPTURE_SCHEMA,
        "evidence_class": "engineering-performance-diagnostic",
        "scientific_scheduler_evidence": False,
        "source_commit_sha": source_before["commit_sha"],
        "source_worktree_digest": source_before["source_worktree_digest"],
        "source_committed_and_clean": not bool(source_before["dirty_worktree"]),
        "config_profile": config.profile,
        "config_digest": config.digest,
        "target_id": case.target_id,
        "target_fingerprint": workload["target_fingerprint"],
        "feature_evaluator_schema_version": workload[
            "feature_evaluator_schema"
        ],
        "training_seed": checkpoint.training_seed,
        "expected_interrupt_expansion": REPLAY_TIMING_EXPECTED_EXPANSIONS,
        "expected_interrupt_observed_in_this_process": expected_interrupt_observed,
        "valid_existing_boundary_reused": existing_exact,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": checkpoint_file_sha256(checkpoint_path),
        "checkpoint_schema_version": checkpoint.schema_version,
        "journal_digest": checkpoint.journal_digest,
        "journal_entry_count": len(checkpoint.journal.entries),
        "full_state_digest_entry_count": sum(
            int(entry.state_digest_verified) for entry in checkpoint.journal.entries
        ),
        "final_entry_has_full_state_digests": True,
        "runtime_snapshot_schema_version": ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA,
        "runtime_snapshot_slot": runtime_snapshot.slot,
        "runtime_snapshot_base_expansion": (
            runtime_snapshot.state.base_expansion
        ),
        "runtime_snapshot_payload_path": str(
            runtime_store.payload_path(runtime_snapshot.slot).resolve()
        ),
        "runtime_snapshot_payload_sha256": runtime_snapshot.manifest[
            "payload_sha256"
        ],
        "runtime_snapshot_payload_bytes": runtime_snapshot.manifest[
            "payload_byte_length"
        ],
        "portable_root_replay_fallback_retained": True,
        "capture_valid": True,
        "pilot_relaunch_ready": False,
        "pilot_relaunch_blocker": "validated replay timing is still required",
    }
    _atomic_json(status_path, payload)
    return payload


def _replay_fixed_pilot_checkpoint(
    checkpoint: MidEpisodeCheckpoint,
    *,
    workload: Mapping[str, object],
    runtime_snapshot_store: ArticleV1RuntimeSnapshotStore | None = None,
) -> ReplayValidationResult:
    case = workload["case"]
    experiment = dict(workload["experiment"])
    assert isinstance(case, ArticleV1EvaluationTarget)
    loaded_runtime = (
        None
        if runtime_snapshot_store is None
        else runtime_snapshot_store.load_compatible(checkpoint)
    )
    if loaded_runtime is None:
        provider, context = _feature_provider(
            case,
            expansion_budget=int(workload["horizon"]),
            feature_schema=str(workload["feature_schema"]),
        )
        policy = LinearQPolicy(
            feature_provider=provider,
            lr=float(experiment["learning_rate"]),
            gamma=1.0,
            seed=int(workload["effective_seed"]),
        )
        environment = CircuitSynthesisEnv(
            Config(
                num_qubits=case.num_qubits,
                budget=case.budget.resource_budget(),
                max_steps=int(workload["horizon"]),
                max_frontier=64,
                discount=1.0,
                seed=int(workload["effective_seed"]),
                fairness_interval=0,
                canonicalization_enabled=bool(
                    experiment["canonicalization_enabled"]
                ),
                pareto_dominance_enabled=bool(
                    experiment["pareto_dominance_enabled"]
                ),
                absorb_clifford_angles=bool(
                    experiment["absorb_clifford_angles"]
                ),
                canonicalization_mode=str(experiment["canonicalization_mode"]),
                reward_mode="article_v1_expansion_potential",
                article_v1_beta=float(experiment["beta"]),
            ),
            ArticleV1CertificationEngine(
                _target(case),
                tau_cert=float(experiment["certification_tolerance"]),
            ),
            feature_provider=provider,
            target_metric=context,
            instrumentation_enabled=True,
            observation_features=False,
        )
        base_expansion = 0
    else:
        environment, policy, base_expansion = _validate_loaded_runtime_snapshot(
            loaded_runtime, checkpoint=checkpoint
        )
        provider = environment.feature_provider
        context = environment.article_target_metric
        if provider is None or context is None:
            raise CheckpointCompatibilityError(
                "runtime snapshot lost the fixed pilot provider/context"
            )
    trainer = Trainer(environment, policy=policy)
    if loaded_runtime is not None:
        trainer.visit_counts = defaultdict(int, loaded_runtime.state.visit_counts)
    trainer.epsilon = float(experiment["epsilon"]["start"])
    trainer.min_epsilon = float(experiment["epsilon"]["minimum"])
    trainer.epsilon_decay = float(experiment["epsilon"]["decay"])
    resumed = _replay_training_checkpoint(
        checkpoint,
        environment=environment,
        policy=policy,
        trainer=trainer,
        base_expansion=base_expansion,
    )
    # `_replay_training_checkpoint` has just compared these checkpoint-bound
    # values to the freshly observed environment/policy/pending row. Reuse that
    # validated result instead of serializing the complete state a second time.
    result = ReplayValidationResult(
        measured_expansions=int(resumed.expansion),
        frontier_active_ids_digest=checkpoint.frontier_active_ids_digest,
        archive_digest=checkpoint.archive_digest,
        generation_count_digest=checkpoint.generation_count_digest,
        policy_weight_digest=policy_weight_digest(policy.theta),
        pending_feature_digest=checkpoint.pending_feature_digest,
        replay_mode=(
            "portable-root-journal"
            if loaded_runtime is None
            else "trusted-runtime-snapshot-plus-delta"
        ),
        runtime_snapshot_schema_version=(
            None
            if loaded_runtime is None
            else str(loaded_runtime.manifest["runtime_snapshot_schema"])
        ),
        runtime_snapshot_base_expansion=base_expansion,
        delta_journal_entry_count=checkpoint.expansion_count - base_expansion,
        runtime_snapshot_payload_sha256=(
            None
            if loaded_runtime is None
            else str(loaded_runtime.manifest["payload_sha256"])
        ),
        portable_replay_fallback_retained=True,
    )
    if result.policy_weight_digest != checkpoint.weight_digest:
        raise CheckpointCompatibilityError(
            "validated replay result does not equal the checkpoint policy state"
        )
    return result


def measure_article_v1_replay_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    projected_full_episode_seconds: float,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> ArticleV1ReplayTimingEvidence:
    """Validate and time replay of the exact canonical expansion-1024 checkpoint."""

    path = Path(checkpoint_path)
    evidence_path = Path(output_path)
    if path.name != "latest.json":
        raise ValueError("replay timing requires the captured latest.json slot")
    if evidence_path.exists():
        raise ValueError("replay timing output already exists; use a new path")
    source_before = git_provenance(refresh=True)
    source_digest = str(source_before.get("source_worktree_digest"))
    if not source_digest.startswith("sha256:"):
        raise ValueError("fixed replay timing requires portable Git provenance")
    workload = _fixed_pilot_replay_workload()
    config = workload["config"]
    case = workload["case"]
    assert isinstance(config, ArticleV1CorpusConfig)
    assert isinstance(case, ArticleV1EvaluationTarget)
    checkpoint = _validate_fixed_pilot_replay_checkpoint(
        ArticleV1TrainingCheckpointStore(path.parent).load("latest"),
        workload=workload,
        source=source_before,
    )
    checkpoint_digest = checkpoint_file_sha256(path)
    runtime_snapshot_store = ArticleV1RuntimeSnapshotStore(path.parent)
    evidence = measure_replay_timing(
        lambda: _replay_fixed_pilot_checkpoint(
            checkpoint,
            workload=workload,
            runtime_snapshot_store=runtime_snapshot_store,
        ),
        source_commit_sha=str(source_before["commit_sha"]),
        source_worktree_digest=str(source_before["source_worktree_digest"]),
        source_committed_and_clean=not bool(source_before["dirty_worktree"]),
        config_digest=config.digest,
        target_id=case.target_id,
        target_fingerprint=str(workload["target_fingerprint"]),
        feature_evaluator_schema_version=str(
            workload["feature_evaluator_schema"]
        ),
        checkpoint_path=path,
        checkpoint_file_sha256=checkpoint_digest,
        checkpoint_schema_version=checkpoint.schema_version,
        journal_digest=checkpoint.journal_digest,
        journal_entry_count=len(checkpoint.journal.entries),
        expected_expansions=REPLAY_TIMING_EXPECTED_EXPANSIONS,
        projected_full_episode_seconds=projected_full_episode_seconds,
        clock_ns=clock_ns,
    )
    source_after = git_provenance(refresh=True)
    _require_unchanged_source(source_before, source_after)
    if load_article_v1_config("pilot").digest != config.digest:
        raise RuntimeError("frozen pilot config changed during replay timing")
    write_replay_timing(evidence_path, evidence)
    return evidence


def _serialized_witness_gates(
    operations: object,
    *,
    num_qubits: int,
) -> tuple[Gate, ...]:
    if not isinstance(operations, (list, tuple)):
        raise ValueError("witness_operations must be a list")
    gates: list[Gate] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping) or set(operation) != {"gate", "qubits"}:
            raise ValueError(
                f"witness operation {index} must contain exactly gate and qubits"
            )
        gate_name = operation["gate"]
        if not isinstance(gate_name, str) or gate_name not in NATIVE_GATE_NAMES:
            raise ValueError(f"witness operation {index} uses a non-native gate")
        qubits_value = operation["qubits"]
        if not isinstance(qubits_value, (list, tuple)) or any(
            isinstance(qubit, bool) or not isinstance(qubit, int)
            for qubit in qubits_value
        ):
            raise ValueError(f"witness operation {index} has invalid qubits")
        qubits = tuple(int(qubit) for qubit in qubits_value)
        gate_type = GateType[gate_name]
        expected_arity = 2 if gate_type is GateType.CNOT else 1
        if (
            len(qubits) != expected_arity
            or len(set(qubits)) != len(qubits)
            or any(qubit < 0 or qubit >= num_qubits for qubit in qubits)
        ):
            raise ValueError(f"witness operation {index} has invalid gate operands")
        gates.append(Gate(gate_type, qubits))
    return tuple(gates)


def _independent_witness_certification_diagnostics(
    case: ArticleV1TargetCase | ArticleV1EvaluationTarget,
    operations: object,
    *,
    certification_tolerance: float,
) -> dict[str, object]:
    """Freshly replay one serialized witness and return certifier diagnostics."""

    gates = _serialized_witness_gates(operations, num_qubits=case.num_qubits)
    state = CircuitState(
        CircuitDAG.from_gates(case.num_qubits, gates),
        case.budget.resource_budget(),
    )
    result = ArticleV1CertificationEngine(
        _target(case), tau_cert=float(certification_tolerance)
    ).certify(state)
    if result.status is not CertStatus.SUCCESS or result.info.get("passed") is not True:
        raise ValueError("reported successful witness failed independent certification")
    return _json_ready(dict(result.info))


def _independent_witness_resource_vector(
    case: ArticleV1TargetCase | ArticleV1EvaluationTarget,
    operations: object,
) -> list[int]:
    """Replay one witness through the budget-aware authoritative state."""

    gates = _serialized_witness_gates(operations, num_qubits=case.num_qubits)
    state = CircuitState(
        CircuitDAG.from_gates(case.num_qubits, gates),
        case.budget.resource_budget(),
    )
    return [int(value) for value in state.resource_vector()]


def evaluate_article_v1_run(
    case: ArticleV1TargetCase | ArticleV1EvaluationTarget,
    *,
    scheduler: str,
    expansion_budget: int,
    evaluation_seed: int,
    checkpoint: ArticleV1Checkpoint | None = None,
    checkpoint_scope: ArticleV1CheckpointScope | None = None,
    beta: float = 1.0,
    certification_tolerance: float = 1e-9,
    canonicalization_enabled: bool = True,
    pareto_dominance_enabled: bool = True,
    absorb_clifford_angles: bool = True,
    canonicalization_mode: str = "enhanced",
    config_digest: str = "unspecified-direct-run",
) -> dict[str, object]:
    """Run one fresh, witness-free Article V1 evaluation trajectory."""

    if scheduler not in PRIMARY_SCHEDULERS:
        raise ValueError(f"scheduler must be one of {PRIMARY_SCHEDULERS!r}")
    if scheduler != "article_sarsa" and checkpoint is not None:
        raise ValueError("only article_sarsa accepts a trained checkpoint")
    if scheduler == "article_sarsa":
        if checkpoint is None:
            raise ValueError("article_sarsa requires a trained checkpoint")
        checkpoint.validate_contract(require_nonzero=True)
        if checkpoint_scope is None:
            raise ValueError(
                "article_sarsa requires an explicit checkpoint evaluation scope"
            )
        checkpoint_scope.validate_evaluation_parameters(
            beta=beta,
            certification_tolerance=certification_tolerance,
        )
        checkpoint.validate_for_evaluation(checkpoint_scope, case)
    target = _target(case)
    schema = (
        ARTICLE_V1_FEATURE_SCHEMA_VERSION
        if checkpoint is None
        else checkpoint.feature_schema_version
    )
    # Every Article V1 scheduler shares the same exact incremental frontier
    # index.  Learned/zero policies also consume its compact linear batch;
    # baselines use it only for the common reward potential (and the direct
    # target-distance arm uses its exact active-minimum heap).
    provider, target_context = _feature_provider(
        case,
        expansion_budget=expansion_budget,
        feature_schema=schema,
    )
    policy = None
    if scheduler in {"zero_weight_linear", "article_sarsa"}:
        policy = LinearQPolicy(
            feature_provider=provider,
            gamma=1.0,
            seed=evaluation_seed,
            lr=1e-3 if checkpoint is None else checkpoint.learning_rate,
        )
        if scheduler == "article_sarsa":
            assert checkpoint is not None
            if checkpoint.ordered_feature_names != tuple(provider.names):
                raise ValueError("checkpoint/provider ordered feature schema mismatch")
            if checkpoint.feature_evaluator_schema_version != str(
                getattr(provider, "evaluator_schema_version", "")
            ):
                raise ValueError("checkpoint/provider feature evaluator schema mismatch")
            policy.theta[:] = np.asarray(checkpoint.weights, dtype=np.float64)

    internal_scheduler = scheduler
    report = evaluate(
        num_qubits=case.num_qubits,
        target_gates=(),
        target_unitary=target.unitary,
        budget=case.budget.resource_budget(),
        max_steps=int(expansion_budget),
        seed=int(evaluation_seed),
        scheduler=internal_scheduler,
        collect_trace=False,
        policy=policy,
        reward_mode="article_v1_expansion_potential",
        article_v1_beta=float(beta),
        fairness_interval=0,
        feature_provider=provider,
        canonicalization_enabled=canonicalization_enabled,
        pareto_dominance_enabled=pareto_dominance_enabled,
        absorb_clifford_angles=absorb_clifford_angles,
        canonicalization_mode=canonicalization_mode,
        certification_engine=ArticleV1CertificationEngine(
            target, tau_cert=float(certification_tolerance)
        ),
        # The Article V1 reward is part of the common environment contract even
        # during frozen evaluation.  Supplying the same context to every
        # scheduler makes the reward computation, cache accounting, and stopping
        # semantics identical; frozen policies do not consume the reward.
        target_metric=target_context,
        instrumentation_enabled=True,
        observation_features=False,
    )
    metrics = dict(report["search_metrics"])
    witness_operations = list(report["witness_operations"])
    certification_diagnostics: dict[str, object] | None = None
    if bool(report["certified"]):
        certification_diagnostics = _independent_witness_certification_diagnostics(
            case,
            witness_operations,
            certification_tolerance=certification_tolerance,
        )
    elif witness_operations:
        raise ValueError("an uncertified run must not expose witness operations")
    checkpoint_digest = "none" if checkpoint is None else checkpoint.weight_digest
    provenance = git_provenance()
    timings = {
        name.removesuffix("_ns") + "_seconds": float(value) / 1e9
        for name, value in metrics.items()
        if name.endswith("_time_ns")
    }
    raw: dict[str, object] = {
        "schema_version": ARTICLE_V1_RAW_RUN_SCHEMA,
        **_case_metadata(case),
        "config_digest": str(config_digest),
        "target_fingerprint": target_context.fingerprint,
        "scheduler": scheduler,
        "scheduler_semantics": report["scheduler_semantics"],
        "action_semantics": "persistent_frontier_record",
        "resource_budget": _resource_budget_payload(case),
        "expansion_budget": int(expansion_budget),
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_family": (
            None if checkpoint is None else checkpoint.checkpoint_family
        ),
        "checkpoint_scope_schema": (
            None if checkpoint_scope is None else checkpoint_scope.schema_version
        ),
        "training_seed": None if checkpoint is None else checkpoint.training_seed,
        "evaluation_seed": int(evaluation_seed),
        "feature_schema_version": (
            ARTICLE_V1_PROFILE.feature_schema
            if provider is None
            else str(provider.schema_version)
        ),
        "feature_evaluator_schema_version": str(
            getattr(
                provider,
                "evaluator_schema_version",
                ARTICLE_V1_PROFILE.feature_evaluator_schema,
            )
        ),
        "reward_schema_version": ARTICLE_V1_PROFILE.reward_schema,
        "reward_parameters": {"beta": float(beta)},
        "target_metric_schema_version": ARTICLE_V1_PROFILE.target_metric_schema,
        "certification_schema_version": ARTICLE_V1_PROFILE.certification_schema,
        "certification_parameters": {
            "phase_frobenius_tolerance": float(certification_tolerance)
        },
        "code_version": provenance["commit_sha"],
        "source_worktree_digest": provenance["source_worktree_digest"],
        "dirty_worktree": provenance["dirty_worktree"],
        "certified": bool(report["certified"]),
        "terminated": bool(report["terminated"]),
        "truncated": bool(report["truncated"]),
        "expansions": int(report["expansions"]),
        "runtime_seconds": float(report["runtime_seconds"]),
        "time_to_solution": report["time_to_solution"],
        "timings": timings,
        "metrics": {
            "feature_evaluations": metrics.get("feature_evaluation_count", 0),
            "dense_target_evaluations": metrics.get(
                "target_metric_evaluation_count", 0
            ),
            "target_metric_cache_hits": metrics.get("target_metric_cache_hits", 0),
            "target_metric_cache_misses": metrics.get("target_metric_cache_misses", 0),
            "certification_count": metrics.get("certification_count", 0),
            "peak_frontier": metrics.get("peak_frontier_records", 0),
            "peak_archive": metrics.get("peak_active_archive_records", 0),
            "maximum_pareto_antichain_width": metrics.get(
                "maximum_pareto_antichain_width", 0
            ),
        },
        "search_metrics": metrics,
        "solution_resource_vector": report["solution_resource_vector"],
        "witness_operations": witness_operations,
        "certification_diagnostics": certification_diagnostics,
        "reference_witness_used": False,
        "target_specific_reachability_oracle": False,
        "profile": ARTICLE_V1_PROFILE.metadata(),
        "search_reduction": {
            "canonicalization_enabled": bool(canonicalization_enabled),
            "pareto_dominance_enabled": bool(pareto_dominance_enabled),
            "absorb_clifford_angles": bool(absorb_clifford_angles),
            "canonicalization_mode": str(canonicalization_mode),
        },
        "evaluation_weights_frozen": True,
        "evaluation_reward_consumed_by_policy": False,
    }
    return raw


def _budget_grid(case, experiment: Mapping[str, Any]) -> tuple[int, ...]:
    multipliers = tuple(float(value) for value in experiment["expansion_budget_multipliers"])
    return tuple(
        sorted(
            {
                max(1, int(round(case.budget.expansion_budget * multiplier)))
                for multiplier in multipliers
            }
        )
    )


@dataclass(frozen=True, slots=True)
class _ExpectedMatrixRun:
    case: ArticleV1TargetCase | ArticleV1EvaluationTarget
    scheduler: str
    expansion_budget: int
    evaluation_seed: int
    checkpoint: ArticleV1Checkpoint | None
    checkpoint_scope: ArticleV1CheckpointScope | None
    identity: Mapping[str, object]

    @property
    def run_key(self) -> str:
        return unique_run_key(self.identity)


def _expected_matrix_runs(
    cases: Sequence[ArticleV1TargetCase | ArticleV1EvaluationTarget],
    *,
    config: ArticleV1CorpusConfig,
    checkpoints: Sequence[ArticleV1Checkpoint],
    checkpoint_scope: ArticleV1CheckpointScope | None,
    schedulers: Sequence[str],
    provenance: Mapping[str, object],
    budget_override: int | None = None,
) -> tuple[_ExpectedMatrixRun, ...]:
    experiment = config.experiment
    beta = float(experiment["beta"])
    certification_tolerance = float(experiment["certification_tolerance"])
    search_reduction = {
        "canonicalization_enabled": bool(experiment["canonicalization_enabled"]),
        "pareto_dominance_enabled": bool(experiment["pareto_dominance_enabled"]),
        "absorb_clifford_angles": bool(experiment["absorb_clifford_angles"]),
        "canonicalization_mode": str(experiment["canonicalization_mode"]),
    }
    expected: list[_ExpectedMatrixRun] = []
    for case in cases:
        budgets = (
            (int(budget_override),)
            if budget_override is not None
            else _budget_grid(case, experiment)
        )
        for expansion_budget in budgets:
            for scheduler in schedulers:
                if scheduler == "seeded_random":
                    trajectories = tuple(
                        (int(seed), None)
                        for seed in experiment["random_scheduler_seeds"]
                    )
                elif scheduler == "article_sarsa":
                    trajectories = tuple((0, checkpoint) for checkpoint in checkpoints)
                else:
                    trajectories = ((0, None),)
                for evaluation_seed, checkpoint in trajectories:
                    identity = run_identity_payload({
                        "target_id": case.target_id,
                        "config_digest": config.digest,
                        "scheduler": scheduler,
                        "resource_budget": _resource_budget_payload(case),
                        "expansion_budget": int(expansion_budget),
                        "checkpoint_digest": (
                            "none" if checkpoint is None else checkpoint.weight_digest
                        ),
                        "training_seed": (
                            None if checkpoint is None else checkpoint.training_seed
                        ),
                        "evaluation_seed": int(evaluation_seed),
                        "feature_schema_version": (
                            ARTICLE_V1_PROFILE.feature_schema
                            if checkpoint is None
                            else checkpoint.feature_schema_version
                        ),
                        "feature_evaluator_schema_version": (
                            ARTICLE_V1_PROFILE.feature_evaluator_schema
                            if checkpoint is None
                            else checkpoint.feature_evaluator_schema_version
                        ),
                        "reward_schema_version": ARTICLE_V1_PROFILE.reward_schema,
                        "reward_parameters": {"beta": beta},
                        "target_metric_schema_version": (
                            ARTICLE_V1_PROFILE.target_metric_schema
                        ),
                        "certification_schema_version": (
                            ARTICLE_V1_PROFILE.certification_schema
                        ),
                        "certification_parameters": {
                            "phase_frobenius_tolerance": certification_tolerance
                        },
                        "code_version": str(provenance["commit_sha"]),
                        "source_worktree_digest": str(
                            provenance["source_worktree_digest"]
                        ),
                        "search_reduction": search_reduction,
                    })
                    expected.append(_ExpectedMatrixRun(
                        case=case,
                        scheduler=scheduler,
                        expansion_budget=int(expansion_budget),
                        evaluation_seed=int(evaluation_seed),
                        checkpoint=checkpoint,
                        checkpoint_scope=(
                            None if checkpoint is None else checkpoint_scope
                        ),
                        identity=identity,
                    ))
    keys = [item.run_key for item in expected]
    if len(keys) != len(set(keys)):
        raise ValueError("expected Article V1 campaign matrix contains duplicate keys")
    return tuple(expected)


def evaluate_article_v1_matrix(
    cases: Sequence[ArticleV1TargetCase | ArticleV1EvaluationTarget],
    *,
    config: ArticleV1CorpusConfig,
    checkpoints: Sequence[ArticleV1Checkpoint],
    raw_path: str | Path,
    checkpoint_scope: ArticleV1CheckpointScope | None = None,
    schedulers: Sequence[str] = PRIMARY_SCHEDULERS,
    budget_override: int | None = None,
) -> dict[str, int]:
    """Evaluate/resume the preregistered scheduler matrix."""

    cases = tuple(cases)
    schedulers = tuple(schedulers)
    if "article_sarsa" in schedulers:
        if not checkpoints:
            raise ValueError("article_sarsa matrix requires checkpoints")
        if checkpoint_scope is None:
            raise ValueError(
                "article_sarsa matrix requires an explicit checkpoint evaluation scope"
            )
        if checkpoint_scope.corpus_config_digest != config.digest:
            raise ValueError("checkpoint scope corpus digest does not match matrix config")
        _validate_checkpoint_campaign(checkpoints, checkpoint_scope)
        for case in cases:
            for checkpoint in checkpoints:
                checkpoint.validate_for_evaluation(checkpoint_scope, case)

    experiment = config.experiment
    beta = float(experiment["beta"])
    certification_tolerance = float(experiment["certification_tolerance"])
    canonicalization_enabled = bool(experiment["canonicalization_enabled"])
    pareto_dominance_enabled = bool(experiment["pareto_dominance_enabled"])
    absorb_clifford_angles = bool(experiment["absorb_clifford_angles"])
    canonicalization_mode = str(experiment["canonicalization_mode"])
    if "article_sarsa" in schedulers:
        assert checkpoint_scope is not None
        checkpoint_scope.validate_evaluation_parameters(
            beta=beta,
            certification_tolerance=certification_tolerance,
        )
    store = ArticleV1RunStore(raw_path)
    completed_keys = store.completed_keys()
    provenance = git_provenance()
    expected_runs = _expected_matrix_runs(
        cases,
        config=config,
        checkpoints=checkpoints,
        checkpoint_scope=checkpoint_scope,
        schedulers=schedulers,
        provenance=provenance,
        budget_override=budget_override,
    )
    appended = skipped = 0
    for expected in expected_runs:
        if expected.run_key in completed_keys:
            skipped += 1
            continue
        run = evaluate_article_v1_run(
            expected.case,
            scheduler=expected.scheduler,
            expansion_budget=expected.expansion_budget,
            evaluation_seed=expected.evaluation_seed,
            checkpoint=expected.checkpoint,
            checkpoint_scope=expected.checkpoint_scope,
            beta=beta,
            certification_tolerance=certification_tolerance,
            canonicalization_enabled=canonicalization_enabled,
            pareto_dominance_enabled=pareto_dominance_enabled,
            absorb_clifford_angles=absorb_clifford_angles,
            canonicalization_mode=canonicalization_mode,
            config_digest=config.digest,
        )
        if run_identity_payload(run) != expected.identity:
            raise ValueError("Article V1 evaluator returned a run with identity drift")
        if store.append(run):
            appended += 1
            completed_keys.add(expected.run_key)
        else:
            skipped += 1
    return {"appended": appended, "skipped": skipped, "completed": appended + skipped}


def _strict_campaign_records(path: str | Path) -> tuple[bytes, tuple[dict[str, object], ...]]:
    ledger_path = Path(path)
    try:
        raw = ledger_path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"campaign raw ledger is missing: {ledger_path}") from error
    if not raw:
        raise ValueError("campaign raw ledger is empty")
    if not raw.endswith(b"\n"):
        raise ValueError("campaign raw ledger has an incomplete final line")

    def reject_nonfinite_constant(value: str) -> object:
        raise ValueError(f"campaign raw ledger contains nonfinite JSON value {value}")

    def reject_duplicate_members(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"campaign raw ledger contains duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    records: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for line_number, encoded in enumerate(raw.splitlines(), start=1):
        if not encoded.strip():
            raise ValueError(f"campaign raw ledger contains blank line {line_number}")
        try:
            decoded = json.loads(
                encoded.decode("utf-8"),
                parse_constant=reject_nonfinite_constant,
                object_pairs_hook=reject_duplicate_members,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"campaign raw ledger has corrupt JSON at line {line_number}"
            ) from error
        if not isinstance(decoded, dict):
            raise ValueError(f"campaign raw ledger line {line_number} is not an object")
        if decoded.get("raw_run_schema") != ARTICLE_V1_RAW_RUN_SCHEMA:
            raise ValueError(
                f"campaign raw ledger line {line_number} has an incompatible raw schema"
            )
        observed_key = decoded.get("run_key")
        expected_key = unique_run_key(decoded)
        if observed_key != expected_key:
            raise ValueError(
                f"campaign raw ledger line {line_number} has an invalid run key"
            )
        if expected_key in seen_keys:
            raise ValueError(
                f"campaign raw ledger contains duplicate run key {expected_key}"
            )
        seen_keys.add(expected_key)
        records.append(decoded)
    return raw, tuple(records)


def _assert_finite_json(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"campaign record contains nonfinite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"campaign record contains unsupported value at {path}")


def _json_values_exact(observed: object, expected: object) -> bool:
    """Compare decoded JSON without Python's bool/int or int/float coercions."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _json_values_exact(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_values_exact(left, right)
            for left, right in zip(observed, expected)
        )
    return bool(observed == expected)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _nonnegative_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


_REQUIRED_TIMING_FIELDS = (
    "wall_time_seconds",
    "environment_step_time_seconds",
    "ranking_time_seconds",
    "feature_time_seconds",
    "dominance_update_time_seconds",
    "compact_batch_time_seconds",
    "last_compact_batch_time_seconds",
    "candidate_gather_time_seconds",
    "standardization_time_seconds",
    "score_time_seconds",
    "selected_row_materialization_time_seconds",
    "target_metric_time_seconds",
    "symbolic_update_time_seconds",
    "canonicalization_time_seconds",
    "archive_time_seconds",
    "certification_time_seconds",
    "reporting_time_seconds",
)
_REQUIRED_COUNTER_FIELDS = (
    "generated",
    "certification_nonmatch",
    "duplicate_rejected",
    "dominated_retired",
    "pareto_incomparable_accepted",
    "reopened",
    "expanded",
    "frontier_peak",
    "archive_size",
    "pareto_width_peak",
    "accepted",
    "canonical_pruned",
    "dominated",
    "peak_frontier",
    "terminal_candidates",
    "terminal_certification_failures",
    "num_expanded",
    "num_gate_attempts",
    "num_generated",
    "num_exact_duplicate_rejections",
    "num_dominance_rejections",
    "num_dominance_replacements",
    "num_pareto_incomparable_acceptances",
    "num_reopenings",
    "frontier_sum",
    "frontier_observation_count",
    "archive_record_count",
    "active_archive_peak",
    "certification_count",
    "feature_evaluation_count",
    "compact_batch_count",
    "target_metric_evaluation_count",
    "target_metric_cache_hits",
    "target_metric_cache_misses",
    "peak_frontier_records",
    "peak_active_archive_records",
    "maximum_pareto_antichain_width",
    "feature_static_cache_hits",
    "feature_static_cache_misses",
    "frontier_index_additions",
    "frontier_index_removals",
    "frontier_index_rebuilds",
    "unique_resource_group_count",
    "resource_group_peak",
    "feature_index_memory_bytes",
    "frontier_revision",
    "generation_count_revision",
)
_REQUIRED_SUMMARY_COUNTER_FIELDS = {
    "feature_evaluations": "feature_evaluation_count",
    "dense_target_evaluations": "target_metric_evaluation_count",
    "target_metric_cache_hits": "target_metric_cache_hits",
    "target_metric_cache_misses": "target_metric_cache_misses",
    "certification_count": "certification_count",
    "peak_frontier": "peak_frontier_records",
    "peak_archive": "peak_active_archive_records",
    "maximum_pareto_antichain_width": "maximum_pareto_antichain_width",
}
_CANONICAL_SEARCH_METRIC_FIELDS = (
    set(_REQUIRED_COUNTER_FIELDS)
    | {
        name.removesuffix("_seconds") + "_ns"
        for name in _REQUIRED_TIMING_FIELDS
    }
    | {"frontier_mean", "frontier_decision_mean"}
)
_CANONICAL_RAW_RECORD_FIELDS = {
    "schema_version",
    "raw_run_schema",
    "run_key",
    "target_id",
    "target_fingerprint",
    "config_digest",
    "split",
    "difficulty",
    "num_qubits",
    "generator_length",
    "budget",
    "resource_budget",
    "scheduler",
    "scheduler_semantics",
    "action_semantics",
    "expansion_budget",
    "checkpoint_digest",
    "checkpoint_family",
    "checkpoint_scope_schema",
    "training_seed",
    "evaluation_seed",
    "feature_schema_version",
    "feature_evaluator_schema_version",
    "reward_schema_version",
    "reward_parameters",
    "target_metric_schema_version",
    "certification_schema_version",
    "certification_parameters",
    "code_version",
    "source_worktree_digest",
    "dirty_worktree",
    "certified",
    "terminated",
    "truncated",
    "expansions",
    "runtime_seconds",
    "time_to_solution",
    "timings",
    "metrics",
    "search_metrics",
    "solution_resource_vector",
    "witness_operations",
    "certification_diagnostics",
    "reference_witness_used",
    "target_specific_reachability_oracle",
    "profile",
    "search_reduction",
    "evaluation_weights_frozen",
    "evaluation_reward_consumed_by_policy",
}
_SCHEDULER_SEMANTICS = {
    "fifo": "fifo",
    "lifo": "lifo",
    "uniform_cost": "uniform_cost",
    "seeded_random": "random",
    "zero_weight_linear": "zero_policy",
    "article_target_distance": "article_target_distance",
    "article_sarsa": "learned",
}


def _audit_campaign_record(
    record: Mapping[str, object],
    expected: _ExpectedMatrixRun,
    *,
    config: ArticleV1CorpusConfig,
    provenance: Mapping[str, object],
) -> bool:
    _assert_finite_json(record, path="run")
    observed_fields = set(record)
    missing_fields = sorted(_CANONICAL_RAW_RECORD_FIELDS - observed_fields)
    unexpected_fields = sorted(observed_fields - _CANONICAL_RAW_RECORD_FIELDS)
    if missing_fields or unexpected_fields:
        raise ValueError(
            "campaign record does not match the canonical raw schema: "
            f"missing={missing_fields}, unexpected={unexpected_fields}"
        )
    if (
        record["schema_version"] != ARTICLE_V1_RAW_RUN_SCHEMA
        or record["raw_run_schema"] != ARTICLE_V1_RAW_RUN_SCHEMA
    ):
        raise ValueError("campaign record has incompatible scientific schemas")
    if run_identity_payload(record) != expected.identity:
        raise ValueError("campaign record identity does not match its frozen matrix cell")
    if type(record["dirty_worktree"]) is not bool:
        raise ValueError("campaign record dirty_worktree must be boolean")

    case = expected.case
    target = _target(case)
    expected_fingerprint = ArticleTargetContext(target).fingerprint
    if dense_target_digest(
        target.unitary, decimals=config.digest_decimals
    ) != case.target_id:
        raise ValueError("campaign corpus target ID does not match its dense target")
    metadata_expectations = {
        "target_id": case.target_id,
        "target_fingerprint": expected_fingerprint,
        "config_digest": config.digest,
        "split": case.split,
        "difficulty": case.difficulty,
        "num_qubits": case.num_qubits,
        "generator_length": case.generator_length,
        "budget": case.budget.metadata(),
        "resource_budget": _resource_budget_payload(case),
        "profile": ARTICLE_V1_PROFILE.metadata(),
        "search_reduction": expected.identity["search_reduction"],
        "code_version": provenance["commit_sha"],
        "source_worktree_digest": provenance["source_worktree_digest"],
        "dirty_worktree": provenance["dirty_worktree"],
        "action_semantics": "persistent_frontier_record",
        "scheduler_semantics": _SCHEDULER_SEMANTICS[expected.scheduler],
        "evaluation_weights_frozen": True,
        "evaluation_reward_consumed_by_policy": False,
        "reference_witness_used": False,
        "target_specific_reachability_oracle": False,
    }
    for name, expected_value in metadata_expectations.items():
        if not _json_values_exact(record.get(name), _json_ready(expected_value)):
            raise ValueError(f"campaign record has incompatible {name}")

    expected_scope_schema = (
        None if expected.checkpoint_scope is None else expected.checkpoint_scope.schema_version
    )
    if record["checkpoint_scope_schema"] != expected_scope_schema:
        raise ValueError("campaign record has incompatible checkpoint scope schema")
    if expected.checkpoint is None:
        if record["checkpoint_family"] is not None:
            raise ValueError("checkpoint-free campaign record declares a checkpoint family")
    elif record["checkpoint_family"] != expected.checkpoint.checkpoint_family:
        raise ValueError("campaign record has incompatible checkpoint family")

    expansions = _nonnegative_int(record["expansions"], name="expansions")
    if expansions < 1:
        raise ValueError("campaign record must contain at least one search expansion")
    if expansions > expected.expansion_budget:
        raise ValueError("campaign record exceeds its expansion budget")
    certified = record["certified"]
    if not isinstance(certified, bool):
        raise ValueError("campaign record certified flag must be boolean")
    runtime_seconds = _nonnegative_number(
        record["runtime_seconds"], name="runtime_seconds"
    )
    timings = record["timings"]
    search_metrics = record["search_metrics"]
    metrics = record["metrics"]
    if not isinstance(timings, Mapping) or not isinstance(search_metrics, Mapping):
        raise ValueError("campaign record timings/search_metrics must be objects")
    if not isinstance(metrics, Mapping):
        raise ValueError("campaign record metrics must be an object")
    if set(timings) != set(_REQUIRED_TIMING_FIELDS):
        raise ValueError("campaign record timings do not match the raw schema")
    if set(search_metrics) != _CANONICAL_SEARCH_METRIC_FIELDS:
        raise ValueError("campaign record search metrics do not match the raw schema")
    for metric_name, metric_value in search_metrics.items():
        _nonnegative_number(metric_value, name=f"search_metrics.{metric_name}")
    for name in _REQUIRED_TIMING_FIELDS:
        if name not in timings:
            raise ValueError(f"campaign record is missing timing {name}")
        seconds = _nonnegative_number(timings[name], name=f"timings.{name}")
        nanoseconds_name = name.removesuffix("_seconds") + "_ns"
        nanoseconds = _nonnegative_int(
            search_metrics.get(nanoseconds_name),
            name=f"search_metrics.{nanoseconds_name}",
        )
        if not math.isclose(
            seconds,
            nanoseconds / 1e9,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"campaign timing {name} disagrees with {nanoseconds_name}"
            )
    for name in _REQUIRED_COUNTER_FIELDS:
        if name not in search_metrics:
            raise ValueError(f"campaign record is missing counter {name}")
        _nonnegative_int(search_metrics[name], name=f"search_metrics.{name}")
    compact_batch_count = int(search_metrics["compact_batch_count"])
    compact_batch_time_ns = int(search_metrics["compact_batch_time_ns"])
    last_compact_batch_time_ns = int(
        search_metrics["last_compact_batch_time_ns"]
    )
    if last_compact_batch_time_ns > compact_batch_time_ns or (
        compact_batch_count == 0
        and (compact_batch_time_ns != 0 or last_compact_batch_time_ns != 0)
    ):
        raise ValueError("campaign compact-batch timing/count telemetry disagrees")
    if int(search_metrics["expanded"]) != expansions or int(
        search_metrics["num_expanded"]
    ) != expansions:
        raise ValueError("campaign record expansion counters disagree")
    alias_equations = (
        ("generated", "num_generated"),
        ("dominated_retired", "dominated", "num_dominance_replacements"),
        (
            "pareto_incomparable_accepted",
            "num_pareto_incomparable_acceptances",
        ),
        ("reopened", "num_reopenings"),
        ("frontier_peak", "peak_frontier", "peak_frontier_records"),
        ("active_archive_peak", "peak_active_archive_records"),
        ("pareto_width_peak", "maximum_pareto_antichain_width"),
    )
    for aliases in alias_equations:
        values = {int(search_metrics[name]) for name in aliases}
        if len(values) != 1:
            raise ValueError(
                "campaign record counter aliases disagree: " + ", ".join(aliases)
            )
    duplicate_rejections = int(search_metrics["duplicate_rejected"])
    if (
        duplicate_rejections != int(search_metrics["canonical_pruned"])
        or duplicate_rejections
        != int(search_metrics["num_exact_duplicate_rejections"])
        + int(search_metrics["num_dominance_rejections"])
    ):
        raise ValueError("campaign record duplicate-rejection counters disagree")
    frontier_decision_mean = _nonnegative_number(
        search_metrics["frontier_decision_mean"],
        name="search_metrics.frontier_decision_mean",
    )
    frontier_sum = int(search_metrics["frontier_sum"])
    frontier_observations = int(search_metrics["frontier_observation_count"])
    expected_frontier_decision_mean = frontier_sum / max(1, frontier_observations)
    if not math.isclose(
        frontier_decision_mean,
        expected_frontier_decision_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("campaign frontier decision mean disagrees with its counters")
    generated = int(search_metrics["generated"])
    certification_nonmatch = int(search_metrics["certification_nonmatch"])
    native_gate_count = len(generate_actions(case.num_qubits))
    native_equations = (
        (
            int(search_metrics["terminal_candidates"]) == generated,
            "terminal_candidates must equal generated",
        ),
        (
            int(search_metrics["certification_count"]) == generated + 1,
            "certification_count must equal generated plus the root certification",
        ),
        (
            int(search_metrics["num_gate_attempts"])
            == expansions * native_gate_count,
            "num_gate_attempts must equal expansions times the native grammar size",
        ),
        (
            int(search_metrics["accepted"])
            == int(search_metrics["archive_record_count"]),
            "accepted must equal archive_record_count",
        ),
        (
            int(search_metrics["accepted"])
            - 1
            + int(search_metrics["duplicate_rejected"])
            == certification_nonmatch,
            "accepted/duplicate counters must account for every certification nonmatch",
        ),
        (
            int(search_metrics["terminal_certification_failures"]) == 0,
            "native campaign rows cannot contain terminal certification failures",
        ),
        (
            frontier_observations == expansions,
            "frontier_observation_count must equal completed expansions",
        ),
        (
            (generated > certification_nonmatch) is certified,
            "certified status disagrees with generated certification outcomes",
        ),
    )
    for condition, message in native_equations:
        if not condition:
            raise ValueError(f"campaign native-search counters disagree: {message}")
    frontier_peak = int(search_metrics["frontier_peak"])
    accepted = int(search_metrics["accepted"])
    archive_size = int(search_metrics["archive_size"])
    archive_record_count = int(search_metrics["archive_record_count"])
    active_archive_peak = int(search_metrics["active_archive_peak"])
    pareto_width_peak = int(search_metrics["pareto_width_peak"])
    native_bounds = (
        (
            generated <= int(search_metrics["num_gate_attempts"]),
            "generated cannot exceed native gate attempts",
        ),
        (
            certification_nonmatch <= generated,
            "certification nonmatches cannot exceed generated children",
        ),
        (
            frontier_peak >= 1,
            "frontier peak must be at least one",
        ),
        (
            frontier_decision_mean >= 1.0,
            "frontier decision mean must be at least one",
        ),
        (
            frontier_decision_mean <= frontier_peak,
            "frontier decision mean cannot exceed frontier peak",
        ),
        (
            accepted >= 1 and archive_record_count >= 1,
            "accepted/archive record counts must be at least one",
        ),
        (
            1 <= archive_size <= archive_record_count,
            "archive size must lie within [1, archive_record_count]",
        ),
        (
            frontier_peak <= active_archive_peak <= archive_record_count,
            "frontier/archive peaks must not exceed archive_record_count",
        ),
        (
            1 <= pareto_width_peak <= active_archive_peak,
            "Pareto width must lie within [1, active_archive_peak]",
        ),
        (
            int(search_metrics["target_metric_evaluation_count"])
            == int(search_metrics["target_metric_cache_misses"]),
            "target metric evaluations must equal cache misses",
        ),
    )
    for condition, message in native_bounds:
        if not condition:
            raise ValueError(f"campaign native-search bounds disagree: {message}")
    if set(metrics) != set(_REQUIRED_SUMMARY_COUNTER_FIELDS):
        raise ValueError("campaign record summary counters do not match the raw schema")
    for summary_name, search_name in _REQUIRED_SUMMARY_COUNTER_FIELDS.items():
        summary_value = _nonnegative_int(
            metrics[summary_name], name=f"metrics.{summary_name}"
        )
        if summary_value != int(search_metrics[search_name]):
            raise ValueError(
                f"campaign summary counter {summary_name} disagrees with {search_name}"
            )
    wall_time_ns = _nonnegative_int(
        search_metrics.get("wall_time_ns"), name="search_metrics.wall_time_ns"
    )
    if wall_time_ns < 1 or runtime_seconds <= 0.0:
        raise ValueError("campaign rows with expansions require positive wall time")
    if int(search_metrics["environment_step_time_ns"]) < 1:
        raise ValueError("campaign rows with expansions require environment step time")
    if int(search_metrics["certification_time_ns"]) < 1:
        raise ValueError("campaign rows with certifications require certification time")
    if not math.isclose(
        runtime_seconds,
        wall_time_ns / 1e9,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("campaign record wall time disagrees with runtime_seconds")
    if not math.isclose(
        float(timings["wall_time_seconds"]),
        runtime_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("campaign record timing envelope disagrees with runtime_seconds")
    for name in _REQUIRED_TIMING_FIELDS:
        if name == "wall_time_seconds":
            continue
        nanoseconds_name = name.removesuffix("_seconds") + "_ns"
        if int(search_metrics[nanoseconds_name]) > wall_time_ns:
            raise ValueError(f"campaign component timing {name} exceeds wall time")

    if not isinstance(record["terminated"], bool) or not isinstance(
        record["truncated"], bool
    ):
        raise ValueError("campaign terminal flags must be boolean")
    if record["terminated"] is record["truncated"]:
        raise ValueError(
            "campaign record must be exactly one of terminated or truncated"
        )
    if not certified and record["truncated"] is True and expansions != (
        expected.expansion_budget
    ):
        raise ValueError(
            "truncated campaign failure must exhaust its expansion budget"
        )
    witness = record["witness_operations"]
    diagnostics = record["certification_diagnostics"]
    if certified:
        if record["terminated"] is not True or record["truncated"] is not False:
            raise ValueError("certified campaign record has invalid terminal flags")
        if not isinstance(witness, list) or not witness:
            raise ValueError("certified campaign record has no witness")
        if expansions < 1:
            raise ValueError("certified campaign record has no search expansion")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("certified campaign record has no certification diagnostics")
        replayed = _independent_witness_certification_diagnostics(
            case,
            witness,
            certification_tolerance=float(
                config.experiment["certification_tolerance"]
            ),
        )
        if not _json_values_exact(dict(diagnostics), replayed):
            raise ValueError("campaign certification diagnostics do not match fresh replay")
        tau = float(config.experiment["certification_tolerance"])
        if (
            diagnostics.get("schema_version") != ARTICLE_V1_PROFILE.certification_schema
            or diagnostics.get("passed") is not True
            or diagnostics.get("reason") != "equivalent_phase_frobenius"
            or float(diagnostics.get("tau_cert", -1.0)) != tau
            or _nonnegative_number(
                diagnostics.get("delta_phi"), name="certification delta_phi"
            )
            > tau
            or diagnostics.get("candidate_finite") is not True
            or diagnostics.get("target_finite") is not True
            or diagnostics.get("candidate_unitary") is not True
            or diagnostics.get("target_unitary") is not True
            or diagnostics.get("candidate_num_qubits") != case.num_qubits
            or diagnostics.get("target_num_qubits") != case.num_qubits
        ):
            raise ValueError("campaign success has invalid certification diagnostics")
        if _nonnegative_int(
            search_metrics["certification_count"],
            name="search_metrics.certification_count",
        ) < 1:
            raise ValueError("campaign success has no certification event")
        time_to_solution = _nonnegative_number(
            record["time_to_solution"], name="time_to_solution"
        )
        if time_to_solution != runtime_seconds:
            raise ValueError("campaign time_to_solution must equal runtime_seconds")
        resource_vector = record["solution_resource_vector"]
        if not isinstance(resource_vector, list) or not resource_vector:
            raise ValueError("campaign success has no solution resource vector")
        for index, value in enumerate(resource_vector):
            _nonnegative_int(value, name=f"solution_resource_vector[{index}]")
        replayed_resource_vector = _independent_witness_resource_vector(case, witness)
        if resource_vector != replayed_resource_vector:
            raise ValueError(
                "campaign solution resource vector does not match fresh witness replay"
            )
        return True

    if witness not in ([], ()) or diagnostics is not None:
        raise ValueError("uncertified campaign record contains forged success evidence")
    if record["solution_resource_vector"] is not None or record["time_to_solution"] is not None:
        raise ValueError("uncertified campaign record contains solution diagnostics")
    return False


def audit_article_v1_campaign(
    corpus: ArticleV1Corpus,
    *,
    checkpoints: Sequence[ArticleV1Checkpoint],
    ood_checkpoints: Sequence[ArticleV1Checkpoint],
    raw_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Fail closed unless the exact primary test/OOD campaign is complete."""

    config = corpus.config
    standard_scope = corpus.checkpoint_scope(
        checkpoint_family=STANDARD_CHECKPOINT_FAMILY
    )
    ood_scope = corpus.checkpoint_scope(
        checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY
    )
    checkpoints = tuple(checkpoints)
    ood_checkpoints = tuple(ood_checkpoints)
    _validate_checkpoint_campaign(checkpoints, standard_scope)
    _validate_checkpoint_campaign(ood_checkpoints, ood_scope)
    for case in corpus.evaluation_targets(split="test"):
        for checkpoint in checkpoints:
            checkpoint.validate_for_evaluation(standard_scope, case)
    for case in corpus.evaluation_targets(split="ood_test"):
        for checkpoint in ood_checkpoints:
            checkpoint.validate_for_evaluation(ood_scope, case)

    provenance = git_provenance()
    standard_expected = _expected_matrix_runs(
        corpus.evaluation_targets(split="test"),
        config=config,
        checkpoints=checkpoints,
        checkpoint_scope=standard_scope,
        schedulers=PRIMARY_SCHEDULERS,
        provenance=provenance,
    )
    ood_expected = _expected_matrix_runs(
        corpus.evaluation_targets(split="ood_test"),
        config=config,
        checkpoints=ood_checkpoints,
        checkpoint_scope=ood_scope,
        schedulers=PRIMARY_SCHEDULERS,
        provenance=provenance,
    )
    expected = standard_expected + ood_expected
    expected_by_key = {item.run_key: item for item in expected}
    raw, records = _strict_campaign_records(raw_path)
    observed_by_key = {str(record["run_key"]): record for record in records}
    missing_keys = sorted(set(expected_by_key) - set(observed_by_key))
    unexpected_keys = sorted(set(observed_by_key) - set(expected_by_key))
    if missing_keys or unexpected_keys:
        raise ValueError(
            "campaign raw ledger does not match the exact expected matrix: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )

    certified_count = 0
    observed_by_split = {"test": 0, "ood_test": 0}
    for run_key in sorted(expected_by_key):
        item = expected_by_key[run_key]
        record = observed_by_key[run_key]
        certified_count += int(_audit_campaign_record(
            record,
            item,
            config=config,
            provenance=provenance,
        ))
        observed_by_split[item.case.split] += 1

    raw_path_value = Path(raw_path)
    output = None if output_path is None else Path(output_path)
    if output is not None:
        try:
            portable_raw_path = raw_path_value.resolve().relative_to(
                output.parent.resolve()
            ).as_posix()
        except ValueError:
            portable_raw_path = str(raw_path_value.resolve())
    else:
        portable_raw_path = raw_path_value.name
    expected_by_split = {
        "test": len(standard_expected),
        "ood_test": len(ood_expected),
    }
    report: dict[str, object] = {
        "schema_version": ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA,
        "passed": True,
        "config_profile": config.profile,
        "config_digest": config.digest,
        "code_version": provenance["commit_sha"],
        "source_worktree_digest": provenance["source_worktree_digest"],
        "raw_ledger_path": portable_raw_path,
        "raw_ledger_sha256": f"sha256:{sha256(raw).hexdigest()}",
        "expected_run_count": len(expected),
        "observed_run_count": len(records),
        "expected_by_split": expected_by_split,
        "observed_by_split": observed_by_split,
        "missing_run_keys": [],
        "unexpected_run_keys": [],
        "duplicate_run_keys": [],
        "independently_certified_success_count": certified_count,
        "integrity_checks": {
            "terminal_newline": True,
            "no_blank_or_duplicate_records": True,
            "no_duplicate_json_members": True,
            "raw_schema_and_run_keys": True,
            "exact_expected_key_set": True,
            "target_ids_and_fingerprints": True,
            "scheduler_seed_budget_checkpoint_binding": True,
            "config_source_and_schema_binding": True,
            "type_strict_scientific_metadata": True,
            "no_reference_witness_fallback": True,
            "finite_counters_and_timings": True,
            "native_search_event_equations": True,
            "successes_independently_certified": True,
        },
    }
    if output is not None:
        _atomic_json(output, report)
    return report


def validate_article_v1_checkpoints(
    cases: Sequence[ArticleV1TargetCase | ArticleV1EvaluationTarget],
    *,
    checkpoints: Sequence[ArticleV1Checkpoint],
    checkpoint_scope: ArticleV1CheckpointScope,
    output_path: str | Path,
    expansion_cap: int | None = None,
    beta: float = 1.0,
    certification_tolerance: float = 1e-9,
) -> dict[str, object]:
    """Evaluate frozen checkpoints on validation targets before opening test."""

    if not cases or not checkpoints:
        raise ValueError("validation requires cases and checkpoints")
    checkpoint_scope.validate_evaluation_parameters(
        beta=beta,
        certification_tolerance=certification_tolerance,
    )
    _validate_checkpoint_campaign(checkpoints, checkpoint_scope)
    for case in cases:
        for checkpoint in checkpoints:
            checkpoint.validate_for_evaluation(checkpoint_scope, case)
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        for case in cases:
            budget = int(case.budget.expansion_budget)
            if expansion_cap is not None:
                budget = min(budget, int(expansion_cap))
            rows.append(
                evaluate_article_v1_run(
                    case,
                    scheduler="article_sarsa",
                    expansion_budget=budget,
                    evaluation_seed=0,
                    checkpoint=checkpoint,
                    checkpoint_scope=checkpoint_scope,
                    beta=beta,
                    certification_tolerance=certification_tolerance,
                    config_digest=checkpoint_scope.corpus_config_digest,
                )
            )
    by_checkpoint: dict[str, dict[str, object]] = {}
    for checkpoint in checkpoints:
        selected = [row for row in rows if row["checkpoint_digest"] == checkpoint.weight_digest]
        successes = [row for row in selected if row["certified"]]
        by_checkpoint[checkpoint.weight_digest] = {
            "training_seed": checkpoint.training_seed,
            "targets": len(selected),
            "successes": len(successes),
            "success_rate": len(successes) / len(selected),
            "mean_successful_expansions": (
                float(np.mean([row["expansions"] for row in successes]))
                if successes
                else None
            ),
        }
    payload = {
        "schema_version": "article-v1-validation-audit-v1",
        "test_targets_observed": False,
        "selection_policy": (
            "publication hyperparameters are frozen by the pilot; all independent "
            "learner checkpoints are retained and reported"
        ),
        "checkpoints": by_checkpoint,
        "raw_runs": rows,
    }
    _atomic_json(Path(output_path), payload)
    return payload


def _run_directory(output_root: str | Path, run_id: str) -> Path:
    destination = Path(output_root) / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("corpus_manifest", "checkpoints", "figures", "tables"):
        (destination / name).mkdir(exist_ok=True)
    return destination


def _load_json_object(path: Path, *, artifact_name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"cannot resume Article V1 run: missing {artifact_name} at {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"cannot resume Article V1 run: corrupt {artifact_name} at {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"cannot resume Article V1 run: {artifact_name} is not a JSON object"
        )
    return payload


def _validate_immutable_run_artifact(
    path: Path,
    expected: Mapping[str, object],
    *,
    artifact_name: str,
) -> None:
    observed = _load_json_object(path, artifact_name=artifact_name)
    normalized_expected = _json_ready(expected)
    if observed != normalized_expected:
        raise ValueError(
            f"cannot resume Article V1 run: existing {artifact_name} conflicts "
            "with the current config, profile, code/worktree, or corpus"
        )


def _assert_checked_in_pilot_publication_disjoint(
    config: ArticleV1CorpusConfig,
    corpus: ArticleV1Corpus,
    *,
    publication_corpus: ArticleV1Corpus | None = None,
) -> None:
    """Fail before pilot execution if its frozen targets leak into publication."""

    checked_in_pilot = load_article_v1_config("pilot")
    if config.profile != "pilot" or config.digest != checked_in_pilot.digest:
        return
    publication = (
        build_article_v1_corpus("publication")
        if publication_corpus is None
        else publication_corpus
    )
    identity_tolerance = max(
        float(config.tau_identity),
        float(publication.config.tau_identity),
    )
    overlap = [
        (pilot_case.target_id, publication_case.target_id)
        for pilot_case in corpus.targets
        for publication_case in publication.targets
        if pilot_case.unitary.shape == publication_case.unitary.shape
        and article_delta_phi(pilot_case.unitary, publication_case.unitary)
        <= identity_tolerance
    ]
    if overlap:
        raise ValueError(
            "checked-in pilot and publication corpora overlap; refusing to expose "
            "publication targets during pilot selection under the shared "
            f"projective identity rule ({len(overlap)} equivalent pairs, "
            f"tau_identity={identity_tolerance})"
        )


def initialize_run(
    config_path: str | Path,
    *,
    output_root: str | Path,
    run_id: str,
) -> tuple[Path, ArticleV1Corpus]:
    config = load_article_v1_config(config_path)
    corpus = build_article_v1_corpus(config)
    _assert_checked_in_pilot_publication_disjoint(config, corpus)
    provenance = environment_metadata()
    manifest = corpus.manifest()
    run_manifest = {
        "schema_version": ARTICLE_V1_RUNNER_SCHEMA,
        "profile": ARTICLE_V1_PROFILE.metadata(),
        "config": config.to_dict(),
        "config_digest": config.digest,
        "code": provenance["git"],
        "seeds": {
            "training": list(config.experiment["training_seeds"]),
            "random_scheduler": list(config.experiment["random_scheduler_seeds"]),
            "validation": list(config.experiment["validation_seeds"]),
            "statistics": int(config.experiment["statistics_seed"]),
        },
        "full_publication_run_is_not_a_unit_test": True,
        "generator_witness_used_for_search": False,
        "execution": provenance["execution"],
    }
    split_manifests: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test", "ood_test"):
        split_payload = {
            key: value for key, value in manifest.items() if key != "cases"
        }
        split_payload["split"] = split
        split_payload["cases"] = [
            case for case in manifest["cases"] if case["split"] == split
        ]
        split_payload["target_count"] = len(split_payload["cases"])
        split_manifests[split] = split_payload

    destination = Path(output_root) / run_id
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Article V1 run destination is not a directory: {destination}")
    existing_entries = list(destination.iterdir()) if destination.exists() else []
    if existing_entries:
        run_manifest_path = destination / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise ValueError(
                "cannot resume Article V1 run: the nonempty destination has no "
                "run_manifest.json; refusing to mix stale checkpoints or raw runs"
            )
        _validate_immutable_run_artifact(
            run_manifest_path,
            run_manifest,
            artifact_name="run manifest",
        )
        _validate_immutable_run_artifact(
            destination / "environment.json",
            provenance,
            artifact_name="environment manifest",
        )
        _validate_immutable_run_artifact(
            destination / "corpus_manifest" / "manifest.json",
            manifest,
            artifact_name="corpus manifest",
        )
        for split, split_payload in split_manifests.items():
            _validate_immutable_run_artifact(
                destination / "corpus_manifest" / f"{split}.json",
                split_payload,
                artifact_name=f"{split} corpus manifest",
            )
        # A compatible resume preserves immutable bytes rather than rewriting
        # provenance around an existing raw ledger/checkpoint family.
        _run_directory(output_root, run_id)
        return destination, corpus

    destination = _run_directory(output_root, run_id)
    _atomic_json(destination / "environment.json", provenance)
    _atomic_json(destination / "corpus_manifest" / "manifest.json", manifest)
    for split, split_payload in split_manifests.items():
        _atomic_json(
            destination / "corpus_manifest" / f"{split}.json",
            split_payload,
        )
    _atomic_json(destination / "run_manifest.json", run_manifest)
    return destination, corpus


def _mini_ci_semantic_checks(
    records: Sequence[Mapping[str, object]],
    *,
    target_ids: Sequence[str],
    experiment: Mapping[str, object],
    checkpoint: ArticleV1Checkpoint,
) -> dict[str, bool]:
    """Validate that mini-CI exercised the scientific contracts, not just rows."""

    scheduler_counts = {
        scheduler: sum(record.get("scheduler") == scheduler for record in records)
        for scheduler in PRIMARY_SCHEDULERS
    }
    random_seeds = tuple(int(seed) for seed in experiment["random_scheduler_seeds"])
    expected_record_count = len(PRIMARY_SCHEDULERS) - 1 + len(random_seeds)
    deterministic_records = [
        record for record in records if record.get("scheduler") != "seeded_random"
    ]
    sarsa_records = [
        record for record in records if record.get("scheduler") == "article_sarsa"
    ]
    fifo_records = [record for record in records if record.get("scheduler") == "fifo"]
    return {
        "exact_record_count": len(records) == expected_record_count,
        "all_scheduler_labels_exact": (
            set(scheduler_counts) == set(PRIMARY_SCHEDULERS)
            and scheduler_counts["seeded_random"] == len(random_seeds)
            and all(
                scheduler_counts[scheduler] == 1
                for scheduler in PRIMARY_SCHEDULERS
                if scheduler != "seeded_random"
            )
        ),
        "seeded_random_trajectories_exact": sorted(
            int(record["evaluation_seed"])
            for record in records
            if record.get("scheduler") == "seeded_random"
        )
        == sorted(random_seeds),
        "deterministic_evaluation_seeds_zero": all(
            int(record.get("evaluation_seed", -1)) == 0
            for record in deterministic_records
        ),
        "target_set_exact": {str(record.get("target_id")) for record in records}
        == set(target_ids),
        "sarsa_checkpoint_binding_exact": len(sarsa_records) == 1
        and sarsa_records[0].get("checkpoint_digest") == checkpoint.weight_digest
        and sarsa_records[0].get("training_seed") == checkpoint.training_seed,
        "article_schemas_exact": all(
            record.get("feature_schema_version") == ARTICLE_V1_PROFILE.feature_schema
            and record.get("feature_evaluator_schema_version")
            == ARTICLE_V1_PROFILE.feature_evaluator_schema
            and record.get("reward_schema_version") == ARTICLE_V1_PROFILE.reward_schema
            and record.get("target_metric_schema_version")
            == ARTICLE_V1_PROFILE.target_metric_schema
            and record.get("certification_schema_version")
            == ARTICLE_V1_PROFILE.certification_schema
            for record in records
        ),
        "reward_and_certifier_parameters_exact": all(
            record.get("reward_parameters")
            == {"beta": float(experiment["beta"])}
            and record.get("certification_parameters")
            == {
                "phase_frobenius_tolerance": float(
                    experiment["certification_tolerance"]
                )
            }
            for record in records
        ),
        "persistent_frontier_action_semantics": all(
            record.get("action_semantics") == "persistent_frontier_record"
            for record in records
        ),
        "no_reference_witness_fallback": all(
            record.get("reference_witness_used") is False
            and record.get("target_specific_reachability_oracle") is False
            for record in records
        ),
        # FIFO is the named deterministic reachability smoke arm.  Its success
        # is produced by the independent Article certifier, never by the stored
        # corpus witness.
        "fifo_independently_certified_known_reachable_target": len(fifo_records) == 1
        and fifo_records[0].get("certified") is True
        and fifo_records[0].get("terminated") is True
        and fifo_records[0].get("truncated") is False,
    }


def mini_ci_benchmark(
    output_root: str | Path,
    *,
    run_id: str = "mini-ci",
    force_retrain: bool = False,
) -> dict[str, object]:
    """Run a deterministic, bounded end-to-end Article V1 smoke benchmark."""

    destination, corpus = initialize_run(
        "pilot", output_root=output_root, run_id=run_id
    )
    training_case = corpus.cases(split="train", difficulty="easy")[:1]
    test_case = corpus.cases(split="test", difficulty="easy")[:1]
    experiment = corpus.config.experiment
    progress_reporter = ArticleV1ProgressReporter(
        destination,
        cadence=ProgressCadence(),
        quiet=True,
    )
    training_seed = int(experiment["training_seeds"][0])
    checkpoint_scope = corpus.checkpoint_scope(
        checkpoint_family=STANDARD_CHECKPOINT_FAMILY,
        training_scope_mode=PARTIAL_SMOKE_TRAINING_SCOPE,
        training_target_ids=tuple(case.target_id for case in training_case),
        allowed_training_seeds=(training_seed,),
        expected_episodes_per_target=1,
        expected_expansion_cap=16,
    )
    checkpoint_path = destination / "checkpoints" / "seed-0.json"
    checkpoint, checkpoint_trained = _load_or_train_article_v1_checkpoint(
        checkpoint_path,
        scope=checkpoint_scope,
        expected_training_seed=training_seed,
        force_retrain=force_retrain,
        train_callback=lambda: train_article_v1_checkpoint(
            training_case,
            corpus_config_digest=corpus.config.digest,
            training_seed=training_seed,
            episodes_per_target=1,
            learning_rate=float(experiment["learning_rate"]),
            epsilon_start=float(experiment["epsilon"]["start"]),
            epsilon_minimum=float(experiment["epsilon"]["minimum"]),
            epsilon_decay=float(experiment["epsilon"]["decay"]),
            beta=float(experiment["beta"]),
            training_scope_mode=PARTIAL_SMOKE_TRAINING_SCOPE,
            expansion_cap=16,
            certification_tolerance=float(
                experiment["certification_tolerance"]
            ),
            training_checkpoint_dir=(
                destination / "training_state" / "standard-seed-0"
            ),
            progress_reporter=progress_reporter,
            run_id=run_id,
            resume_training=not force_retrain,
        ),
    )
    matrix = evaluate_article_v1_matrix(
        test_case,
        config=corpus.config,
        checkpoints=(checkpoint,),
        checkpoint_scope=checkpoint_scope,
        raw_path=destination / "raw_runs.jsonl",
        schedulers=PRIMARY_SCHEDULERS,
        budget_override=16,
    )
    artifacts = write_article_v1_report(
        destination / "raw_runs.jsonl",
        destination,
        stats_seed=int(experiment["statistics_seed"]),
        bootstrap_samples=300,
    )
    records = ArticleV1RunStore(destination / "raw_runs.jsonl").load_records()
    target_ids = [case.target_id for case in test_case]
    semantic_checks = _mini_ci_semantic_checks(
        records,
        target_ids=target_ids,
        experiment=experiment,
        checkpoint=checkpoint,
    )
    summary = {
        "schema_version": "article-v1-mini-ci-v1",
        "passed": all(semantic_checks.values()),
        "matrix": matrix,
        "target_ids": target_ids,
        "checkpoint_digest": checkpoint.weight_digest,
        "checkpoint_training_seed": checkpoint.training_seed,
        "checkpoint_trained_this_run": checkpoint_trained,
        "raw_record_count": len(records),
        "expected_raw_record_count": (
            len(PRIMARY_SCHEDULERS)
            - 1
            + len(experiment["random_scheduler_seeds"])
        ),
        "semantic_checks": semantic_checks,
        "artifacts": artifacts,
        "no_reference_witness_fallback": semantic_checks[
            "no_reference_witness_fallback"
        ],
    }
    _atomic_json(destination / "mini_ci_summary.json", summary)
    return summary


def campaign_plan(
    config: str | Path,
    *,
    worker_count: int = 1,
    pilot_seconds_per_expansion: float | None = None,
) -> dict[str, object]:
    """Return a deterministic, no-execution campaign cardinality report."""

    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ValueError("worker_count must be a positive integer")
    if pilot_seconds_per_expansion is not None and (
        not np.isfinite(pilot_seconds_per_expansion)
        or pilot_seconds_per_expansion <= 0.0
    ):
        raise ValueError("pilot_seconds_per_expansion must be finite and positive")
    resolved = load_article_v1_config(config)
    corpus = build_article_v1_corpus(resolved)
    experiment = resolved.experiment
    splits = ("train", "validation", "test", "ood_test")
    split_counts = {
        split: len(corpus.cases(split=split))
        for split in splits
    }
    learners = len(experiment["training_seeds"])
    random_repeats = len(experiment["random_scheduler_seeds"])
    scheduler_instances = 5 + random_repeats + learners
    budget_grids = {
        split: {
            case.target_id: list(_budget_grid(case, experiment))
            for case in corpus.cases(split=split)
        }
        for split in splits
    }

    def matrix_counts(split: str) -> tuple[int, int, int]:
        grids = budget_grids[split].values()
        runs = sum(len(grid) * scheduler_instances for grid in grids)
        random_runs = sum(len(grid) * random_repeats for grid in grids)
        expansions = sum(sum(grid) * scheduler_instances for grid in grids)
        return runs, random_runs, expansions

    standard_runs, standard_random_runs, standard_expansions = matrix_counts("test")
    ood_runs, ood_random_runs, ood_expansions = matrix_counts("ood_test")
    episodes_per_target = int(experiment["training_episodes_per_target"])
    train_cases = corpus.cases(split="train")
    ood_limit = int(resolved.ood_length_split.training_max_generator_length)
    ood_train_cases = tuple(
        case for case in train_cases if case.generator_length <= ood_limit
    )
    standard_training_episodes = len(train_cases) * learners * episodes_per_target
    ood_training_episodes = len(ood_train_cases) * learners * episodes_per_target
    standard_training_expansions = sum(
        case.budget.expansion_budget * learners * episodes_per_target
        for case in train_cases
    )
    ood_training_expansions = sum(
        case.budget.expansion_budget * learners * episodes_per_target
        for case in ood_train_cases
    )
    validation_cases = corpus.cases(split="validation")
    ood_validation_cases = tuple(
        case for case in validation_cases if case.generator_length <= ood_limit
    )
    primary_validation_runs = len(validation_cases) * learners
    ood_validation_runs = len(ood_validation_cases) * learners
    validation_expansions = sum(
        case.budget.expansion_budget * learners
        for case in validation_cases + ood_validation_cases
    )

    from experiments.article_v1_ablations import (
        ARTICLE_V1_ABLATION_REGISTRY,
        FULL_VALIDATION_SCOPE,
        REQUIRED_ABLATION_IDS,
    )

    subset_count = sum(
        1 for stratum in ("easy", "medium", "hard")
        if corpus.cases(split="validation", difficulty=stratum)
    )
    ablation_runs = 0
    ablation_expansions = 0
    ablation_training_variants = 0
    for ablation_id in REQUIRED_ABLATION_IDS:
        ablation = ARTICLE_V1_ABLATION_REGISTRY[ablation_id]
        case_count = (
            len(validation_cases)
            if ablation.evaluation_scope == FULL_VALIDATION_SCOPE
            else subset_count
        )
        checkpoint_count = 1 if ablation.checkpoint_mode == "none" else learners
        ablation_runs += case_count * checkpoint_count
        cases = (
            validation_cases
            if ablation.evaluation_scope == FULL_VALIDATION_SCOPE
            else tuple(
                corpus.cases(split="validation", difficulty=stratum)[0]
                for stratum in ("easy", "medium", "hard")
                if corpus.cases(split="validation", difficulty=stratum)
            )
        )
        ablation_expansions += sum(
            case.budget.expansion_budget * checkpoint_count for case in cases
        )
        if ablation.checkpoint_mode == "train_variant":
            ablation_training_variants += 1
    ablation_training_episodes = (
        ablation_training_variants * len(train_cases) * learners * episodes_per_target
    )
    ablation_training_expansions = (
        ablation_training_variants * standard_training_expansions
    )
    expansion_breakdown = {
        "standard_training": standard_training_expansions,
        "ood_training": ood_training_expansions,
        "validation": validation_expansions,
        "standard_test": standard_expansions,
        "ood_test": ood_expansions,
        "ablation_training": ablation_training_expansions,
        "ablation_evaluation": ablation_expansions,
    }
    worst_case_expansions = sum(expansion_breakdown.values())
    expected_keys = standard_runs + ood_runs
    estimated_disk_bytes = expected_keys * 16_384 + ablation_runs * 32_768
    cpu_seconds = (
        None
        if pilot_seconds_per_expansion is None
        else worst_case_expansions * float(pilot_seconds_per_expansion)
    )
    report = {
        "schema_version": "article-v1-campaign-plan-v1",
        "config_profile": resolved.profile,
        "config_digest": resolved.digest,
        "target_counts": split_counts,
        "target_counts_by_split_stratum": {
            split: {
                stratum: len(corpus.cases(split=split, difficulty=stratum))
                for stratum in ("easy", "medium", "hard")
            }
            for split in split_counts
        },
        "target_counts_by_split_stratum_qubits": {
            split: {
                stratum: {
                    str(width): sum(
                        case.num_qubits == width
                        for case in corpus.cases(split=split, difficulty=stratum)
                    )
                    for width in resolved.qubits
                }
                for stratum in ("easy", "medium", "hard")
            }
            for split in splits
        },
        "learner_checkpoint_count": 2 * learners,
        "learner_checkpoint_breakdown": {"standard": learners, "ood_length": learners},
        "training_episodes": standard_training_episodes + ood_training_episodes + ablation_training_episodes,
        "training_episode_breakdown": {
            "standard": standard_training_episodes,
            "ood_length": ood_training_episodes,
            "ablation_variants": ablation_training_episodes,
        },
        "scheduler_instances": {
            "deterministic_nonlearner": 5,
            "seeded_random": random_repeats,
            "article_sarsa": learners,
            "total_per_target_budget": scheduler_instances,
        },
        "expansion_budget_multipliers": list(experiment["expansion_budget_multipliers"]),
        "expansion_budgets": budget_grids,
        "standard_test_run_count": standard_runs,
        "ood_run_count": ood_runs,
        "validation_run_count": primary_validation_runs + ood_validation_runs,
        "validation_run_breakdown": {
            "standard": primary_validation_runs,
            "ood_length": ood_validation_runs,
        },
        "ablation_run_count": ablation_runs,
        "ablation_configuration_count": len(REQUIRED_ABLATION_IDS),
        "repeated_random_run_count": standard_random_runs + ood_random_runs,
        "expected_raw_ledger_keys": expected_keys,
        "expected_raw_ledger_key_breakdown": {
            "standard_test": standard_runs,
            "ood_test": ood_runs,
        },
        "worst_case_expansion_count": worst_case_expansions,
        "worst_case_expansion_breakdown": expansion_breakdown,
        "estimated_disk_use": {
            "bytes": estimated_disk_bytes,
            "method": "16384 bytes per primary raw JSONL record plus 32768 bytes per ablation record",
        },
        "estimated_cpu_time_from_pilot": {
            "seconds": cpu_seconds,
            "wall_seconds_at_selected_workers": (
                None if cpu_seconds is None else cpu_seconds / worker_count
            ),
            "pilot_seconds_per_expansion": pilot_seconds_per_expansion,
            "status": "awaiting-pilot-measurement" if cpu_seconds is None else "estimated",
        },
        "selected_worker_count": worker_count,
        "executes_search": False,
    }
    return report


def benchmark_article_v1_features(
    config: str | Path,
    *,
    output_root: str | Path = Path("outputs") / "article_v1",
    run_id: str = "article-v1-feature-index-v2",
    reference_safe_frontier_size: int = 1024,
    microbenchmark_repetitions: int = 3,
    microbenchmark_warmups: int = 1,
    frontier_capture_expansion_limit: int = 512,
    correctness_timeout_seconds: float = 300.0,
    maximum_hard_episode_seconds: float | None = None,
    maximum_peak_index_memory_bytes: int | None = None,
    hard_training_episode_count: int | None = None,
    write_profiles: bool = True,
) -> dict[str, Any]:
    """Safely qualify the real Article V1 feature evaluator before a pilot.

    Structural and reference-equivalence evidence is written first.  The
    repository benchmark adapter is not constructed, and therefore no timing
    begins, unless both preflight gates pass.
    """

    from experiments.article_v1_feature_benchmark import (
        DEFAULT_FRONTIER_SIZES,
        DEFAULT_STAGED_EXPANSION_CAPS,
        PRODUCTION_DOMINANCE_IMPLEMENTATION_CHECK,
        PilotFeasibilityCriteria,
        inspect_production_dominance_update,
        run_focused_correctness_gate,
        run_repository_feature_benchmark,
        write_implementation_check_evidence,
    )

    status_schema = "article-v1-feature-benchmark-status-v1"
    status_name = "benchmark_status.json"

    def source_snapshot(value: Mapping[str, object]) -> dict[str, object]:
        return {
            name: _json_ready(value.get(name))
            for name in (
                "commit_sha",
                "branch",
                "dirty_worktree",
                "source_worktree_digest",
                "relevant_untracked_files",
            )
        }

    def source_is_known(value: Mapping[str, object]) -> bool:
        return (
            value.get("commit_sha") not in {None, "", "unknown"}
            and value.get("source_worktree_digest") not in {None, "", "unknown"}
            and type(value.get("dirty_worktree")) is bool
        )

    def config_snapshot(source: str | Path) -> dict[str, object]:
        resolved = load_article_v1_config(source)
        return {
            "profile": resolved.profile,
            "digest": resolved.digest,
            "resolved_config": resolved.to_dict(),
        }

    def same_config(
        left: Mapping[str, object], right: Mapping[str, object]
    ) -> bool:
        return (
            left.get("profile") == right.get("profile") == "pilot"
            and left.get("digest") == right.get("digest")
            and left.get("resolved_config") == right.get("resolved_config")
        )

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("feature benchmark run_id must be a nonempty string")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("feature benchmark run_id must be one path component")
    if not isinstance(write_profiles, bool):
        raise ValueError("write_profiles must be a bool")
    positive_integer_arguments = {
        "reference_safe_frontier_size": reference_safe_frontier_size,
        "microbenchmark_repetitions": microbenchmark_repetitions,
        "frontier_capture_expansion_limit": frontier_capture_expansion_limit,
    }
    if hard_training_episode_count is not None:
        positive_integer_arguments["hard_training_episode_count"] = (
            hard_training_episode_count
        )
    for name, value in positive_integer_arguments.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(microbenchmark_warmups, bool)
        or not isinstance(microbenchmark_warmups, (int, np.integer))
        or int(microbenchmark_warmups) < 0
    ):
        raise ValueError("microbenchmark_warmups must be a non-negative integer")
    if (
        isinstance(correctness_timeout_seconds, bool)
        or not isinstance(correctness_timeout_seconds, (int, float, np.number))
        or not math.isfinite(float(correctness_timeout_seconds))
        or float(correctness_timeout_seconds) <= 0.0
    ):
        raise ValueError("correctness_timeout_seconds must be finite and positive")
    if (maximum_hard_episode_seconds is None) != (
        maximum_peak_index_memory_bytes is None
    ):
        raise ValueError(
            "both feature benchmark feasibility bounds must be supplied together"
        )

    feasibility_criteria = None
    if maximum_hard_episode_seconds is not None:
        assert maximum_peak_index_memory_bytes is not None
        feasibility_criteria = PilotFeasibilityCriteria(
            maximum_hard_episode_seconds=maximum_hard_episode_seconds,
            maximum_peak_index_memory_bytes=maximum_peak_index_memory_bytes,
        )

    destination = Path(output_root).resolve() / run_id
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        return {
            "schema_version": "article-v1-feature-benchmark-command-v2",
            "passed": False,
            "engineering_qualification_passed": False,
            "pilot_relaunch_ready": False,
            "aborted_before_timing": True,
            "abort_reason": (
                "feature benchmark destination already exists and is nonempty; "
                "use a new run ID"
            ),
            "output_directory": str(destination),
            "status_manifest": None,
            "artifacts": None,
        }
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / status_name

    def artifact_manifest() -> dict[str, object]:
        files: dict[str, dict[str, object]] = {}
        for path in sorted(destination.rglob("*")):
            if (
                not path.is_file()
                or path == status_path
                or path.name.endswith(".tmp")
            ):
                continue
            relative = path.relative_to(destination).as_posix()
            content = path.read_bytes()
            files[relative] = {
                "sha256": f"sha256:{sha256(content).hexdigest()}",
                "bytes": len(content),
            }
        required_files = (
            "baseline.json",
            "microbenchmarks.csv",
            "end_to_end_scaling.csv",
            "scaling_report.md",
            "projected_pilot_cost.json",
        )
        profile_entries = {
            name: record
            for name, record in files.items()
            if name.startswith("profiles/")
        }
        profile_digest = sha256()
        for name, record in profile_entries.items():
            profile_digest.update(name.encode("utf-8"))
            profile_digest.update(str(record["sha256"]).encode("ascii"))
            profile_digest.update(str(record["bytes"]).encode("ascii"))
        return {
            "files": files,
            "required_six_artifacts_complete": (
                all(name in files for name in required_files)
                and bool(profile_entries)
            ),
            "profiles": {
                "file_count": len(profile_entries),
                "sha256": f"sha256:{profile_digest.hexdigest()}",
                "files": profile_entries,
            },
        }

    def publish_status(payload: Mapping[str, object]) -> dict[str, object]:
        document = {
            "schema_version": status_schema,
            **dict(payload),
            "artifact_manifest": artifact_manifest(),
            "written_last": True,
        }
        _atomic_json(status_path, document)
        return document

    initial_source = source_snapshot(git_provenance(refresh=True))
    try:
        canonical_config = config_snapshot("pilot")
        requested_config = config_snapshot(config)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        status = publish_status(
            {
                "phase": "configuration-preflight",
                "engineering_qualification_passed": False,
                "pilot_relaunch_ready": False,
                "abort_reason": f"could not resolve frozen pilot config: {exc}",
                "source_snapshots": {"initial": initial_source},
            }
        )
        return {
            "schema_version": "article-v1-feature-benchmark-command-v2",
            "passed": False,
            "engineering_qualification_passed": False,
            "pilot_relaunch_ready": False,
            "aborted_before_timing": True,
            "abort_reason": status["abort_reason"],
            "output_directory": str(destination),
            "status_manifest": str(status_path),
            "artifacts": None,
        }
    config_binding = {
        "requested_source": str(config),
        "requested": requested_config,
        "canonical_checked_in_pilot": canonical_config,
        "matches_frozen_pilot": same_config(requested_config, canonical_config),
    }
    evidence_binding = {
        "source": initial_source,
        "config_profile": requested_config["profile"],
        "config_digest": requested_config["digest"],
        "canonical_pilot_config_digest": canonical_config["digest"],
    }
    common_summary: dict[str, Any] = {
        "schema_version": "article-v1-feature-benchmark-command-v2",
        "evidence_class": "engineering-performance-diagnostic",
        "scientific_scheduler_evidence": False,
        "output_directory": str(destination),
        "frontier_sizes": list(DEFAULT_FRONTIER_SIZES),
        "staged_expansion_caps": list(DEFAULT_STAGED_EXPANSION_CAPS),
        "reference_safe_frontier_size": int(reference_safe_frontier_size),
        "config_binding": config_binding,
        "source_snapshots": {"initial": initial_source},
        "status_manifest": str(status_path),
    }

    def failed_before_timing(
        reason: str,
        *,
        phase: str,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        payload = {
            **common_summary,
            **dict(extra or {}),
            "phase": phase,
            "engineering_qualification_passed": False,
            "pilot_relaunch_ready": False,
            "abort_reason": reason,
        }
        publish_status(payload)
        return {
            **common_summary,
            **dict(extra or {}),
            "passed": False,
            "engineering_qualification_passed": False,
            "pilot_relaunch_ready": False,
            "aborted_before_timing": True,
            "abort_reason": reason,
            "artifacts": None,
        }

    if not source_is_known(initial_source):
        return failed_before_timing(
            "fresh git source provenance is unavailable",
            phase="source-preflight",
        )
    if config_binding["matches_frozen_pilot"] is not True:
        return failed_before_timing(
            "requested config is not the frozen checked-in pilot config",
            phase="configuration-preflight",
        )

    implementation_report = inspect_production_dominance_update()
    implementation_report["evidence_binding"] = evidence_binding
    implementation_evidence = write_implementation_check_evidence(
        destination, implementation_report
    )
    implementation_checks = {
        PRODUCTION_DOMINANCE_IMPLEMENTATION_CHECK: bool(
            implementation_report.get("passed") is True
        )
    }
    common_summary["implementation_check"] = implementation_report
    common_summary["implementation_evidence"] = str(implementation_evidence)
    if not implementation_checks[PRODUCTION_DOMINANCE_IMPLEMENTATION_CHECK]:
        return failed_before_timing(
            "production dominance implementation check failed",
            phase="implementation-gate",
            extra={"correctness_gate": None},
        )

    correctness_gate, correctness_report = run_focused_correctness_gate(
        destination,
        timeout_seconds=correctness_timeout_seconds,
        evidence_binding=evidence_binding,
    )
    common_summary["correctness_gate"] = correctness_gate.as_dict()
    common_summary["correctness_evidence"] = correctness_report
    if not correctness_gate.passed:
        return failed_before_timing(
            "reference-equivalence correctness gate failed",
            phase="correctness-gate",
        )

    before_timing_source = source_snapshot(git_provenance(refresh=True))
    common_summary["source_snapshots"]["before_timing"] = before_timing_source
    try:
        canonical_before_timing = config_snapshot("pilot")
        requested_before_timing = config_snapshot(config)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return failed_before_timing(
            f"could not re-read frozen config before timing: {exc}",
            phase="pre-timing-integrity",
        )
    config_unchanged_before_timing = (
        same_config(canonical_before_timing, canonical_config)
        and same_config(requested_before_timing, requested_config)
        and same_config(requested_before_timing, canonical_before_timing)
    )
    source_unchanged_before_timing = before_timing_source == initial_source
    common_summary["pre_timing_integrity"] = {
        "source_unchanged": source_unchanged_before_timing,
        "config_unchanged": config_unchanged_before_timing,
    }
    if not source_unchanged_before_timing or not config_unchanged_before_timing:
        return failed_before_timing(
            "source or frozen pilot config changed before timing",
            phase="pre-timing-integrity",
        )

    provenance = before_timing_source
    source_clean = (
        provenance.get("dirty_worktree") is False
        and provenance.get("commit_sha") not in {None, "", "unknown"}
        and provenance.get("source_worktree_digest") not in {None, "", "unknown"}
    )
    artifacts = run_repository_feature_benchmark(
        destination,
        correctness_gate=correctness_gate,
        implementation_checks=implementation_checks,
        config="pilot",
        frontier_sizes=DEFAULT_FRONTIER_SIZES,
        staged_caps=DEFAULT_STAGED_EXPANSION_CAPS,
        feasibility_criteria=feasibility_criteria,
        pilot_relaunch_checks={
            "source_revision_committed_and_clean": source_clean,
        },
        benchmark_provenance={
            "code_version": str(provenance["commit_sha"]),
            "source_worktree_digest": str(provenance["source_worktree_digest"]),
            "worktree_clean": not bool(provenance["dirty_worktree"]),
            "source_snapshots": {
                "initial": initial_source,
                "before_timing": before_timing_source,
            },
            "config_binding": config_binding,
        },
        hard_training_episode_count=hard_training_episode_count,
        write_profiles=write_profiles,
        adapter_kwargs={
            "reference_safe_frontier_size": reference_safe_frontier_size,
            "microbenchmark_repetitions": microbenchmark_repetitions,
            "microbenchmark_warmups": microbenchmark_warmups,
            "frontier_capture_expansion_limit": frontier_capture_expansion_limit,
        },
    )
    artifact_paths = {
        "baseline_json": str(artifacts.baseline_json),
        "microbenchmarks_csv": str(artifacts.microbenchmarks_csv),
        "end_to_end_scaling_csv": str(artifacts.end_to_end_scaling_csv),
        "profiles_directory": str(artifacts.profiles_directory),
        "scaling_report_md": str(artifacts.scaling_report_md),
        "projected_pilot_cost_json": str(artifacts.projected_pilot_cost_json),
    }
    after_artifacts_source = source_snapshot(git_provenance(refresh=True))
    common_summary["source_snapshots"]["after_artifacts"] = after_artifacts_source
    try:
        canonical_after = config_snapshot("pilot")
        requested_after = config_snapshot(config)
        config_unchanged_after = (
            same_config(canonical_after, canonical_config)
            and same_config(requested_after, requested_config)
            and same_config(requested_after, canonical_after)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        config_unchanged_after = False
    source_unchanged_after = after_artifacts_source == initial_source
    engineering_passed = bool(
        artifacts.qualification.get("passed") is True
        and source_unchanged_after
        and config_unchanged_after
    )
    pilot_relaunch_ready = bool(
        engineering_passed
        and artifacts.projection.get("pilot_decision")
        == "configured pilot is feasible unchanged"
    )
    final_integrity = {
        "source_unchanged": source_unchanged_after,
        "config_unchanged": config_unchanged_after,
    }
    final_payload = {
        **common_summary,
        "phase": "complete" if engineering_passed else "post-artifact-integrity-failed",
        "engineering_qualification_passed": engineering_passed,
        "pilot_relaunch_ready": pilot_relaunch_ready,
        "final_integrity": final_integrity,
        "abort_reason": (
            None
            if engineering_passed
            else "performance qualification or final source/config integrity failed"
        ),
        "qualification": dict(artifacts.qualification),
        "projection": dict(artifacts.projection),
        "artifacts": artifact_paths,
    }
    publish_status(final_payload)
    return {
        **final_payload,
        "passed": engineering_passed,
        "aborted_before_timing": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "pilot",
        "generate-corpus",
        "train",
        "evaluate",
        "audit",
        "aggregate",
        "ablations",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output-root", type=Path, default=Path("outputs") / "article_v1")
        child.add_argument("--run-id", default="publication")
        child.add_argument("--force-retrain", action="store_true")
        child.add_argument("--quiet", action="store_true")
        child.add_argument("--progress-every-expansions", type=int, default=25)
        child.add_argument("--progress-every-seconds", type=float, default=10.0)
        child.add_argument("--checkpoint-every-expansions", type=int, default=64)
        child.add_argument("--checkpoint-every-seconds", type=float, default=60.0)
        child.add_argument("--no-training-resume", action="store_true")
    mini = subparsers.add_parser("mini-ci")
    mini.add_argument("--output-root", type=Path, default=Path("outputs") / "article_v1")
    mini.add_argument("--run-id", default="mini-ci")
    mini.add_argument("--force-retrain", action="store_true")
    calibrate = subparsers.add_parser("calibrate-certifier")
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--output-root", type=Path, default=Path("outputs") / "article_v1")
    calibrate.add_argument("--run-id", default="article-v1-raw-metric-calibration")
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--workers", type=int, default=1)
    plan_parser.add_argument("--pilot-seconds-per-expansion", type=float)
    feature_benchmark = subparsers.add_parser(
        "benchmark-features",
        description=(
            "Qualify the exact incremental Article V1 feature evaluator after "
            "a focused reference-equivalence pytest gate. Fixed axes: "
            "F=32,64,128,256,512,1024,2048 and caps=32,64,128,256,512,1024."
        ),
    )
    feature_benchmark.add_argument("--config", type=Path, required=True)
    feature_benchmark.add_argument(
        "--output-root", type=Path, default=Path("outputs") / "article_v1"
    )
    feature_benchmark.add_argument(
        "--run-id", default="article-v1-feature-index-v2"
    )
    feature_benchmark.add_argument(
        "--reference-safe-frontier-size",
        type=int,
        default=1024,
        help="current-host reference is required through F=1024; F=2048 is skipped",
    )
    feature_benchmark.add_argument(
        "--microbenchmark-repetitions", type=int, default=3
    )
    feature_benchmark.add_argument("--microbenchmark-warmups", type=int, default=1)
    feature_benchmark.add_argument(
        "--frontier-capture-expansion-limit", type=int, default=512
    )
    feature_benchmark.add_argument(
        "--correctness-timeout-seconds", type=float, default=300.0
    )
    feature_benchmark.add_argument("--maximum-hard-episode-seconds", type=float)
    feature_benchmark.add_argument(
        "--maximum-peak-index-memory-bytes", type=int
    )
    feature_benchmark.add_argument("--hard-training-episode-count", type=int)
    feature_benchmark.add_argument(
        "--no-profiles",
        action="store_true",
        help="skip cProfile runs; pytest/AST evidence remains in profiles/",
    )
    replay_capture = subparsers.add_parser(
        "capture-replay-checkpoint",
        description=(
            "Run only the frozen pilot hard/3q training workload and stop at "
            "the validated expansion-1024 recovery boundary."
        ),
    )
    replay_capture.add_argument(
        "--output-root", type=Path, default=Path("outputs") / "article_v1"
    )
    replay_capture.add_argument(
        "--run-id", default="article-v1-replay-capture-1024"
    )
    replay_capture.add_argument("--quiet", action="store_true")
    replay_capture.add_argument(
        "--checkpoint-every-expansions", type=int, default=64
    )
    replay_capture.add_argument("--checkpoint-every-seconds", type=float, default=60.0)
    replay_measure = subparsers.add_parser(
        "measure-replay-timing",
        description=(
            "Replay and validate an exact canonical expansion-1024 checkpoint, "
            "then write strict article-v1-replay-timing-v1 evidence."
        ),
    )
    replay_measure.add_argument("--checkpoint", type=Path, required=True)
    replay_measure.add_argument("--output", type=Path, required=True)
    replay_measure.add_argument(
        "--projected-full-episode-seconds", type=float, required=True
    )
    args = parser.parse_args(argv)

    if args.command == "capture-replay-checkpoint":
        result = capture_article_v1_replay_checkpoint(
            args.output_root / args.run_id,
            quiet=bool(args.quiet),
            checkpoint_cadence=CheckpointCadence(
                every_expansions=args.checkpoint_every_expansions,
                every_seconds=args.checkpoint_every_seconds,
            ),
        )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "measure-replay-timing":
        result = measure_article_v1_replay_checkpoint(
            args.checkpoint,
            args.output,
            projected_full_episode_seconds=args.projected_full_episode_seconds,
        )
        print(json.dumps(_json_ready(result.to_payload()), indent=2, sort_keys=True))
        return 0 if result.engineering_timing_valid else 1
    if args.command == "benchmark-features":
        result = benchmark_article_v1_features(
            args.config,
            output_root=args.output_root,
            run_id=args.run_id,
            reference_safe_frontier_size=args.reference_safe_frontier_size,
            microbenchmark_repetitions=args.microbenchmark_repetitions,
            microbenchmark_warmups=args.microbenchmark_warmups,
            frontier_capture_expansion_limit=args.frontier_capture_expansion_limit,
            correctness_timeout_seconds=args.correctness_timeout_seconds,
            maximum_hard_episode_seconds=args.maximum_hard_episode_seconds,
            maximum_peak_index_memory_bytes=args.maximum_peak_index_memory_bytes,
            hard_training_episode_count=args.hard_training_episode_count,
            write_profiles=not bool(args.no_profiles),
        )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0 if result["engineering_qualification_passed"] else 1
    if args.command == "mini-ci":
        result = mini_ci_benchmark(
            args.output_root,
            run_id=args.run_id,
            force_retrain=bool(args.force_retrain),
        )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "calibrate-certifier":
        from benchmarks.article_v1_calibration import calibrate_certifier

        path = args.output_root / args.run_id / "certifier_calibration.json"
        result = calibrate_certifier(args.config, path)
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "plan":
        print(json.dumps(_json_ready(campaign_plan(
            args.config,
            worker_count=args.workers,
            pilot_seconds_per_expansion=args.pilot_seconds_per_expansion,
        )), indent=2, sort_keys=True))
        return 0

    destination, corpus = initialize_run(
        args.config, output_root=args.output_root, run_id=args.run_id
    )
    if args.command == "generate-corpus":
        print(destination / "corpus_manifest" / "manifest.json")
        return 0

    experiment = corpus.config.experiment
    standard_checkpoint_scope = corpus.checkpoint_scope(
        checkpoint_family=STANDARD_CHECKPOINT_FAMILY
    )
    ood_checkpoint_scope = corpus.checkpoint_scope(
        checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY
    )
    checkpoint_paths: list[Path] = []
    ood_checkpoint_paths: list[Path] = []
    if args.command in {"pilot", "train"}:
        progress_reporter = ArticleV1ProgressReporter(
            destination,
            cadence=ProgressCadence(
                every_expansions=args.progress_every_expansions,
                every_seconds=args.progress_every_seconds,
            ),
            quiet=bool(args.quiet),
        )
        checkpoint_cadence = CheckpointCadence(
            every_expansions=args.checkpoint_every_expansions,
            every_seconds=args.checkpoint_every_seconds,
        )
        ood_limit = int(
            corpus.config.ood_length_split.training_max_generator_length
        )
        ood_training_cases = tuple(
            case
            for case in corpus.evaluation_targets(split="train")
            if case.generator_length <= ood_limit
        )
        for seed in experiment["training_seeds"]:
            resolved_seed = int(seed)
            path = destination / "checkpoints" / f"seed-{int(seed)}.json"
            checkpoint, _trained = _load_or_train_article_v1_checkpoint(
                path,
                scope=standard_checkpoint_scope,
                expected_training_seed=resolved_seed,
                force_retrain=bool(args.force_retrain),
                train_callback=lambda resolved_seed=resolved_seed: (
                    train_article_v1_checkpoint(
                        corpus.evaluation_targets(split="train"),
                        corpus_config_digest=corpus.config.digest,
                        training_seed=resolved_seed,
                        episodes_per_target=int(
                            experiment["training_episodes_per_target"]
                        ),
                        learning_rate=float(experiment["learning_rate"]),
                        epsilon_start=float(experiment["epsilon"]["start"]),
                        epsilon_minimum=float(experiment["epsilon"]["minimum"]),
                        epsilon_decay=float(experiment["epsilon"]["decay"]),
                        beta=float(experiment["beta"]),
                        certification_tolerance=float(
                            experiment["certification_tolerance"]
                        ),
                        training_checkpoint_dir=(
                            destination
                            / "training_state"
                            / f"standard-seed-{resolved_seed}"
                        ),
                        progress_reporter=progress_reporter,
                        checkpoint_cadence=checkpoint_cadence,
                        resume_training=(
                            not bool(args.force_retrain)
                            and not bool(args.no_training_resume)
                        ),
                        run_id=args.run_id,
                    )
                ),
            )
            checkpoint_paths.append(path)
            ood_path = destination / "checkpoints" / f"ood-seed-{int(seed)}.json"
            ood_checkpoint, _ood_trained = _load_or_train_article_v1_checkpoint(
                ood_path,
                scope=ood_checkpoint_scope,
                expected_training_seed=resolved_seed,
                force_retrain=bool(args.force_retrain),
                train_callback=lambda resolved_seed=resolved_seed: (
                    train_article_v1_checkpoint(
                        ood_training_cases,
                        corpus_config_digest=corpus.config.digest,
                        training_seed=resolved_seed,
                        episodes_per_target=int(
                            experiment["training_episodes_per_target"]
                        ),
                        learning_rate=float(experiment["learning_rate"]),
                        epsilon_start=float(experiment["epsilon"]["start"]),
                        epsilon_minimum=float(experiment["epsilon"]["minimum"]),
                        epsilon_decay=float(experiment["epsilon"]["decay"]),
                        beta=float(experiment["beta"]),
                        checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY,
                        certification_tolerance=float(
                            experiment["certification_tolerance"]
                        ),
                        training_checkpoint_dir=(
                            destination
                            / "training_state"
                            / f"ood-seed-{resolved_seed}"
                        ),
                        progress_reporter=progress_reporter,
                        checkpoint_cadence=checkpoint_cadence,
                        resume_training=(
                            not bool(args.force_retrain)
                            and not bool(args.no_training_resume)
                        ),
                        run_id=args.run_id,
                    )
                ),
            )
            ood_checkpoint_paths.append(ood_path)
        checkpoints_for_validation = tuple(
            ArticleV1Checkpoint.load(path) for path in checkpoint_paths
        )
        validate_article_v1_checkpoints(
            corpus.evaluation_targets(split="validation"),
            checkpoints=checkpoints_for_validation,
            checkpoint_scope=standard_checkpoint_scope,
            output_path=destination / "validation_audit.json",
            beta=float(experiment["beta"]),
            certification_tolerance=float(experiment["certification_tolerance"]),
        )
        ood_limit = int(
            corpus.config.ood_length_split.training_max_generator_length
        )
        ood_validation_cases = tuple(
            case
            for case in corpus.evaluation_targets(split="validation")
            if case.generator_length <= ood_limit
        )
        validate_article_v1_checkpoints(
            ood_validation_cases,
            checkpoints=tuple(
                ArticleV1Checkpoint.load(path) for path in ood_checkpoint_paths
            ),
            checkpoint_scope=ood_checkpoint_scope,
            output_path=destination / "ood_validation_audit.json",
            beta=float(experiment["beta"]),
            certification_tolerance=float(experiment["certification_tolerance"]),
        )
        if args.command == "train":
            return 0

    checkpoint_paths = checkpoint_paths or sorted((destination / "checkpoints").glob("seed-*.json"))
    checkpoint_paths = [path for path in checkpoint_paths if not path.name.startswith("ood-")]
    ood_checkpoint_paths = ood_checkpoint_paths or sorted(
        (destination / "checkpoints").glob("ood-seed-*.json")
    )
    checkpoints = tuple(ArticleV1Checkpoint.load(path) for path in checkpoint_paths)
    ood_checkpoints = tuple(
        ArticleV1Checkpoint.load(path) for path in ood_checkpoint_paths
    )
    if args.command == "audit":
        result = audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=destination / "raw_runs.jsonl",
            output_path=destination / "campaign_audit.json",
        )
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
        return 0
    if args.command == "ablations":
        from experiments.article_v1_ablations import run_article_v1_ablations

        result = run_article_v1_ablations(
            corpus,
            output_csv=destination / "ablations.csv",
            checkpoint_output_dir=destination / "checkpoints" / "ablations",
            primary_checkpoints=checkpoints,
        )
        _atomic_json(destination / "ablation_results.json", result)
        return 0
    if args.command in {"pilot", "evaluate"}:
        evaluate_article_v1_matrix(
            corpus.evaluation_targets(split="test"),
            config=corpus.config,
            checkpoints=checkpoints,
            checkpoint_scope=standard_checkpoint_scope,
            raw_path=destination / "raw_runs.jsonl",
        )
        evaluate_article_v1_matrix(
            corpus.evaluation_targets(split="ood_test"),
            config=corpus.config,
            checkpoints=ood_checkpoints,
            checkpoint_scope=ood_checkpoint_scope,
            raw_path=destination / "raw_runs.jsonl",
        )
        if args.command == "evaluate":
            return 0

    if args.command in {"pilot", "aggregate"}:
        audit_result = audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=destination / "raw_runs.jsonl",
            output_path=destination / "campaign_audit.json",
        )
        if audit_result.get("passed") is not True:
            raise ValueError("campaign audit did not report passed=true")
        audited_raw_sha256 = audit_result.get("raw_ledger_sha256")
        if not isinstance(audited_raw_sha256, str):
            raise ValueError("campaign audit did not bind the raw ledger digest")
        write_article_v1_report(
            destination / "raw_runs.jsonl",
            destination,
            stats_seed=int(experiment["statistics_seed"]),
            expected_raw_sha256=audited_raw_sha256,
        )
        return 0
    return 0


__all__ = [
    "ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA",
    "ARTICLE_V1_CHECKPOINT_SCHEMA",
    "ARTICLE_V1_REPLAY_CAPTURE_SCHEMA",
    "ARTICLE_V1_RUNNER_SCHEMA",
    "PRIMARY_SCHEDULERS",
    "ArticleV1Checkpoint",
    "audit_article_v1_campaign",
    "benchmark_article_v1_features",
    "capture_article_v1_replay_checkpoint",
    "environment_metadata",
    "evaluate_article_v1_matrix",
    "evaluate_article_v1_run",
    "git_provenance",
    "initialize_run",
    "main",
    "measure_article_v1_replay_checkpoint",
    "mini_ci_benchmark",
    "campaign_plan",
    "train_article_v1_checkpoint",
    "validate_article_v1_checkpoints",
]
