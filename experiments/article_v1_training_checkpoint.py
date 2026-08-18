"""Portable, event-sourced Article V1 training recovery checkpoints.

This module is intentionally independent of the trainer and environment.  It
defines the durable contract and validation hooks; integration code remains
responsible for taking snapshots only at a completed SARSA decision boundary
and for replaying selected persistent record IDs through normal environment
transitions.  Raw pickle is neither produced nor accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


ARTICLE_V1_TRAINING_PROGRESS_SCHEMA = "article-v1-training-progress-v1"
ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA = (
    "article-v1-mid-episode-replay-checkpoint-v2"
)
ARTICLE_V1_EVENT_JOURNAL_SCHEMA = "article-v1-training-event-journal-v2"
ARTICLE_V1_EVENT_SCHEMA = "article-v1-training-expansion-event-v2"
ARTICLE_V1_CHECKPOINT_MANIFEST_SCHEMA = "article-v1-training-checkpoint-manifest-v1"

INTERNAL_TRAINING_CHECKPOINT_SCHEMAS = frozenset(
    {
        ARTICLE_V1_TRAINING_PROGRESS_SCHEMA,
        ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA,
    }
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class CheckpointFormatError(ValueError):
    """Raised for corrupt, partial, non-portable, or incoherent checkpoints."""


class CheckpointCompatibilityError(CheckpointFormatError):
    """Raised when a valid checkpoint does not match the requested resume run."""


class IncompleteTrainingCheckpointError(CheckpointFormatError):
    """Raised when an internal recovery checkpoint is offered for evaluation."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if not _is_int(value) or int(value) < minimum:
        raise CheckpointFormatError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointFormatError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointFormatError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise CheckpointFormatError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CheckpointFormatError(f"{name} must be <= {maximum}")
    return result


def _string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointFormatError(f"{name} must be a nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    result = _string(name, value)
    if _SHA256_RE.fullmatch(result) is None:
        raise CheckpointFormatError(f"{name} must be a canonical sha256 digest")
    return result


def _freeze_json(value: object, *, path: str = "$") -> object:
    """Validate and recursively freeze the strict portable-JSON value subset."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if _is_int(value):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointFormatError(f"{path} contains a non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointFormatError(f"{path} contains a non-string object key")
            if key in frozen:
                raise CheckpointFormatError(f"{path} contains duplicate key {key!r}")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise CheckpointFormatError(
        f"{path} contains non-portable value of type {type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    portable = _thaw_json(_freeze_json(value))
    try:
        return (
            json.dumps(
                portable,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # defensive after strict normalization
        raise CheckpointFormatError("value is not strict portable JSON") from error


def portable_digest(value: object, *, domain: str) -> str:
    """Domain-separated digest over canonical portable JSON (without final LF)."""

    _string("digest domain", domain)
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(value).rstrip(b"\n"))
    return f"sha256:{digest.hexdigest()}"


def feature_row_digest(values: Sequence[float]) -> str:
    row = tuple(_number("feature row value", value) for value in values)
    if not row:
        raise CheckpointFormatError("pending feature row must be nonempty")
    return portable_digest(row, domain="article-v1-frozen-feature-row-v1")


def policy_weight_digest(values: Sequence[float]) -> str:
    weights = tuple(_number("policy weight", value) for value in values)
    if not weights:
        raise CheckpointFormatError("policy weights must be nonempty")
    return portable_digest(weights, domain="article-v1-recovery-policy-weights-v1")


@dataclass(frozen=True, slots=True)
class ArticleV1CheckpointProvenance:
    """All scientific/source bindings that must match before deterministic replay."""

    source_commit_sha: str
    source_worktree_digest: str
    config_digest: str
    corpus_digest: str
    profile_digest: str
    target_id: str
    target_fingerprint: str
    feature_schema_version: str
    feature_evaluator_schema_version: str
    reward_schema_version: str
    certifier_schema_version: str

    def __post_init__(self) -> None:
        _string("source_commit_sha", self.source_commit_sha)
        for name in (
            "source_worktree_digest",
            "config_digest",
            "corpus_digest",
            "profile_digest",
            "target_fingerprint",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "target_id",
            "feature_schema_version",
            "feature_evaluator_schema_version",
            "reward_schema_version",
            "certifier_schema_version",
        ):
            _string(name, getattr(self, name))

    def to_payload(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "ArticleV1CheckpointProvenance":
        _require_exact_members(payload, {field.name for field in fields(cls)}, "provenance")
        return cls(**dict(payload))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ArticleV1JournalEntry:
    """One deterministic post-update expansion record; timing is forbidden."""

    expansion_index: int
    selected_record_id: int
    selected_feature_digest: str
    reward: float
    terminated: bool
    truncated: bool
    frontier_revision: int
    state_digest_verified: bool
    frontier_active_ids_digest: str | None
    archive_digest: str | None
    generation_count_digest: str | None
    policy_weight_digest_after_update: str
    pending_next_record_id: int | None
    schema_version: str = ARTICLE_V1_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_EVENT_SCHEMA:
            raise CheckpointFormatError("unsupported training-expansion event schema")
        _integer("expansion_index", self.expansion_index, minimum=1)
        _integer("selected_record_id", self.selected_record_id)
        _digest("selected_feature_digest", self.selected_feature_digest)
        _number("reward", self.reward)
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise CheckpointFormatError("terminated and truncated must be booleans")
        if self.terminated and self.truncated:
            raise CheckpointFormatError("an expansion cannot terminate and truncate together")
        _integer("frontier_revision", self.frontier_revision)
        if type(self.state_digest_verified) is not bool:
            raise CheckpointFormatError("state_digest_verified must be a boolean")
        state_digest_names = (
            "frontier_active_ids_digest",
            "archive_digest",
            "generation_count_digest",
        )
        if self.state_digest_verified:
            for name in state_digest_names:
                _digest(name, getattr(self, name))
        elif any(getattr(self, name) is not None for name in state_digest_names):
            raise CheckpointFormatError(
                "unverified journal entry must omit all full-state digests"
            )
        _digest(
            "policy_weight_digest_after_update",
            self.policy_weight_digest_after_update,
        )
        if self.pending_next_record_id is not None:
            _integer("pending_next_record_id", self.pending_next_record_id)
        if (self.terminated or self.truncated) and self.pending_next_record_id is not None:
            raise CheckpointFormatError(
                "terminal/truncated journal entry cannot have a pending next record"
            )
        if not (self.terminated or self.truncated) and self.pending_next_record_id is None:
            raise CheckpointFormatError(
                "nonterminal journal entry must bind its pending next record"
            )

    def to_payload(self) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "schema_version"
        }
        result["journal_entry_schema"] = self.schema_version
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ArticleV1JournalEntry":
        members = {field.name for field in fields(cls) if field.name != "schema_version"}
        members.add("journal_entry_schema")
        _require_exact_members(payload, members, "journal entry")
        values = dict(payload)
        values["schema_version"] = values.pop("journal_entry_schema")
        return cls(**values)  # type: ignore[arg-type]


class ArticleV1EventJournal:
    """Append-only in-memory journal with deterministic portable serialization."""

    def __init__(
        self,
        entries: Iterable[ArticleV1JournalEntry] = (),
        *,
        base_expansion: int = 0,
        _sealed: bool = False,
    ) -> None:
        self.base_expansion = _integer("base_expansion", base_expansion)
        self._entries: list[ArticleV1JournalEntry] = []
        self._sealed = False
        for entry in entries:
            self.append(entry)
        self._sealed = bool(_sealed)

    @property
    def entries(self) -> tuple[ArticleV1JournalEntry, ...]:
        return tuple(self._entries)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ArticleV1EventJournal)
            and self.base_expansion == other.base_expansion
            and self.entries == other.entries
        )

    def __repr__(self) -> str:
        return (
            f"ArticleV1EventJournal(base_expansion={self.base_expansion}, "
            f"entries={self.entries!r})"
        )

    @property
    def expansion_count(self) -> int:
        return self.base_expansion + len(self._entries)

    def append(self, entry: ArticleV1JournalEntry) -> None:
        if self._sealed:
            raise CheckpointFormatError("cannot mutate a sealed checkpoint journal")
        if not isinstance(entry, ArticleV1JournalEntry):
            raise TypeError("journal accepts only ArticleV1JournalEntry values")
        expected = self.expansion_count + 1
        if entry.expansion_index != expected:
            raise CheckpointFormatError(
                f"journal expansion sequence mismatch: expected {expected}, "
                f"found {entry.expansion_index}"
            )
        if self._entries and (
            self._entries[-1].terminated or self._entries[-1].truncated
        ):
            raise CheckpointFormatError("cannot append after a terminal journal entry")
        self._entries.append(entry)

    def bind_latest_state_digests(
        self,
        *,
        frontier_active_ids_digest: str,
        archive_digest: str,
        generation_count_digest: str,
    ) -> ArticleV1JournalEntry:
        """Bind a full-state verification point to the latest expansion.

        This is used immediately before publishing a mid-episode checkpoint when
        an asynchronous interrupt lands between ordinary cadence boundaries.
        Earlier entries remain immutable values and the journal remains ordered.
        """

        if self._sealed:
            raise CheckpointFormatError("cannot mutate a sealed checkpoint journal")
        if not self._entries:
            raise CheckpointFormatError("cannot bind digests to an empty journal")
        latest = self._entries[-1]
        replacement = replace(
            latest,
            state_digest_verified=True,
            frontier_active_ids_digest=_digest(
                "frontier_active_ids_digest", frontier_active_ids_digest
            ),
            archive_digest=_digest("archive_digest", archive_digest),
            generation_count_digest=_digest(
                "generation_count_digest", generation_count_digest
            ),
        )
        self._entries[-1] = replacement
        return replacement

    def frozen_copy(self) -> "ArticleV1EventJournal":
        return ArticleV1EventJournal(
            self._entries, base_expansion=self.base_expansion, _sealed=True
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "event_journal_schema": ARTICLE_V1_EVENT_JOURNAL_SCHEMA,
            "base_expansion": self.base_expansion,
            "entries": [entry.to_payload() for entry in self._entries],
        }

    @property
    def digest(self) -> str:
        return portable_digest(
            self.to_payload(), domain="article-v1-training-event-journal-digest-v2"
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ArticleV1EventJournal":
        _require_exact_members(
            payload,
            {"event_journal_schema", "base_expansion", "entries"},
            "event journal",
        )
        if payload["event_journal_schema"] != ARTICLE_V1_EVENT_JOURNAL_SCHEMA:
            raise CheckpointFormatError("unsupported training event-journal schema")
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise CheckpointFormatError("event journal entries must be a list")
        return cls(
            (ArticleV1JournalEntry.from_payload(entry) for entry in entries),
            base_expansion=_integer("base_expansion", payload["base_expansion"]),
        )


def _float_tuple(name: str, values: object, *, nonempty: bool = True) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise CheckpointFormatError(f"{name} must be a list")
    result = tuple(_number(f"{name} value", value) for value in values)
    if nonempty and not result:
        raise CheckpointFormatError(f"{name} must be nonempty")
    return result


@dataclass(frozen=True, slots=True)
class MidEpisodeCheckpoint:
    """Portable safe-boundary state restored after deterministic journal replay."""

    provenance: ArticleV1CheckpointProvenance
    training_seed: int
    episode_index: int
    episode_count: int
    expansion_count: int
    expansion_cap: int
    journal: ArticleV1EventJournal
    episode_initial_theta: tuple[float, ...]
    theta: tuple[float, ...]
    epsilon: float
    policy_rng_state: Mapping[str, object]
    environment_rng_state: Mapping[str, object]
    pending_next_record_id: int
    pending_next_feature_row: tuple[float, ...]
    total_reward: float
    training_aggregates: Mapping[str, object]
    search_metrics: Mapping[str, object]
    frontier_revision: int
    frontier_active_ids_digest: str
    archive_digest: str
    generation_count_digest: str
    schema_version: str = ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA:
            raise CheckpointFormatError("unsupported mid-episode checkpoint schema")
        if not isinstance(self.provenance, ArticleV1CheckpointProvenance):
            raise CheckpointFormatError("checkpoint provenance has the wrong type")
        _integer("training_seed", self.training_seed)
        episode_index = _integer("episode_index", self.episode_index)
        episode_count = _integer("episode_count", self.episode_count, minimum=1)
        if episode_index >= episode_count:
            raise CheckpointFormatError("episode_index must be smaller than episode_count")
        expansion_count = _integer("expansion_count", self.expansion_count)
        expansion_cap = _integer("expansion_cap", self.expansion_cap, minimum=1)
        if expansion_count >= expansion_cap:
            raise CheckpointFormatError(
                "mid-episode expansion_count must be below expansion_cap"
            )
        if not isinstance(self.journal, ArticleV1EventJournal):
            raise CheckpointFormatError("journal has the wrong type")
        object.__setattr__(self, "journal", self.journal.frozen_copy())
        if self.journal.expansion_count != expansion_count:
            raise CheckpointFormatError("journal does not end at expansion_count")
        initial_theta = _float_tuple("episode_initial_theta", self.episode_initial_theta)
        theta = _float_tuple("theta", self.theta)
        if len(initial_theta) != len(theta):
            raise CheckpointFormatError(
                "episode-initial and current policy dimensions disagree"
            )
        object.__setattr__(self, "episode_initial_theta", initial_theta)
        object.__setattr__(self, "theta", theta)
        _number("epsilon", self.epsilon, minimum=0.0, maximum=1.0)
        object.__setattr__(
            self, "policy_rng_state", _freeze_json(self.policy_rng_state, path="policy_rng_state")
        )
        object.__setattr__(
            self,
            "environment_rng_state",
            _freeze_json(self.environment_rng_state, path="environment_rng_state"),
        )
        _integer("pending_next_record_id", self.pending_next_record_id)
        pending = _float_tuple("pending_next_feature_row", self.pending_next_feature_row)
        object.__setattr__(self, "pending_next_feature_row", pending)
        _number("total_reward", self.total_reward)
        object.__setattr__(
            self,
            "training_aggregates",
            _freeze_json(self.training_aggregates, path="training_aggregates"),
        )
        object.__setattr__(
            self, "search_metrics", _freeze_json(self.search_metrics, path="search_metrics")
        )
        _integer("frontier_revision", self.frontier_revision)
        for name in (
            "frontier_active_ids_digest",
            "archive_digest",
            "generation_count_digest",
        ):
            _digest(name, getattr(self, name))
        if self.journal.entries:
            final = self.journal.entries[-1]
            if final.terminated or final.truncated:
                raise CheckpointFormatError(
                    "mid-episode checkpoint cannot end at a terminal journal entry"
                )
            if not final.state_digest_verified:
                raise CheckpointFormatError(
                    "mid-episode checkpoint must end at a full-state digest boundary"
                )
            checks = {
                "pending next record": (
                    final.pending_next_record_id,
                    self.pending_next_record_id,
                ),
                "frontier revision": (final.frontier_revision, self.frontier_revision),
                "frontier digest": (
                    final.frontier_active_ids_digest,
                    self.frontier_active_ids_digest,
                ),
                "archive digest": (final.archive_digest, self.archive_digest),
                "generation digest": (
                    final.generation_count_digest,
                    self.generation_count_digest,
                ),
                "policy weight digest": (
                    final.policy_weight_digest_after_update,
                    policy_weight_digest(theta),
                ),
            }
            for label, (journal_value, checkpoint_value) in checks.items():
                if journal_value != checkpoint_value:
                    raise CheckpointFormatError(
                        f"journal/checkpoint {label} mismatch"
                    )

    @property
    def journal_digest(self) -> str:
        return self.journal.digest

    @property
    def weight_digest(self) -> str:
        return policy_weight_digest(self.theta)

    @property
    def pending_feature_digest(self) -> str:
        return feature_row_digest(self.pending_next_feature_row)

    def require_evaluation_eligible(self) -> None:
        raise IncompleteTrainingCheckpointError(
            "mid-episode recovery checkpoint is incomplete and not transferable "
            "for evaluation"
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "checkpoint_schema": self.schema_version,
            "checkpoint_kind": "internal-mid-episode-replay",
            "safe_sarsa_boundary": True,
            "training_complete": False,
            "transferable_for_evaluation": False,
            "provenance": self.provenance.to_payload(),
            "training_seed": self.training_seed,
            "episode_index": self.episode_index,
            "episode_count": self.episode_count,
            "expansion_count": self.expansion_count,
            "expansion_cap": self.expansion_cap,
            "journal": self.journal.to_payload(),
            "journal_digest": self.journal_digest,
            "episode_initial_theta": list(self.episode_initial_theta),
            "episode_initial_weight_digest": policy_weight_digest(
                self.episode_initial_theta
            ),
            "theta": list(self.theta),
            "policy_weight_digest": self.weight_digest,
            "epsilon": self.epsilon,
            "policy_rng_state": _thaw_json(self.policy_rng_state),
            "environment_rng_state": _thaw_json(self.environment_rng_state),
            "pending_next_record_id": self.pending_next_record_id,
            "pending_next_feature_row": list(self.pending_next_feature_row),
            "pending_next_feature_digest": self.pending_feature_digest,
            "total_reward": self.total_reward,
            "training_aggregates": _thaw_json(self.training_aggregates),
            "search_metrics": _thaw_json(self.search_metrics),
            "frontier_revision": self.frontier_revision,
            "frontier_active_ids_digest": self.frontier_active_ids_digest,
            "archive_digest": self.archive_digest,
            "generation_count_digest": self.generation_count_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MidEpisodeCheckpoint":
        _require_exact_members(payload, _MID_EPISODE_MEMBERS, "mid-episode checkpoint")
        if payload["checkpoint_schema"] != ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA:
            raise CheckpointFormatError("unsupported mid-episode checkpoint schema")
        if (
            payload["checkpoint_kind"] != "internal-mid-episode-replay"
            or payload["safe_sarsa_boundary"] is not True
            or payload["training_complete"] is not False
            or payload["transferable_for_evaluation"] is not False
        ):
            raise CheckpointFormatError("invalid mid-episode checkpoint safety markers")
        provenance = payload["provenance"]
        journal_payload = payload["journal"]
        if not isinstance(provenance, Mapping) or not isinstance(journal_payload, Mapping):
            raise CheckpointFormatError("checkpoint provenance/journal must be objects")
        journal = ArticleV1EventJournal.from_payload(journal_payload)
        if payload["journal_digest"] != journal.digest:
            raise CheckpointFormatError("checkpoint journal digest mismatch")
        initial_theta = _float_tuple(
            "episode_initial_theta", payload["episode_initial_theta"]
        )
        if payload["episode_initial_weight_digest"] != policy_weight_digest(
            initial_theta
        ):
            raise CheckpointFormatError(
                "checkpoint episode-initial weight digest mismatch"
            )
        theta = _float_tuple("theta", payload["theta"])
        if payload["policy_weight_digest"] != policy_weight_digest(theta):
            raise CheckpointFormatError("checkpoint policy-weight digest mismatch")
        feature_row = _float_tuple(
            "pending_next_feature_row", payload["pending_next_feature_row"]
        )
        if payload["pending_next_feature_digest"] != feature_row_digest(feature_row):
            raise CheckpointFormatError("checkpoint pending-feature digest mismatch")
        policy_rng = payload["policy_rng_state"]
        environment_rng = payload["environment_rng_state"]
        aggregates = payload["training_aggregates"]
        metrics = payload["search_metrics"]
        for name, value in (
            ("policy_rng_state", policy_rng),
            ("environment_rng_state", environment_rng),
            ("training_aggregates", aggregates),
            ("search_metrics", metrics),
        ):
            if not isinstance(value, Mapping):
                raise CheckpointFormatError(f"{name} must be an object")
        return cls(
            provenance=ArticleV1CheckpointProvenance.from_payload(provenance),
            training_seed=payload["training_seed"],  # type: ignore[arg-type]
            episode_index=payload["episode_index"],  # type: ignore[arg-type]
            episode_count=payload["episode_count"],  # type: ignore[arg-type]
            expansion_count=payload["expansion_count"],  # type: ignore[arg-type]
            expansion_cap=payload["expansion_cap"],  # type: ignore[arg-type]
            journal=journal,
            episode_initial_theta=initial_theta,
            theta=theta,
            epsilon=payload["epsilon"],  # type: ignore[arg-type]
            policy_rng_state=policy_rng,
            environment_rng_state=environment_rng,
            pending_next_record_id=payload["pending_next_record_id"],  # type: ignore[arg-type]
            pending_next_feature_row=feature_row,
            total_reward=payload["total_reward"],  # type: ignore[arg-type]
            training_aggregates=aggregates,
            search_metrics=metrics,
            frontier_revision=payload["frontier_revision"],  # type: ignore[arg-type]
            frontier_active_ids_digest=payload["frontier_active_ids_digest"],  # type: ignore[arg-type]
            archive_digest=payload["archive_digest"],  # type: ignore[arg-type]
            generation_count_digest=payload["generation_count_digest"],  # type: ignore[arg-type]
        )


_MID_EPISODE_MEMBERS = {
    "checkpoint_schema",
    "checkpoint_kind",
    "safe_sarsa_boundary",
    "training_complete",
    "transferable_for_evaluation",
    "provenance",
    "training_seed",
    "episode_index",
    "episode_count",
    "expansion_count",
    "expansion_cap",
    "journal",
    "journal_digest",
    "episode_initial_theta",
    "episode_initial_weight_digest",
    "theta",
    "policy_weight_digest",
    "epsilon",
    "policy_rng_state",
    "environment_rng_state",
    "pending_next_record_id",
    "pending_next_feature_row",
    "pending_next_feature_digest",
    "total_reward",
    "training_aggregates",
    "search_metrics",
    "frontier_revision",
    "frontier_active_ids_digest",
    "archive_digest",
    "generation_count_digest",
}


@dataclass(frozen=True, slots=True)
class TrainingProgressCheckpoint:
    """Episode/target cursor state; always internal even after the final target."""

    provenance: ArticleV1CheckpointProvenance
    training_seed: int
    target_cursor: int
    target_count: int
    episode_cursor: int
    episodes_per_target: int
    theta: tuple[float, ...]
    epsilon: float
    policy_rng_state: Mapping[str, object]
    training_history: tuple[object, ...]
    completed_target_ids: tuple[str, ...]
    effective_budgets: Mapping[str, object]
    schema_version: str = ARTICLE_V1_TRAINING_PROGRESS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_TRAINING_PROGRESS_SCHEMA:
            raise CheckpointFormatError("unsupported training-progress checkpoint schema")
        if not isinstance(self.provenance, ArticleV1CheckpointProvenance):
            raise CheckpointFormatError("checkpoint provenance has the wrong type")
        _integer("training_seed", self.training_seed)
        target_cursor = _integer("target_cursor", self.target_cursor)
        target_count = _integer("target_count", self.target_count, minimum=1)
        if target_cursor > target_count:
            raise CheckpointFormatError("target_cursor must not exceed target_count")
        episode_cursor = _integer("episode_cursor", self.episode_cursor)
        episode_count = _integer(
            "episodes_per_target", self.episodes_per_target, minimum=1
        )
        if episode_cursor >= episode_count:
            raise CheckpointFormatError(
                "episode_cursor must identify the next episode and be below the count"
            )
        if target_cursor == target_count and episode_cursor != 0:
            raise CheckpointFormatError(
                "completed target cursor must reset the episode cursor to zero"
            )
        theta = _float_tuple("theta", self.theta)
        object.__setattr__(self, "theta", theta)
        _number("epsilon", self.epsilon, minimum=0.0, maximum=1.0)
        object.__setattr__(
            self, "policy_rng_state", _freeze_json(self.policy_rng_state, path="policy_rng_state")
        )
        history = _freeze_json(self.training_history, path="training_history")
        if not isinstance(history, tuple):
            raise CheckpointFormatError("training_history must be a list")
        object.__setattr__(self, "training_history", history)
        completed = tuple(
            _string("completed target ID", target_id)
            for target_id in self.completed_target_ids
        )
        if len(set(completed)) != len(completed):
            raise CheckpointFormatError("completed target IDs must be unique")
        if len(completed) != target_cursor:
            raise CheckpointFormatError(
                "completed target IDs must have exactly target_cursor entries"
            )
        object.__setattr__(self, "completed_target_ids", completed)
        budgets = _freeze_json(self.effective_budgets, path="effective_budgets")
        if not isinstance(budgets, Mapping) or not budgets:
            raise CheckpointFormatError("effective_budgets must be a nonempty object")
        for target_id, budget in budgets.items():
            _string("effective budget target ID", target_id)
            _integer("effective expansion budget", budget, minimum=1)
        object.__setattr__(self, "effective_budgets", budgets)

    @property
    def all_targets_completed(self) -> bool:
        return self.target_cursor == self.target_count

    @property
    def weight_digest(self) -> str:
        return policy_weight_digest(self.theta)

    def require_evaluation_eligible(self) -> None:
        raise IncompleteTrainingCheckpointError(
            "training-progress recovery checkpoint is internal and not transferable "
            "for evaluation; produce the final scope-validated checkpoint"
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "checkpoint_schema": self.schema_version,
            "checkpoint_kind": "internal-training-progress",
            "safe_sarsa_boundary": True,
            "training_complete": self.all_targets_completed,
            "transferable_for_evaluation": False,
            "provenance": self.provenance.to_payload(),
            "training_seed": self.training_seed,
            "target_cursor": self.target_cursor,
            "target_count": self.target_count,
            "episode_cursor": self.episode_cursor,
            "episodes_per_target": self.episodes_per_target,
            "theta": list(self.theta),
            "policy_weight_digest": self.weight_digest,
            "epsilon": self.epsilon,
            "policy_rng_state": _thaw_json(self.policy_rng_state),
            "training_history": _thaw_json(self.training_history),
            "completed_target_ids": list(self.completed_target_ids),
            "effective_budgets": _thaw_json(self.effective_budgets),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TrainingProgressCheckpoint":
        _require_exact_members(
            payload, _TRAINING_PROGRESS_MEMBERS, "training-progress checkpoint"
        )
        if payload["checkpoint_schema"] != ARTICLE_V1_TRAINING_PROGRESS_SCHEMA:
            raise CheckpointFormatError("unsupported training-progress checkpoint schema")
        if (
            payload["checkpoint_kind"] != "internal-training-progress"
            or payload["safe_sarsa_boundary"] is not True
            or type(payload["training_complete"]) is not bool
            or payload["transferable_for_evaluation"] is not False
        ):
            raise CheckpointFormatError("invalid training-progress safety markers")
        provenance = payload["provenance"]
        policy_rng = payload["policy_rng_state"]
        history = payload["training_history"]
        completed = payload["completed_target_ids"]
        budgets = payload["effective_budgets"]
        if not isinstance(provenance, Mapping):
            raise CheckpointFormatError("checkpoint provenance must be an object")
        if not isinstance(policy_rng, Mapping):
            raise CheckpointFormatError("policy_rng_state must be an object")
        if not isinstance(history, list):
            raise CheckpointFormatError("training_history must be a list")
        if not isinstance(completed, list):
            raise CheckpointFormatError("completed_target_ids must be a list")
        if not isinstance(budgets, Mapping):
            raise CheckpointFormatError("effective_budgets must be an object")
        theta = _float_tuple("theta", payload["theta"])
        if payload["policy_weight_digest"] != policy_weight_digest(theta):
            raise CheckpointFormatError("checkpoint policy-weight digest mismatch")
        checkpoint = cls(
            provenance=ArticleV1CheckpointProvenance.from_payload(provenance),
            training_seed=payload["training_seed"],  # type: ignore[arg-type]
            target_cursor=payload["target_cursor"],  # type: ignore[arg-type]
            target_count=payload["target_count"],  # type: ignore[arg-type]
            episode_cursor=payload["episode_cursor"],  # type: ignore[arg-type]
            episodes_per_target=payload["episodes_per_target"],  # type: ignore[arg-type]
            theta=theta,
            epsilon=payload["epsilon"],  # type: ignore[arg-type]
            policy_rng_state=policy_rng,
            training_history=tuple(history),
            completed_target_ids=tuple(completed),  # type: ignore[arg-type]
            effective_budgets=budgets,
        )
        if payload["training_complete"] != checkpoint.all_targets_completed:
            raise CheckpointFormatError("training_complete marker disagrees with cursor")
        return checkpoint


_TRAINING_PROGRESS_MEMBERS = {
    "checkpoint_schema",
    "checkpoint_kind",
    "safe_sarsa_boundary",
    "training_complete",
    "transferable_for_evaluation",
    "provenance",
    "training_seed",
    "target_cursor",
    "target_count",
    "episode_cursor",
    "episodes_per_target",
    "theta",
    "policy_weight_digest",
    "epsilon",
    "policy_rng_state",
    "training_history",
    "completed_target_ids",
    "effective_budgets",
}


TrainingRecoveryCheckpoint = MidEpisodeCheckpoint | TrainingProgressCheckpoint


def checkpoint_from_payload(payload: Mapping[str, object]) -> TrainingRecoveryCheckpoint:
    schema = payload.get("checkpoint_schema") if isinstance(payload, Mapping) else None
    if schema == ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA:
        return MidEpisodeCheckpoint.from_payload(payload)
    if schema == ARTICLE_V1_TRAINING_PROGRESS_SCHEMA:
        return TrainingProgressCheckpoint.from_payload(payload)
    raise CheckpointFormatError(f"unsupported internal checkpoint schema {schema!r}")


def reject_internal_checkpoint_for_evaluation(
    checkpoint: TrainingRecoveryCheckpoint | Mapping[str, object],
) -> None:
    """Fail before an internal progress artifact can reach evaluation code."""

    if isinstance(checkpoint, (MidEpisodeCheckpoint, TrainingProgressCheckpoint)):
        checkpoint.require_evaluation_eligible()
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be an internal checkpoint or JSON object")
    schema = checkpoint.get("checkpoint_schema")
    internal_schema_prefixes = (
        "article-v1-mid-episode-replay-checkpoint-",
        "article-v1-training-progress-",
    )
    if schema in INTERNAL_TRAINING_CHECKPOINT_SCHEMAS or (
        isinstance(schema, str)
        and schema.startswith(internal_schema_prefixes)
    ):
        raise IncompleteTrainingCheckpointError(
            f"internal checkpoint schema {schema!r} is not transferable for evaluation"
        )


@dataclass(frozen=True, slots=True)
class ResumeExpectation:
    """Exact run identity required before accepting a mid-episode checkpoint."""

    provenance: ArticleV1CheckpointProvenance
    training_seed: int
    episode_index: int
    episode_count: int
    expansion_cap: int
    feature_dimension: int
    checkpoint_schema: str = ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.checkpoint_schema != ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA:
            raise CheckpointCompatibilityError("resume expects the wrong checkpoint schema")
        _integer("training_seed", self.training_seed)
        episode_index = _integer("episode_index", self.episode_index)
        episode_count = _integer("episode_count", self.episode_count, minimum=1)
        if episode_index >= episode_count:
            raise CheckpointCompatibilityError("invalid expected episode cursor")
        _integer("expansion_cap", self.expansion_cap, minimum=1)
        _integer("feature_dimension", self.feature_dimension, minimum=1)


def validate_resume_compatibility(
    checkpoint: MidEpisodeCheckpoint,
    expected: ResumeExpectation,
) -> None:
    """Compare every schema/provenance/seed/episode/budget binding fail-closed."""

    if checkpoint.schema_version != expected.checkpoint_schema:
        raise CheckpointCompatibilityError("checkpoint schema mismatch")
    for field in fields(ArticleV1CheckpointProvenance):
        actual_value = getattr(checkpoint.provenance, field.name)
        expected_value = getattr(expected.provenance, field.name)
        if actual_value != expected_value:
            raise CheckpointCompatibilityError(
                f"checkpoint provenance {field.name} mismatch"
            )
    comparisons = {
        "training seed": (checkpoint.training_seed, expected.training_seed),
        "episode index": (checkpoint.episode_index, expected.episode_index),
        "episode count": (checkpoint.episode_count, expected.episode_count),
        "expansion cap": (checkpoint.expansion_cap, expected.expansion_cap),
        "feature dimension": (
            len(checkpoint.pending_next_feature_row),
            expected.feature_dimension,
        ),
        "weight dimension": (len(checkpoint.theta), expected.feature_dimension),
    }
    for label, (actual, wanted) in comparisons.items():
        if actual != wanted:
            raise CheckpointCompatibilityError(f"checkpoint {label} mismatch")
    # Accessing and recomputing the digest is intentional even though construction
    # already validates it; callers may have crossed a serialization boundary.
    if checkpoint.journal_digest != checkpoint.journal.digest:
        raise CheckpointCompatibilityError("checkpoint journal digest mismatch")


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """Observed deterministic post-transition values used to validate one replay."""

    expansion_index: int
    selected_record_id: int
    selected_feature_digest: str
    reward: float
    terminated: bool
    truncated: bool
    frontier_revision: int
    state_digest_verified: bool
    frontier_active_ids_digest: str | None
    archive_digest: str | None
    generation_count_digest: str | None
    policy_weight_digest_after_update: str
    pending_next_record_id: int | None

    def __post_init__(self) -> None:
        # Reuse the journal entry's complete strict validation contract.
        ArticleV1JournalEntry(**{
            field.name: getattr(self, field.name) for field in fields(self)
        })


def validate_replay_observation(
    recorded: ArticleV1JournalEntry,
    observed: ReplayObservation,
) -> None:
    for field in fields(ReplayObservation):
        expected_value = getattr(recorded, field.name)
        actual_value = getattr(observed, field.name)
        if actual_value != expected_value or type(actual_value) is not type(expected_value):
            raise CheckpointCompatibilityError(
                f"deterministic replay {field.name} mismatch at expansion "
                f"{recorded.expansion_index}"
            )


def replay_and_validate(
    journal: ArticleV1EventJournal,
    replay_step: Callable[[int], ReplayObservation],
) -> int:
    """Replay recorded IDs without policy ranking and validate every transition."""

    count = 0
    for entry in journal.entries:
        observation = replay_step(entry.selected_record_id)
        validate_replay_observation(entry, observation)
        count += 1
    return count


def validate_pending_resume_state(
    checkpoint: MidEpisodeCheckpoint,
    *,
    active_record_ids: Iterable[int],
    recomputed_pending_feature_row: Sequence[float],
    frontier_revision: int,
    frontier_active_ids_digest: str,
    archive_digest: str,
    generation_count_digest: str,
) -> None:
    """Validate the freshly replayed frontier before restoring RNG/policy state."""

    active = set(active_record_ids)
    if any(not _is_int(record_id) or record_id < 0 for record_id in active):
        raise CheckpointCompatibilityError("active record IDs are invalid")
    if checkpoint.pending_next_record_id not in active:
        raise CheckpointCompatibilityError("pending next record is not open after replay")
    if feature_row_digest(recomputed_pending_feature_row) != checkpoint.pending_feature_digest:
        raise CheckpointCompatibilityError(
            "pending next feature row differs after deterministic replay"
        )
    checks = {
        "frontier revision": (frontier_revision, checkpoint.frontier_revision),
        "frontier active-ID digest": (
            frontier_active_ids_digest,
            checkpoint.frontier_active_ids_digest,
        ),
        "archive digest": (archive_digest, checkpoint.archive_digest),
        "generation-count digest": (
            generation_count_digest,
            checkpoint.generation_count_digest,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise CheckpointCompatibilityError(f"replayed {label} mismatch")


@dataclass(frozen=True, slots=True)
class CheckpointCadence:
    every_expansions: int | None = 64
    every_seconds: float | None = 60.0

    def __post_init__(self) -> None:
        if self.every_expansions is not None:
            _integer("every_expansions", self.every_expansions, minimum=1)
        if self.every_seconds is not None:
            seconds = _number("every_seconds", self.every_seconds, minimum=0.0)
            if seconds <= 0.0:
                raise CheckpointFormatError("every_seconds must be > 0")
        if self.every_expansions is None and self.every_seconds is None:
            raise CheckpointFormatError("at least one checkpoint cadence is required")


class CheckpointCadenceGate:
    def __init__(
        self,
        cadence: CheckpointCadence | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        initial_expansion: int = 0,
    ) -> None:
        self.cadence = cadence or CheckpointCadence()
        self._clock = clock
        self._last_expansion = _integer("initial_expansion", initial_expansion)
        self._last_time = float(clock())

    def due(
        self,
        expansion: int,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> bool:
        expansion = _integer("expansion", expansion)
        current_time = float(self._clock() if now is None else now)
        if expansion < self._last_expansion or current_time < self._last_time:
            raise CheckpointFormatError("checkpoint cadence state regressed")
        if force:
            return True
        return bool(
            (
                self.cadence.every_expansions is not None
                and expansion - self._last_expansion
                >= self.cadence.every_expansions
            )
            or (
                self.cadence.every_seconds is not None
                and current_time - self._last_time >= self.cadence.every_seconds
            )
        )

    def mark_saved(self, expansion: int, *, now: float | None = None) -> None:
        expansion = _integer("expansion", expansion)
        current_time = float(self._clock() if now is None else now)
        if expansion < self._last_expansion or current_time < self._last_time:
            raise CheckpointFormatError("cannot mark a regressed checkpoint")
        self._last_expansion = expansion
        self._last_time = current_time

    def reset(self, *, expansion: int = 0, now: float | None = None) -> None:
        self._last_expansion = _integer("expansion", expansion)
        self._last_time = float(self._clock() if now is None else now)


@dataclass(frozen=True, slots=True)
class CheckpointWriteReceipt:
    slot: str
    path: Path
    digest: str
    byte_length: int
    elapsed_ns: int


def _reject_constant(value: str) -> None:
    raise CheckpointFormatError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointFormatError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, artifact: str) -> Mapping[str, object]:
    if not raw or not raw.endswith(b"\n"):
        raise CheckpointFormatError(f"{artifact} is empty or lacks a committed final LF")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointFormatError(f"{artifact} is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise CheckpointFormatError(f"{artifact} must contain one JSON object")
    return payload


def _require_exact_members(
    payload: Mapping[str, object], expected: set[str], label: str
) -> None:
    if not isinstance(payload, Mapping):
        raise CheckpointFormatError(f"{label} must be an object")
    actual = set(payload)
    if actual != expected:
        raise CheckpointFormatError(
            f"{label} members mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class ArticleV1TrainingCheckpointStore:
    """Atomic `latest`/`previous`/`episode-final` store with manifest-last commits."""

    _SLOTS = frozenset({"latest", "previous", "episode-final"})

    def __init__(
        self,
        directory: str | Path,
        *,
        timing_clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._timing_clock_ns = timing_clock_ns
        self.checkpoint_io_time_ns = 0

    def checkpoint_path(self, slot: str) -> Path:
        if slot not in self._SLOTS:
            raise ValueError(f"unsupported checkpoint slot {slot!r}")
        return self.directory / f"{slot}.json"

    def manifest_path(self, slot: str) -> Path:
        if slot not in self._SLOTS:
            raise ValueError(f"unsupported checkpoint slot {slot!r}")
        return self.directory / f"{slot}.manifest.json"

    @staticmethod
    def _encoded(checkpoint: TrainingRecoveryCheckpoint) -> bytes:
        if not isinstance(checkpoint, (MidEpisodeCheckpoint, TrainingProgressCheckpoint)):
            raise TypeError("store accepts only portable internal training checkpoints")
        return _canonical_json_bytes(checkpoint.to_payload())

    def _manifest(
        self, *, slot: str, encoded: bytes, checkpoint_schema: str
    ) -> dict[str, object]:
        return {
            "checkpoint_manifest_schema": ARTICLE_V1_CHECKPOINT_MANIFEST_SCHEMA,
            "slot": slot,
            "checkpoint_schema": checkpoint_schema,
            "checkpoint_sha256": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            "checkpoint_byte_length": len(encoded),
        }

    def _write_slot(
        self,
        slot: str,
        encoded: bytes,
        *,
        checkpoint_schema: str,
    ) -> CheckpointWriteReceipt:
        started = self._timing_clock_ns()
        checkpoint_path = self.checkpoint_path(slot)
        manifest_path = self.manifest_path(slot)
        manifest = self._manifest(
            slot=slot, encoded=encoded, checkpoint_schema=checkpoint_schema
        )
        # The manifest is the commit record and is deliberately replaced last.
        _atomic_write_bytes(checkpoint_path, encoded)
        _atomic_write_bytes(manifest_path, _canonical_json_bytes(manifest))
        elapsed = self._timing_clock_ns() - started
        self.checkpoint_io_time_ns += elapsed
        return CheckpointWriteReceipt(
            slot=slot,
            path=checkpoint_path,
            digest=str(manifest["checkpoint_sha256"]),
            byte_length=len(encoded),
            elapsed_ns=elapsed,
        )

    def _slot_exists(self, slot: str) -> bool:
        checkpoint_exists = self.checkpoint_path(slot).exists()
        manifest_exists = self.manifest_path(slot).exists()
        if checkpoint_exists != manifest_exists:
            raise CheckpointFormatError(
                f"checkpoint slot {slot!r} is incomplete (payload/manifest mismatch)"
            )
        return checkpoint_exists

    def _read_slot_bytes(self, slot: str) -> tuple[bytes, Mapping[str, object]]:
        if not self._slot_exists(slot):
            raise FileNotFoundError(self.checkpoint_path(slot))
        raw = self.checkpoint_path(slot).read_bytes()
        manifest = _strict_json_bytes(
            self.manifest_path(slot).read_bytes(), artifact=f"{slot} manifest"
        )
        _require_exact_members(
            manifest,
            {
                "checkpoint_manifest_schema",
                "slot",
                "checkpoint_schema",
                "checkpoint_sha256",
                "checkpoint_byte_length",
            },
            "checkpoint manifest",
        )
        if manifest["checkpoint_manifest_schema"] != ARTICLE_V1_CHECKPOINT_MANIFEST_SCHEMA:
            raise CheckpointFormatError("unsupported checkpoint manifest schema")
        if manifest["slot"] != slot:
            raise CheckpointFormatError("checkpoint manifest slot mismatch")
        expected_digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if manifest["checkpoint_sha256"] != expected_digest:
            raise CheckpointFormatError("checkpoint payload digest mismatch")
        if manifest["checkpoint_byte_length"] != len(raw) or not _is_int(
            manifest["checkpoint_byte_length"]
        ):
            raise CheckpointFormatError("checkpoint byte length mismatch")
        return raw, manifest

    def save_latest(
        self, checkpoint: TrainingRecoveryCheckpoint
    ) -> CheckpointWriteReceipt:
        encoded = self._encoded(checkpoint)
        if self._slot_exists("latest"):
            previous_bytes, previous_manifest = self._read_slot_bytes("latest")
            self._write_slot(
                "previous",
                previous_bytes,
                checkpoint_schema=str(previous_manifest["checkpoint_schema"]),
            )
        return self._write_slot(
            "latest", encoded, checkpoint_schema=checkpoint.schema_version
        )

    def save_episode_final(
        self, checkpoint: TrainingProgressCheckpoint
    ) -> CheckpointWriteReceipt:
        """Rotate latest, then publish the same episode boundary to its durable slot."""

        self.save_latest(checkpoint)
        encoded = self._encoded(checkpoint)
        return self._write_slot(
            "episode-final", encoded, checkpoint_schema=checkpoint.schema_version
        )

    def load(self, slot: str = "latest") -> TrainingRecoveryCheckpoint:
        raw, manifest = self._read_slot_bytes(slot)
        payload = _strict_json_bytes(raw, artifact=f"{slot} checkpoint")
        checkpoint = checkpoint_from_payload(payload)
        if manifest["checkpoint_schema"] != checkpoint.schema_version:
            raise CheckpointFormatError("checkpoint manifest schema binding mismatch")
        return checkpoint

    def load_latest_or_previous(self) -> TrainingRecoveryCheckpoint:
        """Recover from an interrupted latest write while still validating fail-closed."""

        try:
            return self.load("latest")
        except (CheckpointFormatError, FileNotFoundError) as latest_error:
            try:
                return self.load("previous")
            except (CheckpointFormatError, FileNotFoundError):
                raise latest_error

    def load_for_resume(
        self,
        expected: ResumeExpectation,
        *,
        slot: str = "latest",
    ) -> MidEpisodeCheckpoint:
        checkpoint = self.load(slot)
        if not isinstance(checkpoint, MidEpisodeCheckpoint):
            raise CheckpointCompatibilityError(
                "requested mid-episode resume but checkpoint is target/episode progress"
            )
        validate_resume_compatibility(checkpoint, expected)
        return checkpoint


__all__ = [
    "ARTICLE_V1_CHECKPOINT_MANIFEST_SCHEMA",
    "ARTICLE_V1_EVENT_JOURNAL_SCHEMA",
    "ARTICLE_V1_EVENT_SCHEMA",
    "ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA",
    "ARTICLE_V1_TRAINING_PROGRESS_SCHEMA",
    "ArticleV1CheckpointProvenance",
    "ArticleV1EventJournal",
    "ArticleV1JournalEntry",
    "ArticleV1TrainingCheckpointStore",
    "CheckpointCadence",
    "CheckpointCadenceGate",
    "CheckpointCompatibilityError",
    "CheckpointFormatError",
    "CheckpointWriteReceipt",
    "IncompleteTrainingCheckpointError",
    "MidEpisodeCheckpoint",
    "ReplayObservation",
    "ResumeExpectation",
    "TrainingProgressCheckpoint",
    "checkpoint_from_payload",
    "feature_row_digest",
    "policy_weight_digest",
    "portable_digest",
    "reject_internal_checkpoint_for_evaluation",
    "replay_and_validate",
    "validate_pending_resume_state",
    "validate_replay_observation",
    "validate_resume_compatibility",
]
