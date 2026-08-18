"""Validated trusted-local runtime snapshots for Article V1 recovery.

The portable JSON event journal remains the authoritative, cross-machine
fallback.  This module stores an optional Python-specific cache only after a
portable mid-episode checkpoint has committed.  The cache is hashed and bound
to the checkpoint prefix, source/config/target schemas, RNG states, and exact
frontier/archive/generation digests before unpickling is permitted.

Pickle is intentionally confined to this trusted local cache.  Callers must
never accept a snapshot supplied by an untrusted party.
"""

from __future__ import annotations

import copyreg
from dataclasses import dataclass
from hashlib import sha256
import io
import os
from pathlib import Path
import pickle
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

from experiments.article_v1_training_checkpoint import (
    ArticleV1EventJournal,
    CheckpointFormatError,
    MidEpisodeCheckpoint,
    _atomic_write_bytes,
    _canonical_json_bytes,
    _require_exact_members,
    _strict_json_bytes,
    portable_digest,
)


ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA = "article-v1-trusted-runtime-snapshot-v1"
ARTICLE_V1_RUNTIME_SNAPSHOT_MANIFEST_SCHEMA = (
    "article-v1-trusted-runtime-snapshot-manifest-v1"
)
DEFAULT_RUNTIME_SNAPSHOT_INTERVAL = 256
_TRUST_BOUNDARY = "trusted-local-cache-with-portable-json-replay-fallback"


def _restore_mapping_proxy(values: Mapping[object, object]) -> MappingProxyType:
    return MappingProxyType(dict(values))


def _reduce_mapping_proxy(
    value: MappingProxyType,
) -> tuple[Callable[..., MappingProxyType], tuple[dict[object, object]]]:
    return _restore_mapping_proxy, (dict(value),)


class _RuntimeSnapshotPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[MappingProxyType] = _reduce_mapping_proxy


def _pickle_bytes(value: object) -> bytes:
    stream = io.BytesIO()
    _RuntimeSnapshotPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return stream.getvalue()


def _visit_counts_digest(visit_counts: Mapping[object, int]) -> str:
    """Hash tuple-keyed visit counts through a deterministic JSON sequence."""

    entries = [
        (portable_digest(key, domain="article-v1-runtime-visit-key-v1"), key, count)
        for key, count in visit_counts.items()
    ]
    entries.sort(key=lambda item: item[0])
    return portable_digest(
        [[key, count] for _, key, count in entries],
        domain="article-v1-runtime-visit-counts-v1",
    )


@dataclass(slots=True)
class ArticleV1RuntimeState:
    """Python-specific state cached at one verified checkpoint boundary."""

    base_expansion: int
    environment: object
    visit_counts: dict[object, int]
    schema_version: str = ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA:
            raise CheckpointFormatError("unsupported runtime snapshot schema")
        if (
            isinstance(self.base_expansion, bool)
            or not isinstance(self.base_expansion, int)
            or self.base_expansion < 1
        ):
            raise CheckpointFormatError("runtime snapshot expansion must be positive")
        if not isinstance(self.visit_counts, dict) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.visit_counts.values()
        ):
            raise CheckpointFormatError(
                "runtime snapshot visit counts must be non-negative integers"
            )


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotReceipt:
    payload_path: Path
    manifest_path: Path
    payload_sha256: str
    payload_byte_length: int
    base_expansion: int
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class LoadedRuntimeSnapshot:
    state: ArticleV1RuntimeState
    manifest: Mapping[str, object]
    slot: str


_PROVENANCE_FIELDS = (
    "source_commit_sha",
    "source_worktree_digest",
    "config_digest",
    "corpus_digest",
    "profile_digest",
    "target_id",
    "target_fingerprint",
    "feature_schema_version",
    "feature_evaluator_schema_version",
    "reward_schema_version",
    "certifier_schema_version",
)

_MANIFEST_FIELDS = {
    "runtime_snapshot_manifest_schema",
    "runtime_snapshot_schema",
    "trust_boundary",
    "slot",
    "payload_sha256",
    "payload_byte_length",
    "base_expansion",
    "base_journal_digest",
    "checkpoint_schema_version",
    "frontier_active_ids_digest",
    "archive_digest",
    "generation_count_digest",
    "policy_weight_digest",
    "pending_record_id",
    "pending_feature_digest",
    "policy_rng_state_digest",
    "environment_rng_state_digest",
    "visit_counts_digest",
    *_PROVENANCE_FIELDS,
}


def _checkpoint_prefix_digest(
    checkpoint: MidEpisodeCheckpoint, expansion: int
) -> str:
    if expansion < 1 or expansion > len(checkpoint.journal.entries):
        raise CheckpointFormatError("runtime snapshot base is outside the journal")
    return ArticleV1EventJournal(
        checkpoint.journal.entries[:expansion],
        base_expansion=checkpoint.journal.base_expansion,
    ).digest


def _manifest_for(
    *,
    slot: str,
    payload: bytes,
    checkpoint: MidEpisodeCheckpoint,
    state: ArticleV1RuntimeState,
    pending_feature_digest: str,
    policy_rng_state_digest: str,
    environment_rng_state_digest: str,
) -> dict[str, object]:
    if state.base_expansion != checkpoint.expansion_count:
        raise CheckpointFormatError(
            "runtime snapshot must bind the checkpoint's current expansion"
        )
    final = checkpoint.journal.entries[-1]
    if not final.state_digest_verified:
        raise CheckpointFormatError(
            "runtime snapshot requires a full-state-verified final journal entry"
        )
    provenance = checkpoint.provenance
    result: dict[str, object] = {
        "runtime_snapshot_manifest_schema": (
            ARTICLE_V1_RUNTIME_SNAPSHOT_MANIFEST_SCHEMA
        ),
        "runtime_snapshot_schema": ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA,
        "trust_boundary": _TRUST_BOUNDARY,
        "slot": slot,
        "payload_sha256": f"sha256:{sha256(payload).hexdigest()}",
        "payload_byte_length": len(payload),
        "base_expansion": state.base_expansion,
        "base_journal_digest": checkpoint.journal_digest,
        "checkpoint_schema_version": checkpoint.schema_version,
        "frontier_active_ids_digest": checkpoint.frontier_active_ids_digest,
        "archive_digest": checkpoint.archive_digest,
        "generation_count_digest": checkpoint.generation_count_digest,
        "policy_weight_digest": checkpoint.weight_digest,
        "pending_record_id": checkpoint.pending_next_record_id,
        "pending_feature_digest": pending_feature_digest,
        "policy_rng_state_digest": policy_rng_state_digest,
        "environment_rng_state_digest": environment_rng_state_digest,
        "visit_counts_digest": _visit_counts_digest(state.visit_counts),
    }
    for name in _PROVENANCE_FIELDS:
        result[name] = getattr(provenance, name)
    return result


class ArticleV1RuntimeSnapshotStore:
    """Two-slot trusted cache with manifest-last atomic publication."""

    _SLOTS = ("latest", "previous")

    def __init__(
        self,
        directory: str | Path,
        *,
        timing_clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock_ns = timing_clock_ns
        self.snapshot_io_time_ns = 0
        self.snapshot_write_count = 0
        self.snapshot_bytes_written = 0

    def payload_path(self, slot: str) -> Path:
        if slot not in self._SLOTS:
            raise ValueError(f"unsupported runtime snapshot slot {slot!r}")
        return self.directory / f"runtime-snapshot-{slot}.pickle"

    def manifest_path(self, slot: str) -> Path:
        if slot not in self._SLOTS:
            raise ValueError(f"unsupported runtime snapshot slot {slot!r}")
        return self.directory / f"runtime-snapshot-{slot}.manifest.json"

    def _slot_presence(self, slot: str) -> bool:
        payload_exists = self.payload_path(slot).exists()
        manifest_exists = self.manifest_path(slot).exists()
        if payload_exists != manifest_exists:
            raise CheckpointFormatError(
                f"runtime snapshot slot {slot!r} is incomplete"
            )
        return payload_exists

    def _read_validated_bytes(
        self, slot: str
    ) -> tuple[bytes, Mapping[str, object]]:
        if not self._slot_presence(slot):
            raise FileNotFoundError(self.payload_path(slot))
        payload = self.payload_path(slot).read_bytes()
        manifest = _strict_json_bytes(
            self.manifest_path(slot).read_bytes(),
            artifact=f"runtime snapshot {slot} manifest",
        )
        _require_exact_members(manifest, _MANIFEST_FIELDS, "runtime snapshot manifest")
        if (
            manifest["runtime_snapshot_manifest_schema"]
            != ARTICLE_V1_RUNTIME_SNAPSHOT_MANIFEST_SCHEMA
            or manifest["runtime_snapshot_schema"]
            != ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA
            or manifest["trust_boundary"] != _TRUST_BOUNDARY
            or manifest["slot"] != slot
        ):
            raise CheckpointFormatError("runtime snapshot manifest identity mismatch")
        expected = f"sha256:{sha256(payload).hexdigest()}"
        if manifest["payload_sha256"] != expected:
            raise CheckpointFormatError("runtime snapshot payload digest mismatch")
        if (
            isinstance(manifest["payload_byte_length"], bool)
            or manifest["payload_byte_length"] != len(payload)
        ):
            raise CheckpointFormatError("runtime snapshot byte length mismatch")
        return payload, manifest

    def save_latest(
        self,
        checkpoint: MidEpisodeCheckpoint,
        state: ArticleV1RuntimeState,
        *,
        pending_feature_digest: str,
        policy_rng_state_digest: str,
        environment_rng_state_digest: str,
    ) -> RuntimeSnapshotReceipt:
        started = self._clock_ns()
        payload = _pickle_bytes(state)
        manifest = _manifest_for(
            slot="latest",
            payload=payload,
            checkpoint=checkpoint,
            state=state,
            pending_feature_digest=pending_feature_digest,
            policy_rng_state_digest=policy_rng_state_digest,
            environment_rng_state_digest=environment_rng_state_digest,
        )
        if self._slot_presence("latest"):
            previous_payload, previous_manifest = self._read_validated_bytes("latest")
            previous_manifest = dict(previous_manifest)
            previous_manifest["slot"] = "previous"
            _atomic_write_bytes(self.payload_path("previous"), previous_payload)
            _atomic_write_bytes(
                self.manifest_path("previous"),
                _canonical_json_bytes(previous_manifest),
            )
        _atomic_write_bytes(self.payload_path("latest"), payload)
        _atomic_write_bytes(
            self.manifest_path("latest"), _canonical_json_bytes(manifest)
        )
        elapsed = self._clock_ns() - started
        self.snapshot_io_time_ns += elapsed
        self.snapshot_write_count += 1
        self.snapshot_bytes_written += len(payload)
        return RuntimeSnapshotReceipt(
            payload_path=self.payload_path("latest"),
            manifest_path=self.manifest_path("latest"),
            payload_sha256=str(manifest["payload_sha256"]),
            payload_byte_length=len(payload),
            base_expansion=state.base_expansion,
            elapsed_ns=elapsed,
        )

    @staticmethod
    def _validate_binding(
        manifest: Mapping[str, object], checkpoint: MidEpisodeCheckpoint
    ) -> None:
        base = manifest["base_expansion"]
        if isinstance(base, bool) or not isinstance(base, int):
            raise CheckpointFormatError("runtime snapshot base expansion is invalid")
        if base < 1 or base > checkpoint.expansion_count:
            raise CheckpointFormatError(
                "runtime snapshot is newer than the portable checkpoint"
            )
        if (
            manifest["checkpoint_schema_version"] != checkpoint.schema_version
            or manifest["base_journal_digest"]
            != _checkpoint_prefix_digest(checkpoint, base)
        ):
            raise CheckpointFormatError("runtime snapshot journal binding mismatch")
        provenance = checkpoint.provenance
        for name in _PROVENANCE_FIELDS:
            if manifest[name] != getattr(provenance, name):
                raise CheckpointFormatError(
                    f"runtime snapshot provenance mismatch: {name}"
                )
        entry = checkpoint.journal.entries[base - 1]
        expected = {
            "frontier_active_ids_digest": entry.frontier_active_ids_digest,
            "archive_digest": entry.archive_digest,
            "generation_count_digest": entry.generation_count_digest,
            "policy_weight_digest": entry.policy_weight_digest_after_update,
            "pending_record_id": entry.pending_next_record_id,
        }
        if not entry.state_digest_verified or any(
            manifest[name] != value for name, value in expected.items()
        ):
            raise CheckpointFormatError("runtime snapshot state binding mismatch")

    def load_compatible(
        self, checkpoint: MidEpisodeCheckpoint
    ) -> LoadedRuntimeSnapshot | None:
        """Load the newest compatible trusted cache, else request JSON replay.

        Missing, torn, corrupt, or incompatible caches are never authoritative;
        callers fall back to the portable journal instead.
        """

        for slot in self._SLOTS:
            try:
                payload, manifest = self._read_validated_bytes(slot)
                self._validate_binding(manifest, checkpoint)
                value = pickle.loads(payload)
                if not isinstance(value, ArticleV1RuntimeState):
                    raise CheckpointFormatError(
                        "runtime snapshot payload has the wrong type"
                    )
                if value.base_expansion != manifest["base_expansion"]:
                    raise CheckpointFormatError(
                        "runtime snapshot payload expansion mismatch"
                    )
                if _visit_counts_digest(value.visit_counts) != manifest[
                    "visit_counts_digest"
                ]:
                    raise CheckpointFormatError(
                        "runtime snapshot visit-count digest mismatch"
                    )
                return LoadedRuntimeSnapshot(value, manifest, slot)
            except (
                FileNotFoundError,
                CheckpointFormatError,
                pickle.PickleError,
                EOFError,
                AttributeError,
                ImportError,
                TypeError,
                ValueError,
            ):
                continue
        return None


__all__ = [
    "ARTICLE_V1_RUNTIME_SNAPSHOT_MANIFEST_SCHEMA",
    "ARTICLE_V1_RUNTIME_SNAPSHOT_SCHEMA",
    "DEFAULT_RUNTIME_SNAPSHOT_INTERVAL",
    "ArticleV1RuntimeSnapshotStore",
    "ArticleV1RuntimeState",
    "LoadedRuntimeSnapshot",
    "RuntimeSnapshotReceipt",
]
