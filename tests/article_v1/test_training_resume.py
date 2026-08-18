from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

import experiments.article_v1_runner as runner_module

from benchmarks.article_native_corpus import ArticleV1Budget, ArticleV1EvaluationTarget
from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.gate import Gate
from enums import GateType

from experiments.article_v1_training_checkpoint import (
    ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA,
    ArticleV1CheckpointProvenance,
    ArticleV1EventJournal,
    ArticleV1JournalEntry,
    ArticleV1TrainingCheckpointStore,
    CheckpointCadence,
    CheckpointCadenceGate,
    CheckpointCompatibilityError,
    CheckpointFormatError,
    IncompleteTrainingCheckpointError,
    MidEpisodeCheckpoint,
    ReplayObservation,
    ResumeExpectation,
    TrainingProgressCheckpoint,
    checkpoint_from_payload,
    feature_row_digest,
    policy_weight_digest,
    portable_digest,
    reject_internal_checkpoint_for_evaluation,
    replay_and_validate,
    validate_pending_resume_state,
    validate_replay_observation,
    validate_resume_compatibility,
)
from experiments.article_v1_progress import (
    ARTICLE_V1_PROGRESS_EVENT_SCHEMA,
    ArticleV1ProgressReporter,
    ProgressCadence,
    load_progress_events,
)
from rl.policy import LinearQPolicy
from train import Trainer, TrainerBoundaryEvent
from experiments.article_v1_runner import train_article_v1_checkpoint


def _d(label: object) -> str:
    return portable_digest(label, domain="article-v1-test-digest-v1")


def _provenance(**changes: object) -> ArticleV1CheckpointProvenance:
    values: dict[str, object] = {
        "source_commit_sha": "f653193ec1fd15b17b948a476a0e89a343cbf062",
        "source_worktree_digest": _d("source"),
        "config_digest": _d("config"),
        "corpus_digest": _d("corpus"),
        "profile_digest": _d("profile"),
        "target_id": "train-3q-hard-000",
        "target_fingerprint": _d("target"),
        "feature_schema_version": "article-v1-31d",
        "feature_evaluator_schema_version": "article-v1-exact-incremental-v2",
        "reward_schema_version": "article-v1-expansion-potential-v1",
        "certifier_schema_version": "article-v1-certifier-v1",
    }
    values.update(changes)
    return ArticleV1CheckpointProvenance(**values)  # type: ignore[arg-type]


def _journal(
    count: int, *, theta: tuple[float, ...] = (0.25, -0.5)
) -> ArticleV1EventJournal:
    journal = ArticleV1EventJournal()
    for expansion in range(1, count + 1):
        final = expansion == count
        journal.append(
            ArticleV1JournalEntry(
                expansion_index=expansion,
                selected_record_id=10 + expansion,
                selected_feature_digest=_d(["feature", expansion]),
                reward=-float(expansion),
                terminated=False,
                truncated=False,
                frontier_revision=100 + expansion,
                state_digest_verified=True,
                frontier_active_ids_digest=_d(["frontier", expansion]),
                archive_digest=_d(["archive", expansion]),
                generation_count_digest=_d(["generation", expansion]),
                policy_weight_digest_after_update=(
                    policy_weight_digest(theta)
                    if final
                    else _d(["weights", expansion])
                ),
                pending_next_record_id=20 + expansion,
            )
        )
    return journal


def _mid_checkpoint(
    *,
    count: int = 3,
    theta: tuple[float, ...] = (0.25, -0.5),
    provenance: ArticleV1CheckpointProvenance | None = None,
) -> MidEpisodeCheckpoint:
    journal = _journal(count, theta=theta)
    final = journal.entries[-1]
    return MidEpisodeCheckpoint(
        provenance=provenance or _provenance(),
        training_seed=17,
        episode_index=0,
        episode_count=2,
        expansion_count=count,
        expansion_cap=64,
        journal=journal,
        episode_initial_theta=(0.0,) * len(theta),
        theta=theta,
        epsilon=0.2,
        policy_rng_state={"bit_generator": "fixture", "state": {"x": 42}},
        environment_rng_state={"bit_generator": "fixture", "state": {"x": 8}},
        pending_next_record_id=final.pending_next_record_id,  # type: ignore[arg-type]
        pending_next_feature_row=(0.5, -0.25),
        total_reward=-6.0,
        training_aggregates={"td_error_count": count, "td_error_sum": -1.25},
        search_metrics={"expansions": count, "generated": count * 21},
        frontier_revision=final.frontier_revision,
        frontier_active_ids_digest=final.frontier_active_ids_digest,
        archive_digest=final.archive_digest,
        generation_count_digest=final.generation_count_digest,
    )


def _progress_checkpoint(
    *, theta: tuple[float, ...] = (0.5, -0.25)
) -> TrainingProgressCheckpoint:
    return TrainingProgressCheckpoint(
        provenance=_provenance(target_id="train-3q-next"),
        training_seed=17,
        target_cursor=1,
        target_count=2,
        episode_cursor=0,
        episodes_per_target=2,
        theta=theta,
        epsilon=0.1,
        policy_rng_state={"state": 123},
        training_history=({"target_id": "train-2q-first", "episodes": 2},),
        completed_target_ids=("train-2q-first",),
        effective_budgets={"train-2q-first": 32, "train-3q-next": 64},
    )


def _expectation(
    checkpoint: MidEpisodeCheckpoint, **changes: object
) -> ResumeExpectation:
    values: dict[str, object] = {
        "provenance": checkpoint.provenance,
        "training_seed": checkpoint.training_seed,
        "episode_index": checkpoint.episode_index,
        "episode_count": checkpoint.episode_count,
        "expansion_cap": checkpoint.expansion_cap,
        "feature_dimension": len(checkpoint.theta),
    }
    values.update(changes)
    return ResumeExpectation(**values)  # type: ignore[arg-type]


def test_mid_episode_checkpoint_roundtrip_is_portable_and_journal_is_sealed(
    tmp_path: Path,
) -> None:
    checkpoint = _mid_checkpoint()
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    receipt = store.save_latest(checkpoint)
    restored = store.load()

    assert restored == checkpoint
    assert receipt.digest.startswith("sha256:")
    assert receipt.byte_length == receipt.path.stat().st_size
    assert b"pickle" not in receipt.path.read_bytes().lower()
    assert json.loads(receipt.path.read_text(encoding="utf-8"))[
        "checkpoint_schema"
    ] == ARTICLE_V1_MID_EPISODE_CHECKPOINT_SCHEMA
    with pytest.raises(CheckpointFormatError, match="sealed"):
        restored.journal.append(restored.journal.entries[-1])


def test_interval_v2_checkpoint_and_journal_reject_v1_artifacts_fail_closed() -> None:
    checkpoint_payload = _mid_checkpoint().to_payload()
    checkpoint_payload["checkpoint_schema"] = (
        "article-v1-mid-episode-replay-checkpoint-v1"
    )
    with pytest.raises(CheckpointFormatError, match="unsupported.*checkpoint schema"):
        checkpoint_from_payload(checkpoint_payload)
    with pytest.raises(IncompleteTrainingCheckpointError, match="not transferable"):
        reject_internal_checkpoint_for_evaluation(checkpoint_payload)

    journal_payload = _journal(1).to_payload()
    journal_payload["event_journal_schema"] = "article-v1-training-event-journal-v1"
    with pytest.raises(CheckpointFormatError, match="unsupported.*journal schema"):
        ArticleV1EventJournal.from_payload(journal_payload)


def test_store_rotates_latest_previous_and_episode_final(tmp_path: Path) -> None:
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    first = _mid_checkpoint(theta=(0.25, -0.5))
    second = _mid_checkpoint(theta=(0.5, -0.75))
    store.save_latest(first)
    store.save_latest(second)
    assert store.load("latest") == second
    assert store.load("previous") == first

    progress = _progress_checkpoint()
    receipt = store.save_episode_final(progress)
    assert receipt.slot == "episode-final"
    assert store.load("episode-final") == progress
    assert store.load("latest") == progress
    assert store.load("previous") == second
    for slot in ("latest", "previous", "episode-final"):
        assert store.checkpoint_path(slot).read_bytes().endswith(b"\n")
        assert store.manifest_path(slot).read_bytes().endswith(b"\n")
    assert store.checkpoint_io_time_ns >= receipt.elapsed_ns >= 0
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_payload_tampering_fails_closed_and_previous_remains_recoverable(
    tmp_path: Path,
) -> None:
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    first = _mid_checkpoint(theta=(0.25, -0.5))
    second = _mid_checkpoint(theta=(0.5, -0.75))
    store.save_latest(first)
    store.save_latest(second)
    latest = store.checkpoint_path("latest")
    latest.write_bytes(latest.read_bytes().replace(b'"epsilon":0.2', b'"epsilon":0.3'))
    with pytest.raises(CheckpointFormatError, match="digest mismatch"):
        store.load("latest")
    assert store.load_latest_or_previous() == first


def test_payload_without_manifest_is_rejected_as_incomplete(tmp_path: Path) -> None:
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    store.checkpoint_path("latest").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="incomplete"):
        store.load()


def test_duplicate_json_members_are_rejected_even_with_matching_manifest(
    tmp_path: Path,
) -> None:
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    store.save_latest(_mid_checkpoint())
    checkpoint_path = store.checkpoint_path("latest")
    raw = checkpoint_path.read_bytes()
    forged = (
        b'{"checkpoint_schema":"article-v1-mid-episode-replay-checkpoint-v2",'
        + raw[1:]
    )
    checkpoint_path.write_bytes(forged)
    manifest_path = store.manifest_path("latest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = "sha256:" + hashlib.sha256(forged).hexdigest()
    manifest["checkpoint_byte_length"] = len(forged)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CheckpointFormatError, match="duplicate"):
        store.load()


def test_failed_atomic_replace_preserves_committed_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    first = _mid_checkpoint(theta=(0.25, -0.5))
    second = _mid_checkpoint(theta=(0.5, -0.75))
    store.save_latest(first)

    import experiments.article_v1_training_checkpoint as checkpoint_module

    real_replace = checkpoint_module.os.replace

    def fail_latest(source: str, destination: str | Path) -> None:
        if Path(destination) == store.checkpoint_path("latest"):
            raise OSError("simulated latest replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_latest)
    with pytest.raises(OSError, match="simulated"):
        store.save_latest(second)
    assert store.load("latest") == first
    assert store.load("previous") == first
    assert not tuple(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_resume_rejects_every_provenance_or_schema_mismatch(field: str) -> None:
    checkpoint = _mid_checkpoint()
    old = getattr(checkpoint.provenance, field)
    foreign = (
        _d(["foreign", field])
        if isinstance(old, str) and old.startswith("sha256:")
        else f"foreign-{old}"
    )
    expected_provenance = replace(checkpoint.provenance, **{field: foreign})
    with pytest.raises(CheckpointCompatibilityError, match=field):
        validate_resume_compatibility(
            checkpoint, _expectation(checkpoint, provenance=expected_provenance)
        )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"training_seed": 18}, "training seed"),
        ({"episode_index": 1}, "episode index"),
        ({"episode_count": 3}, "episode count"),
        ({"expansion_cap": 65}, "expansion cap"),
        ({"feature_dimension": 3}, "feature dimension"),
    ],
)
def test_resume_rejects_seed_episode_budget_or_dimension_mismatch(
    change: dict[str, object], message: str
) -> None:
    checkpoint = _mid_checkpoint()
    with pytest.raises(CheckpointCompatibilityError, match=message):
        validate_resume_compatibility(checkpoint, _expectation(checkpoint, **change))


def test_journal_digest_and_safe_boundary_coherence_are_validated() -> None:
    checkpoint = _mid_checkpoint()
    payload = checkpoint.to_payload()
    payload["journal_digest"] = _d("forged journal")
    with pytest.raises(CheckpointFormatError, match="journal digest"):
        checkpoint_from_payload(payload)

    final = checkpoint.journal.entries[-1]
    with pytest.raises(CheckpointFormatError, match="pending next record"):
        replace(checkpoint, pending_next_record_id=final.pending_next_record_id + 1)  # type: ignore[operator]


def test_pending_state_validation_requires_open_id_exact_feature_and_digests() -> None:
    checkpoint = _mid_checkpoint()
    kwargs = {
        "active_record_ids": {checkpoint.pending_next_record_id, 999},
        "recomputed_pending_feature_row": checkpoint.pending_next_feature_row,
        "frontier_revision": checkpoint.frontier_revision,
        "frontier_active_ids_digest": checkpoint.frontier_active_ids_digest,
        "archive_digest": checkpoint.archive_digest,
        "generation_count_digest": checkpoint.generation_count_digest,
    }
    validate_pending_resume_state(checkpoint, **kwargs)
    with pytest.raises(CheckpointCompatibilityError, match="not open"):
        validate_pending_resume_state(
            checkpoint, **{**kwargs, "active_record_ids": {999}}
        )
    with pytest.raises(CheckpointCompatibilityError, match="feature row"):
        validate_pending_resume_state(
            checkpoint,
            **{**kwargs, "recomputed_pending_feature_row": (0.5, -0.2500000001)},
        )
    with pytest.raises(CheckpointCompatibilityError, match="archive"):
        validate_pending_resume_state(
            checkpoint, **{**kwargs, "archive_digest": _d("foreign archive")}
        )


def test_replay_observation_validation_is_type_and_value_strict() -> None:
    entry = _journal(1).entries[0]
    observation = ReplayObservation(
        **{
            field: getattr(entry, field)
            for field in ReplayObservation.__dataclass_fields__
        }
    )
    validate_replay_observation(entry, observation)
    with pytest.raises(CheckpointCompatibilityError, match="reward"):
        validate_replay_observation(entry, replace(observation, reward=entry.reward + 1.0))


def test_interval_state_digests_are_all_or_none_and_can_bind_latest() -> None:
    complete = _journal(1).entries[0]
    interval = replace(
        complete,
        state_digest_verified=False,
        frontier_active_ids_digest=None,
        archive_digest=None,
        generation_count_digest=None,
    )
    journal = ArticleV1EventJournal((interval,))
    payload = journal.to_payload()
    entry_payload = payload["entries"][0]
    assert entry_payload["state_digest_verified"] is False
    assert entry_payload["frontier_active_ids_digest"] is None
    assert ArticleV1EventJournal.from_payload(payload) == journal
    interval_observation = ReplayObservation(
        **{
            field: getattr(interval, field)
            for field in ReplayObservation.__dataclass_fields__
        }
    )
    validate_replay_observation(interval, interval_observation)

    rebound = journal.bind_latest_state_digests(
        frontier_active_ids_digest=complete.frontier_active_ids_digest,
        archive_digest=complete.archive_digest,
        generation_count_digest=complete.generation_count_digest,
    )
    assert rebound.state_digest_verified is True
    assert journal.entries[-1] == rebound

    with pytest.raises(CheckpointFormatError, match="omit all"):
        replace(interval, archive_digest=_d("partial"))
    with pytest.raises(CheckpointFormatError, match="nonempty string"):
        replace(interval, state_digest_verified=True)

    unbound = ArticleV1EventJournal((interval,))
    checkpoint = _mid_checkpoint(count=1)
    with pytest.raises(CheckpointFormatError, match="full-state digest boundary"):
        replace(checkpoint, journal=unbound)


def test_internal_training_checkpoints_can_never_be_evaluated() -> None:
    for checkpoint in (_mid_checkpoint(), _progress_checkpoint()):
        with pytest.raises(IncompleteTrainingCheckpointError, match="not transferable"):
            checkpoint.require_evaluation_eligible()
        with pytest.raises(IncompleteTrainingCheckpointError, match="not transferable"):
            reject_internal_checkpoint_for_evaluation(checkpoint.to_payload())


def test_training_progress_remains_nontransferable_after_all_targets_complete() -> None:
    completed = TrainingProgressCheckpoint(
        provenance=_provenance(),
        training_seed=17,
        target_cursor=2,
        target_count=2,
        episode_cursor=0,
        episodes_per_target=2,
        theta=(0.5, -0.25),
        epsilon=0.1,
        policy_rng_state={"state": 123},
        training_history=({"target_id": "a"}, {"target_id": "b"}),
        completed_target_ids=("a", "b"),
        effective_budgets={"a": 32, "b": 64},
    )
    assert completed.all_targets_completed
    assert completed.to_payload()["training_complete"] is True
    assert completed.to_payload()["transferable_for_evaluation"] is False
    with pytest.raises(IncompleteTrainingCheckpointError):
        completed.require_evaluation_eligible()


def test_checkpoint_cadence_uses_expansion_or_time_and_supports_forced_boundaries() -> None:
    gate = CheckpointCadenceGate(
        CheckpointCadence(every_expansions=64, every_seconds=60.0),
        clock=lambda: 100.0,
    )
    assert not gate.due(63, now=159.9)
    assert gate.due(64, now=101.0)
    gate.mark_saved(64, now=101.0)
    assert gate.due(65, now=161.0)
    gate.mark_saved(65, now=161.0)
    assert gate.due(65, force=True, now=161.0)
    gate.reset(expansion=0, now=200.0)
    assert not gate.due(1, now=201.0)

    resumed = CheckpointCadenceGate(
        CheckpointCadence(every_expansions=64, every_seconds=None),
        clock=lambda: 300.0,
        initial_expansion=64,
    )
    assert not resumed.due(127, now=301.0)
    assert resumed.due(128, now=301.0)


# A bounded deterministic fixture proves the recovery API without depending on
# the much slower full Article V1 environment integration.
@dataclass
class _FixtureEnv:
    state: int = 3
    expansions: int = 0

    @property
    def active_ids(self) -> set[int]:
        return {self.state % 11, (self.state + 3) % 11, (self.state + 7) % 11}

    def feature(self, record_id: int) -> tuple[float, float]:
        return (record_id / 10.0, self.state / 97.0)

    def step(self, selected: int) -> float:
        assert selected in self.active_ids
        self.state = (self.state * 13 + selected * 7 + 5) % 97
        self.expansions += 1
        return float((self.state % 9) - 4) / 4.0

    def digests(self) -> tuple[str, str, str]:
        return (
            _d(["frontier", sorted(self.active_ids)]),
            _d(["archive", self.state, self.expansions]),
            _d(["generation", self.expansions, self.state % 5]),
        )


@dataclass
class _FixturePolicy:
    theta: list[float]
    rng: int

    def select(self, active: set[int]) -> int:
        self.rng = (1103515245 * self.rng + 12345) % (2**31)
        return sorted(active)[self.rng % len(active)]

    def update(self, feature: tuple[float, float], reward: float) -> float:
        td_error = reward - sum(w * x for w, x in zip(self.theta, feature))
        for index, value in enumerate(feature):
            self.theta[index] += 0.05 * td_error * value
        return td_error


def _fixture_entry(
    env: _FixtureEnv,
    policy: _FixturePolicy,
    selected: int,
    *,
    forced_pending: int | None = None,
) -> tuple[ArticleV1JournalEntry, float, int | None]:
    selected_feature = env.feature(selected)
    reward = env.step(selected)
    td_error = policy.update(selected_feature, reward)
    terminal = env.expansions == 8
    pending = None if terminal else (
        forced_pending if forced_pending is not None else policy.select(env.active_ids)
    )
    frontier_digest, archive_digest, generation_digest = env.digests()
    entry = ArticleV1JournalEntry(
        expansion_index=env.expansions,
        selected_record_id=selected,
        selected_feature_digest=feature_row_digest(selected_feature),
        reward=reward,
        terminated=terminal,
        truncated=False,
        frontier_revision=env.expansions,
        state_digest_verified=True,
        frontier_active_ids_digest=frontier_digest,
        archive_digest=archive_digest,
        generation_count_digest=generation_digest,
        policy_weight_digest_after_update=policy_weight_digest(policy.theta),
        pending_next_record_id=pending,
    )
    return entry, td_error, pending


def _execute_fixture(stop_after: int | None = None):
    env = _FixtureEnv()
    policy = _FixturePolicy([0.1, -0.2], rng=19)
    journal = ArticleV1EventJournal()
    td_errors: list[float] = []
    pending = policy.select(env.active_ids)
    while env.expansions < 8 and (stop_after is None or env.expansions < stop_after):
        entry, td_error, pending = _fixture_entry(env, policy, pending)
        td_errors.append(td_error)
        journal.append(entry)
    return env, policy, journal, td_errors, pending


def test_interrupted_replay_resume_equals_uninterrupted_bounded_fixture(
    tmp_path: Path,
) -> None:
    full_env, full_policy, full_journal, full_td, _ = _execute_fixture()
    env, policy, journal, td_errors, pending = _execute_fixture(stop_after=4)
    assert pending is not None
    frontier_digest, archive_digest, generation_digest = env.digests()
    checkpoint = MidEpisodeCheckpoint(
        provenance=_provenance(),
        training_seed=17,
        episode_index=0,
        episode_count=1,
        expansion_count=env.expansions,
        expansion_cap=8,
        journal=journal,
        episode_initial_theta=(0.1, -0.2),
        theta=tuple(policy.theta),
        epsilon=0.2,
        policy_rng_state={"lcg_state": policy.rng},
        environment_rng_state={"deterministic": True},
        pending_next_record_id=pending,
        pending_next_feature_row=env.feature(pending),
        total_reward=sum(entry.reward for entry in journal.entries),
        training_aggregates={"td_errors": td_errors},
        search_metrics={"expansions": env.expansions},
        frontier_revision=env.expansions,
        frontier_active_ids_digest=frontier_digest,
        archive_digest=archive_digest,
        generation_count_digest=generation_digest,
    )
    store = ArticleV1TrainingCheckpointStore(tmp_path)
    store.save_latest(checkpoint)
    restored = store.load_for_resume(_expectation(checkpoint))

    replay_env = _FixtureEnv()
    replay_policy = _FixturePolicy([0.1, -0.2], rng=19)
    replay_index = 0

    def replay_step(selected_record_id: int) -> ReplayObservation:
        nonlocal replay_index
        recorded = restored.journal.entries[replay_index]
        # The recorded next action is reused; policy.select is deliberately absent.
        entry, _td_error, _pending = _fixture_entry(
            replay_env,
            replay_policy,
            selected_record_id,
            forced_pending=recorded.pending_next_record_id,
        )
        replay_index += 1
        return ReplayObservation(
            **{
                field: getattr(entry, field)
                for field in ReplayObservation.__dataclass_fields__
            }
        )

    assert replay_and_validate(restored.journal, replay_step) == 4
    validate_pending_resume_state(
        restored,
        active_record_ids=replay_env.active_ids,
        recomputed_pending_feature_row=replay_env.feature(restored.pending_next_record_id),
        frontier_revision=replay_env.expansions,
        frontier_active_ids_digest=replay_env.digests()[0],
        archive_digest=replay_env.digests()[1],
        generation_count_digest=replay_env.digests()[2],
    )

    # Restore policy/RNG only after replay and continue with the stored pending
    # action, so resume consumes no extra exploratory random draw.
    replay_policy.theta[:] = restored.theta
    replay_policy.rng = int(restored.policy_rng_state["lcg_state"])
    resumed_journal = ArticleV1EventJournal(restored.journal.entries)
    resumed_td = list(restored.training_aggregates["td_errors"])
    pending = restored.pending_next_record_id
    while replay_env.expansions < 8:
        entry, td_error, pending = _fixture_entry(replay_env, replay_policy, pending)
        resumed_journal.append(entry)
        resumed_td.append(td_error)

    assert [entry.selected_record_id for entry in resumed_journal.entries] == [
        entry.selected_record_id for entry in full_journal.entries
    ]
    assert [entry.reward for entry in resumed_journal.entries] == [
        entry.reward for entry in full_journal.entries
    ]
    assert resumed_td == full_td
    assert replay_policy.theta == full_policy.theta
    assert policy_weight_digest(replay_policy.theta) == policy_weight_digest(
        full_policy.theta
    )
    assert replay_env.state == full_env.state  # certification result/witness surrogate
    assert replay_env.expansions == full_env.expansions
    assert replay_env.digests() == full_env.digests()


class _BoundaryState:
    def __init__(self, value: float) -> None:
        self.value = float(value)


class _BoundaryNode:
    def __init__(self, record_id: int, value: float) -> None:
        self.record_id = record_id
        self.state = _BoundaryState(value)


class _BoundaryProvider:
    schema_version = "bounded-callback-feature-v1"
    dimension = 2
    names = ("bias", "value")

    def extract(self, state: _BoundaryState, frontier=None):
        return np.asarray((1.0, state.value), dtype=np.float64)

    def metadata(self):
        return {"target_fingerprint": "bounded-callback-target"}


class _BoundaryCanonicalizer:
    @staticmethod
    def semantic_key(state: _BoundaryState):
        return (state.value,)


class _BoundaryFrontier:
    def __init__(self, env: "_BoundaryEnv") -> None:
        self.env = env
        self.revision = 0

    def active_record_ids(self):
        return tuple(node.record_id for node in self.env.nodes)


class _BoundaryEnv:
    """Two-transition Article-mode fixture for Trainer callback integration."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            discount=1.0,
            seed=11,
            reward_mode="article_v1_expansion_potential",
            fairness_interval=0,
            max_steps=2,
        )
        self.feature_provider = _BoundaryProvider()
        self.canonicalizer = _BoundaryCanonicalizer()
        self.frontier = _BoundaryFrontier(self)
        self.nodes = [_BoundaryNode(0, 1.0)]
        self.steps = 0
        self.solution_node = None
        self.search_metrics = {"frontier_peak": 1, "feature_time_ns": 0}

    def current_nodes(self):
        return list(self.nodes)

    def reset(self, seed=None):
        self.nodes = [_BoundaryNode(0, 1.0)]
        self.steps = 0
        self.frontier.revision = 1
        self.search_metrics = {"frontier_peak": 1, "feature_time_ns": 0}
        return None, {"initial_certified": False}

    def select_record(self, record_id: int):
        assert record_id == self.nodes[0].record_id
        self.steps += 1
        selected = self.nodes[0]
        if self.steps == 1:
            self.nodes = [_BoundaryNode(4, 2.0)]
            terminated = False
            truncated = False
        else:
            self.nodes = []
            terminated = False
            truncated = True
        self.frontier.revision += 1
        self.search_metrics = {
            "frontier_peak": 1,
            "feature_time_ns": self.steps * 10,
            "expansions": self.steps,
        }
        return None, -1.0, terminated, truncated, {
            "selected_record_id": selected.record_id,
            "selected_by_fairness": False,
            "frontier_size": len(self.nodes),
            "num_children": 1,
            "search_metrics": dict(self.search_metrics),
        }

    @staticmethod
    def reward_spec():
        return {"mode": "article_v1_expansion_potential"}


def test_trainer_emits_safe_expansion_and_forced_episode_boundary_callbacks() -> None:
    env = _BoundaryEnv()
    policy = LinearQPolicy(feature_provider=env.feature_provider, lr=0.1, seed=11)
    order: list[tuple[str, str, int]] = []
    checkpoint_events: list[TrainerBoundaryEvent] = []
    progress_events: list[TrainerBoundaryEvent] = []

    def checkpoint_callback(event: TrainerBoundaryEvent) -> None:
        order.append(("checkpoint", event.boundary, event.expansion))
        checkpoint_events.append(event)
        assert event.safe_sarsa_boundary
        assert event.policy_weight_digest_after_update == policy.weight_digest()
        if event.boundary == "expansion" and not (event.terminated or event.truncated):
            assert event.next_record_id in event.frontier_active_record_ids

    def progress_callback(event: TrainerBoundaryEvent) -> None:
        order.append(("progress", event.boundary, event.expansion))
        progress_events.append(event)

    trainer = Trainer(
        env,
        policy=policy,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.epsilon = 0.0
    trainer.min_epsilon = 0.0
    history = trainer.train(1)

    assert history[0]["steps"] == 2
    assert [event.boundary for event in checkpoint_events] == [
        "expansion",
        "expansion",
        "episode_end",
    ]
    assert progress_events == checkpoint_events
    assert order == [
        ("checkpoint", "expansion", 1),
        ("progress", "expansion", 1),
        ("checkpoint", "expansion", 2),
        ("progress", "expansion", 2),
        ("checkpoint", "episode_end", 2),
        ("progress", "episode_end", 2),
    ]
    first, second, episode_end = checkpoint_events
    assert first.selected_record_id == 0
    assert first.selected_features == (1.0, 1.0)
    assert first.next_record_id == 4
    assert first.next_features == (1.0, 2.0)
    # The next SARSA action consumes exactly the feature row frozen previously.
    assert second.selected_features == first.next_features
    assert second.next_record_id is None
    assert second.next_features is None
    assert second.truncated
    assert episode_end.epsilon == trainer.epsilon
    assert dict(first.search_metrics)["expansions"] == 1
    assert dict(second.search_metrics)["expansions"] == 2
    with pytest.raises(TypeError):
        first.search_metrics["expansions"] = 99  # type: ignore[index]
    assert trainer.checkpoint_callback_time_ns >= 0
    assert trainer.progress_callback_time_ns >= 0


def test_trainer_callback_failure_aborts_instead_of_claiming_success() -> None:
    env = _BoundaryEnv()
    policy = LinearQPolicy(feature_provider=env.feature_provider, lr=0.1, seed=11)

    def fail(_event: TrainerBoundaryEvent) -> None:
        raise RuntimeError("checkpoint publication failed")

    trainer = Trainer(env, policy=policy, checkpoint_callback=fail)
    trainer.epsilon = 0.0
    with pytest.raises(RuntimeError, match="publication failed"):
        trainer.train(1)


def test_training_without_recovery_does_not_build_journal_state_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = ArticleV1EvaluationTarget(
        target_id="train-no-recovery-digest",
        split="train",
        difficulty="easy",
        num_qubits=2,
        generator_length=2,
        budget=ArticleV1Budget(
            max_t_count=0,
            max_two_qubit_count=1,
            max_gates=3,
            max_depth=3,
            expansion_budget=1,
        ),
        target=SynthesisTarget(
            unitary_from_gates(2, (Gate(GateType.X, (0,)),))
        ),
    )

    def forbidden_digest(_environment: object) -> tuple[str, str, str]:
        raise AssertionError("no-recovery training must not serialize full state")

    monkeypatch.setattr(runner_module, "_training_state_digests", forbidden_digest)
    checkpoint = train_article_v1_checkpoint(
        (case,),
        corpus_config_digest=_d("no-recovery-corpus"),
        training_seed=7,
        episodes_per_target=1,
        learning_rate=1e-3,
        epsilon_start=0.2,
        epsilon_minimum=0.05,
        epsilon_decay=0.995,
        beta=1.0,
        expansion_cap=1,
        certification_tolerance=1e-9,
    )
    history = checkpoint.training_histories[0]
    assert history["checkpoint_callback_time_ns"] == 0
    assert history["checkpoint_state_digest_time_ns"] == 0
    assert history["checkpoint_io_time_ns"] == 0


def test_replay_capture_never_accepts_interrupt_without_valid_mid_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidStore:
        def __init__(self, _directory: object) -> None:
            self.loads = 0

        def load(self, _slot: str = "latest") -> object:
            self.loads += 1
            if self.loads == 1:
                raise FileNotFoundError
            return object()

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        runner_module, "ArticleV1TrainingCheckpointStore", InvalidStore
    )
    monkeypatch.setattr(runner_module, "train_article_v1_checkpoint", interrupt)
    with pytest.raises(
        CheckpointCompatibilityError,
        match="internal mid-episode checkpoint",
    ):
        runner_module.capture_article_v1_replay_checkpoint(
            tmp_path / "invalid-interrupt",
            quiet=True,
        )
    assert not (
        tmp_path / "invalid-interrupt" / "replay_checkpoint_capture.json"
    ).exists()


def test_real_article_training_interrupt_replays_and_resumes_identically(
    tmp_path: Path,
) -> None:
    case = ArticleV1EvaluationTarget(
        target_id="train-bounded-x-resume",
        split="train",
        difficulty="hard",
        num_qubits=2,
        generator_length=2,
        budget=ArticleV1Budget(
            max_t_count=0,
            max_two_qubit_count=1,
            max_gates=3,
            max_depth=3,
            expansion_budget=4,
        ),
        target=SynthesisTarget(
            unitary_from_gates(2, (Gate(GateType.X, (0,)),))
        ),
    )
    common = {
        "corpus_config_digest": _d("real-resume-corpus"),
        "training_seed": 31,
        "episodes_per_target": 1,
        "learning_rate": 1e-3,
        "epsilon_start": 0.2,
        "epsilon_minimum": 0.05,
        "epsilon_decay": 0.995,
        "beta": 1.0,
        "expansion_cap": 4,
        "certification_tolerance": 1e-9,
        "runtime_snapshot_every_expansions": 1,
    }
    uninterrupted = train_article_v1_checkpoint(
        (case,),
        training_checkpoint_dir=tmp_path / "uninterrupted-state",
        **common,
    )
    interrupted_state = tmp_path / "interrupted-state"
    progress_directory = tmp_path / "interrupted-progress"
    progress_reporter = ArticleV1ProgressReporter(
        progress_directory,
        cadence=ProgressCadence(every_expansions=25, every_seconds=None),
        quiet=True,
    )
    with pytest.raises(KeyboardInterrupt):
        train_article_v1_checkpoint(
            (case,),
            training_checkpoint_dir=interrupted_state,
            progress_reporter=progress_reporter,
            run_id="bounded-real-resume",
            interrupt_after_expansions=1,
            **common,
        )
    stored = ArticleV1TrainingCheckpointStore(interrupted_state).load("latest")
    assert isinstance(stored, MidEpisodeCheckpoint)
    assert stored.expansion_count == 1
    assert stored.journal.entries[-1].state_digest_verified is True
    assert stored.journal.entries[-1].frontier_active_ids_digest is not None
    assert (interrupted_state / "runtime-snapshot-latest.pickle").is_file()
    assert (
        interrupted_state / "runtime-snapshot-latest.manifest.json"
    ).is_file()
    events = load_progress_events(progress_directory / "progress.jsonl")
    assert len(events) == 1
    interrupt_event = events[0]
    assert interrupt_event.schema_version == ARTICLE_V1_PROGRESS_EVENT_SCHEMA
    assert interrupt_event.feature_evaluator_schema_version == (
        "article-v1-exact-incremental-v2"
    )
    assert interrupt_event.expansion == 1
    assert interrupt_event.checkpoint_path is not None
    assert Path(interrupt_event.checkpoint_path).name == "latest.json"
    assert interrupt_event.last_feature_batch_seconds > 0.0
    assert interrupt_event.rolling_feature_batch_seconds > 0.0
    assert interrupt_event.elapsed_seconds > 0.0
    assert interrupt_event.expansions_per_second == pytest.approx(
        interrupt_event.expansion / interrupt_event.elapsed_seconds
    )

    resumed = train_article_v1_checkpoint(
        (case,),
        training_checkpoint_dir=interrupted_state,
        **common,
    )
    assert resumed.weights == uninterrupted.weights
    assert resumed.weight_digest == uninterrupted.weight_digest
    assert resumed.training_histories[0]["episodes"] == (
        uninterrupted.training_histories[0]["episodes"]
    )
    assert resumed.training_histories[0]["episodes"][0]["search_metrics"] == (
        uninterrupted.training_histories[0]["episodes"][0]["search_metrics"]
    )
    assert resumed.training_histories[0]["runtime_snapshot_base_expansion"] == 1
    assert (
        resumed.training_histories[0]["runtime_snapshot_schema_version"]
        == "article-v1-trusted-runtime-snapshot-v1"
    )
    for timing_name in (
        "checkpoint_callback_time_ns",
        "checkpoint_state_digest_time_ns",
        "checkpoint_io_time_ns",
        "runtime_snapshot_restore_time_ns",
        "recovery_replay_time_ns",
        "runtime_snapshot_io_time_ns",
    ):
        assert int(resumed.training_histories[0][timing_name]) >= 0
    final_store = ArticleV1TrainingCheckpointStore(interrupted_state)
    episode_final = final_store.load("episode-final")
    clean_exit = final_store.load("latest")
    assert isinstance(episode_final, TrainingProgressCheckpoint)
    assert isinstance(clean_exit, TrainingProgressCheckpoint)
    assert episode_final.target_cursor == episode_final.target_count == 1
    assert clean_exit.target_cursor == clean_exit.target_count == 1
    assert episode_final.all_targets_completed
    assert clean_exit.all_targets_completed
    assert episode_final.completed_target_ids == (case.target_id,)
    assert clean_exit.completed_target_ids == (case.target_id,)
