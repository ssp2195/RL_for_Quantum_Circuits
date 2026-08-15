from __future__ import annotations

import numpy as np
import pytest

from experiments.profiles import ARTICLE_V1_PROFILE, experiment_profile
from rl.policy import LinearQPolicy


class _State:
    def __init__(self, value):
        self.value = float(value)


class _Node:
    def __init__(self, record_id, value):
        self.record_id = record_id
        self.state = _State(value)


class _Provider:
    schema_version = "test-schema-v1"
    dimension = 2
    names = ("bias", "value")

    def extract(self, state, frontier=None):
        return np.asarray([1.0, state.value], dtype=np.float64)

    def metadata(self):
        return {"target_fingerprint": "target:test"}


def test_article_v1_profile_is_explicit_and_complete():
    assert experiment_profile("article_v1") is ARTICLE_V1_PROFILE
    assert ARTICLE_V1_PROFILE.metadata() == {
        "name": "article_v1",
        "feature_schema": "article-v1-31d",
        "reward_schema": "article-v1-expansion-potential-amended",
        "target_metric_schema": "process-infidelity-v1",
        "certification_schema": "phase-frobenius-v1",
        "gamma": 1.0,
        "reward_clip": None,
        "exploration_bonus": 0.0,
        "frozen_evaluation_fairness_interval": 0,
    }


def test_frozen_feature_update_does_not_reextract_current_state():
    provider = _Provider()
    policy = LinearQPolicy(feature_provider=provider, lr=0.1, gamma=1.0, seed=0)
    node = _Node(1, 2.0)
    batch = policy.build_feature_batch([node])
    frozen = np.array(batch.features_for(node), copy=True)
    node.state.value = 99.0
    error = policy.update_from_features(
        current_features=frozen,
        reward=1.0,
        next_features=None,
        done=True,
    )
    assert error == pytest.approx(1.0)
    assert policy.theta.tolist() == pytest.approx([0.1, 0.2])
    assert frozen.tolist() == [1.0, 2.0]


def test_checkpoint_rejects_feature_schema_or_dimension_reinterpretation(tmp_path):
    policy = LinearQPolicy(feature_provider=_Provider(), seed=7)
    policy.theta[:] = (0.25, -0.5)
    checkpoint = tmp_path / "policy.json"
    policy.save_checkpoint(
        checkpoint,
        reward_schema_version=ARTICLE_V1_PROFILE.reward_schema,
        target_metric_schema_version=ARTICLE_V1_PROFILE.target_metric_schema,
        certification_schema_version=ARTICLE_V1_PROFILE.certification_schema,
        epsilon_schedule={"start": 0.2, "minimum": 0.05, "decay": 0.995},
        code_commit_sha="deadbeef",
        dirty_worktree=True,
        profile_name=ARTICLE_V1_PROFILE.name,
    )
    restored, metadata = LinearQPolicy.load_checkpoint(
        checkpoint,
        feature_provider=_Provider(),
        expected_profile_name="article_v1",
        expected_reward_schema=ARTICLE_V1_PROFILE.reward_schema,
        expected_target_metric_schema=ARTICLE_V1_PROFILE.target_metric_schema,
        expected_certification_schema=ARTICLE_V1_PROFILE.certification_schema,
    )
    assert restored.theta.tolist() == pytest.approx(policy.theta.tolist())
    assert metadata["ordered_feature_names"] == ["bias", "value"]

    class _WrongProvider(_Provider):
        schema_version = "test-schema-v2"

    with pytest.raises(ValueError, match="feature_schema_version mismatch"):
        LinearQPolicy.load_checkpoint(checkpoint, feature_provider=_WrongProvider())
