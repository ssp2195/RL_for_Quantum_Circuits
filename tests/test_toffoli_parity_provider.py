"""Focused opt-in RL coverage for the Toffoli parity provider layer.

The fixture injects a tiny immutable progress analyzer.  This verifies the
provider adapter contract without duplicating or depending on the search
problem's own parity-analysis implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pytest

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.features import (
    LEGACY_FEATURE_DIMENSION,
    LegacyFeatureProvider,
    extract_features,
)
from rl.policy import LinearQPolicy
from rl.toffoli_parity import (
    TOFFOLI_PARITY_FEATURE_DIMENSION,
    TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION,
    ToffoliParityFeatureProvider,
    ToffoliParityRewardConfig,
    ToffoliParityRewardModel,
)
from search.node import SearchNode
from search.problems.toffoli_parity import ToffoliParityNetworkProblem


class _Stage(Enum):
    PRE_H = auto()
    CORE = auto()
    POST_H = auto()
    DONE = auto()


@dataclass(frozen=True)
class _Progress:
    stage: _Stage
    basis_rows: tuple[int, int, int]
    emitted_terms: int


# The test only needs stable labels; it does not implement any target replay.
_REQUIRED_PHASE_TERMS = {
    1: 1,
    2: 1,
    3: -1,
    4: 1,
    5: -1,
    6: -1,
    7: 1,
}
_BASIS_DISTANCE = {
    (1, 2, 4): 0,
    (3, 5, 4): 2,
}


def _state(*gates: Gate) -> CircuitState:
    state = CircuitState(
        CircuitDAG(3),
        ResourceBudget(
            max_t_count=7,
            max_depth=12,
            max_gates=20,
            max_two_qubit_count=6,
        ),
    )
    for gate in gates:
        assert state.apply_gate(gate)
    return state


def _analyze(state: CircuitState) -> _Progress:
    names = tuple(gate.gate_type.name for gate in state.dag.gates)
    if "H" in names:
        # The provider must ignore these non-core rows and use identity.
        return _Progress(_Stage.DONE, (3, 5, 4), 0b1111111)
    if "CNOT" in names:
        return _Progress(_Stage.CORE, (3, 5, 4), 0b0000011)
    return _Progress(_Stage.PRE_H, (1, 2, 4), 0)


def _provider(*, target_fingerprint: str = "toffoli-fixture-target"):
    return ToffoliParityFeatureProvider(
        analyzer=_analyze,
        required_phase_terms=_REQUIRED_PHASE_TERMS,
        cnot_basis_distance_to_identity=_BASIS_DISTANCE,
        target_fingerprint=target_fingerprint,
        problem_schema_version="fixture-toffoli-parity-v1",
        qubit_convention="q0 is LSB; controls q0,q1; target q2",
    )


def _feature(provider: ToffoliParityFeatureProvider, vector: np.ndarray, name: str) -> float:
    return float(vector[provider.names.index(name)])


def test_toffoli_parity_provider_has_the_documented_66d_schema_and_exact_potential():
    provider = _provider()
    root = _state()
    core = _state(Gate(GateType.CNOT, (0, 1)))
    done = _state(Gate(GateType.H, (2,)))

    root_features = provider.extract(root, [root, core])
    core_features = provider.extract(core, [root, core])
    done_features = provider.extract(done, [done])

    assert provider.dimension == TOFFOLI_PARITY_FEATURE_DIMENSION == 66
    assert provider.schema_version == TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION
    assert provider.names[-1] == "bias"
    assert root_features.shape == (66,)
    assert root_features.dtype == np.float32
    assert np.isfinite(root_features).all()

    # PRE_H uses the identity basis.  Its three exposed, un-emitted terms are
    # masks 001, 010, and 100; p=0 and e=3/7, hence Phi=.20*(3/7).
    assert _feature(provider, root_features, "remaining_term_001") == 1.0
    assert _feature(provider, root_features, "basis_row_0_mask_001") == 1.0
    assert _feature(provider, root_features, "basis_row_1_mask_010") == 1.0
    assert _feature(provider, root_features, "basis_row_2_mask_100") == 1.0
    assert _feature(provider, root_features, "exposed_remaining_term_001") == 1.0
    assert _feature(provider, root_features, "stage_pre_h") == 1.0
    assert _feature(provider, root_features, "identity_basis") == 1.0
    assert _feature(provider, root_features, "toffoli_parity_potential") == pytest.approx(
        0.20 * (3.0 / 7.0)
    )

    # In CORE, two term bits are emitted, three of five remaining masks are
    # exposed, and the CNOT basis is at the maximum injected distance.
    assert _feature(provider, core_features, "emitted_term_001") == 1.0
    assert _feature(provider, core_features, "emitted_term_010") == 1.0
    assert _feature(provider, core_features, "remaining_term_001") == 0.0
    assert _feature(provider, core_features, "basis_row_0_mask_011") == 1.0
    assert _feature(provider, core_features, "basis_row_1_mask_101") == 1.0
    assert _feature(provider, core_features, "identity_basis_distance_normalized") == 1.0
    assert _feature(provider, core_features, "stage_core") == 1.0
    assert _feature(provider, core_features, "toffoli_parity_potential") == pytest.approx(
        0.55 * (2.0 / 7.0) + 0.20 * (3.0 / 5.0)
    )

    # POST_H/DONE deliberately normalizes back to the identity basis even if
    # the analyzer retains a stale core basis tuple.
    assert _feature(provider, done_features, "stage_done") == 1.0
    assert _feature(provider, done_features, "basis_row_0_mask_001") == 1.0
    assert _feature(provider, done_features, "identity_basis") == 1.0
    assert _feature(provider, done_features, "identity_basis_distance_normalized") == 0.0
    assert _feature(provider, done_features, "toffoli_parity_potential") == pytest.approx(0.8)


def test_toffoli_provider_frontier_reductions_are_order_invariant_and_have_no_node_id():
    provider = _provider()
    root = _state()
    core = _state(Gate(GateType.CNOT, (0, 1)))
    root_node = SearchNode(priority=0.0, state=root, record_id=1)
    duplicate_root = SearchNode(priority=0.0, state=root, record_id=999)
    core_node = SearchNode(priority=0.0, state=core, record_id=2)

    forward = provider.extract(core, [root_node, core_node, duplicate_root])
    backward = provider.extract(core, [duplicate_root, core_node, root_node])
    np.testing.assert_array_equal(forward, backward)
    np.testing.assert_array_equal(
        provider.extract(root, [root_node]), provider.extract(root, [duplicate_root])
    )
    assert provider.frontier_potential([root_node, core_node]) == pytest.approx(
        max(provider.potential(root), provider.potential(core))
    )


def test_provider_metadata_binds_the_phase_problem_target_and_qubit_convention():
    metadata = _provider().metadata()

    assert metadata["feature_schema_version"] == TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION
    assert metadata["feature_dim"] == 66
    assert metadata["target_fingerprint"] == "toffoli-fixture-target"
    assert metadata["problem_schema_version"] == "fixture-toffoli-parity-v1"
    assert metadata["qubit_convention"].startswith("q0 is LSB")
    assert metadata["phase_term_digest"].startswith("sha256:")
    assert metadata["basis_distance_digest"].startswith("sha256:")
    assert metadata["required_phase_masks"] == (1, 2, 3, 4, 5, 6, 7)


def test_default_provider_adapts_the_public_parity_analyzer_module_and_instance():
    """The lazy default and a problem injection share the public analyzer API."""

    problem = ToffoliParityNetworkProblem()
    default_provider = ToffoliParityFeatureProvider()
    injected_provider = ToffoliParityFeatureProvider(problem)
    root = _state()

    np.testing.assert_array_equal(
        default_provider.extract(root), injected_provider.extract(root)
    )
    assert default_provider.metrics(root).stage == "PRE_H"
    assert default_provider.metadata()["problem_schema_version"] == problem.schema_version
    assert (
        default_provider.metadata()["phase_term_digest"]
        == injected_provider.metadata()["phase_term_digest"]
    )


def test_policy_uses_an_explicit_provider_and_preserves_legacy_adapter_behavior():
    provider = _provider()
    root = SearchNode(priority=0.0, state=_state(), record_id=10)
    core = SearchNode(
        priority=0.0,
        state=_state(Gate(GateType.CNOT, (0, 1))),
        record_id=11,
    )
    policy = LinearQPolicy(feature_provider=provider, seed=7)

    assert policy.feature_dim == 66
    assert policy.feature_schema_version == TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION
    assert policy.target_fingerprint == "toffoli-fixture-target"
    metadata = policy.metadata()
    assert metadata["target_fingerprint"] == "toffoli-fixture-target"
    assert metadata["feature_provider_binding_digest"].startswith("sha256:")
    assert policy.weight_digest().startswith("sha256:")

    policy.theta[provider.names.index("toffoli_parity_potential")] = 1.0
    assert policy.select_node([root, core], epsilon=0.0) is core
    assert policy.select_node([core, root], epsilon=0.0) is core

    with pytest.raises(ValueError, match="does not match"):
        LinearQPolicy(feature_dim=LEGACY_FEATURE_DIMENSION, feature_provider=provider)

    legacy_provider = LegacyFeatureProvider()
    legacy_policy = LinearQPolicy(seed=7)
    legacy_state = _state()
    np.testing.assert_array_equal(
        legacy_provider.extract(legacy_state), extract_features(legacy_state)
    )
    assert legacy_policy.feature_dim == LEGACY_FEATURE_DIMENSION
    legacy_policy.theta[0] = 1.0
    with pytest.raises(ValueError, match="non-zero"):
        legacy_policy.bind_feature_provider(provider)

    rebound = LinearQPolicy(seed=7)
    rebound.bind_feature_provider(provider)
    assert rebound.feature_dim == 66
    rebound.theta[0] = 1.0
    with pytest.raises(ValueError, match="non-zero"):
        rebound.bind_feature_provider(_provider(target_fingerprint="other-target"))


def test_toffoli_parity_reward_uses_frontier_potential_deltas_without_hidden_terms():
    provider = _provider()
    model = ToffoliParityRewardModel(
        provider,
        ToffoliParityRewardConfig(
            terminal_bonus=20.0,
            step_cost=0.05,
            potential_scale=4.0,
            dead_end_cost=1.0,
            reward_clip=None,
        ),
    )
    result = model.reward(
        potential_before=0.10,
        potential_after=0.40,
        certified=False,
        dead_end=False,
    )

    assert result.potential_delta == pytest.approx(0.30)
    assert result.raw_reward == pytest.approx(-0.05 + 4.0 * 0.30)
    assert result.reward == pytest.approx(result.raw_reward)
    assert result.selected_node_potential == pytest.approx(0.10)
    assert result.best_generated_child_potential == pytest.approx(0.40)

    terminal = model.reward(
        potential_before=0.40,
        potential_after=0.80,
        certified=True,
        dead_end=True,
    )
    assert terminal.raw_reward == pytest.approx(20.0 - 0.05 + 4.0 * 0.40 - 1.0)
    assert terminal.terminal_bonus == pytest.approx(20.0)
    assert terminal.dead_end_cost == pytest.approx(1.0)
    assert model.metadata()["potential_formula"] == "0.55*p + 0.20*e + 0.25*p*r"
