from __future__ import annotations

import numpy as np

from hybrid_qcs.benchmarks import (
    qft2_matrix,
    qft2_target,
    structured_toffoli_target,
    toffoli_matrix,
)
from hybrid_qcs.certify import certify_state, equal_up_to_global_phase
from hybrid_qcs.model import generate_gates
from hybrid_qcs.search import HybridSearch
from hybrid_qcs.structured_toffoli import (
    StructuredToffoliSearch,
    phase_identity_holds,
)


def _symbolic_record_id(records) -> int:
    return min(
        records,
        key=lambda record: (
            record.symbolic_distance,
            record.state.gate_count,
            record.state.t_count,
            record.state.cnot_count,
            record.record_id,
        ),
    ).record_id


def test_exact_qft2_is_an_unrestricted_native_heldout_target() -> None:
    target = qft2_target()
    assert target.split == "test"
    assert target.family == "unrestricted-native-qft2"
    assert "final native SWAP" in target.convention
    assert not hasattr(target, "hidden_witness")
    assert np.allclose(target.unitary, qft2_matrix(include_output_swap=True))

    search = HybridSearch(target, max_expansions=8_192)
    assert search.actions == generate_gates(2)
    assert target.budget.max_gates == 10
    assert target.budget.max_t_count == 3
    assert target.budget.max_cnot_count == 5


def test_structured_toffoli_stress_test_is_independently_certified() -> None:
    assert phase_identity_holds()
    target = structured_toffoli_target()
    dense_match, error = equal_up_to_global_phase(target.unitary, toffoli_matrix())
    assert dense_match, error
    assert target.family == "structured-toffoli-parity-network"

    search = StructuredToffoliSearch(target, max_expansions=4_096)
    state = search.run_scheduler(_symbolic_record_id)
    assert state is not None
    progress = search.solution_progress()
    assert progress is not None and progress.stage.value == "DONE"
    result = certify_state(target, state)
    assert result.success
    assert result.gate_count == 15
    assert result.t_count == 7
    assert result.cnot_count == 6
