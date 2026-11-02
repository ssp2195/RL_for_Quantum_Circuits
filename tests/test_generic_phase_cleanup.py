"""Regression tests for exact generic Boolean phase oracles and causal cleanup."""
from itertools import product

import pytest

from hybrid_qcs.cleanup.contract import Limits, times
from hybrid_qcs.cleanup.contextual import (
    LiveStage, MeasuredValue, causal_corrections, correction_rank,
    necessary_peak_ancillas, phase_generators,
)
from hybrid_qcs.cleanup.general_boolean import (
    BooleanPhaseProblem, compile_boolean_phase, optimize_boolean_phase,
)
from hybrid_qcs.cleanup.phase_protocol import verify_phase
from hybrid_qcs.cleanup.ir import local_word
from hybrid_qcs.native_exact import ExactMatrix, ONE, ZERO


def bit(i):
    return frozenset({1 << i})


def test_causal_future_value_and_no_time_travel():
    x, y, z = (bit(i) for i in range(3))
    cubic = times(times(x, y), z)
    baseline = LiveStage((x, y, z))
    later = LiveStage((x, y, z, times(x, y)))
    measurements = (MeasuredValue(cubic, 0),)
    assert correction_rank(baseline, measurements) == 1
    assert causal_corrections((baseline,), measurements) is None
    corrections = causal_corrections((baseline, later), measurements)
    assert corrections is not None
    assert any(op.stage == 1 and op.kind == 'CZ' for op in corrections)
    assert causal_corrections((later, baseline), (MeasuredValue(cubic, 1),)) is None
    assert necessary_peak_ancillas(baseline, measurements, 2) == 1


def test_causal_rank_family():
    q = 3
    x = [bit(i) for i in range(q)]
    y = [bit(q + i) for i in range(2 * q)]
    initial = LiveStage(tuple(x + y))
    garbage = tuple(MeasuredValue(times(x[i], times(y[2*j], y[2*j+1])), 0)
                    for i, j in product(range(q), repeat=2))
    assert correction_rank(initial, garbage) == q * q
    stages = tuple(LiveStage(initial.values + (times(y[2*j], y[2*j+1]),)) for j in range(q))
    witness = causal_corrections(stages, garbage)
    assert witness is not None
    assert len(witness) == q * q
    assert necessary_peak_ancillas(initial, garbage, q) == 1


def test_truth_table_anf_roundtrip():
    values = (0, 0, 0, 1, 0, 1, 1, 1)
    problem = BooleanPhaseProblem.from_truth_table('majority', values)
    for x in range(8):
        got = sum((x & m) == m for m in problem.target_polynomial) % 2
        assert got == values[x]


@pytest.mark.parametrize('n', (3, 4, 5))
@pytest.mark.parametrize('mode', ('measured', 'direct'))
def test_arbitrary_degree_conjunction_has_exact_measured_branches(n, mode):
    problem = BooleanPhaseProblem(f'and-{n}', n, frozenset({(1 << n) - 1}))
    circuit = compile_boolean_phase(problem, (mode,))
    checked = verify_phase(problem, circuit)
    assert checked['valid'], checked
    counts = checked['resources']
    assert counts['peak_aux'] == n - (3 if mode == 'direct' else 2)
    assert counts['t_count'] == 4 * (n - (3 if mode == 'direct' else 2)) + (7 if mode == 'direct' else 0)
    assert counts['aux_at_end'] == 0
    assert counts['cleanup_t'] == 0
    circuit.persistent_tail()


def test_ancilla_first_exact_menu_and_t_budget():
    p = BooleanPhaseProblem('ccz', 3, frozenset({7}))
    result = optimize_boolean_phase(p)
    assert result['status'] == 'certified_finite_menu_optimum'
    assert result['candidate_count'] == 2
    assert result['selected']['resources']['peak_aux'] == 0
    assert result['selected']['resources']['t_count'] == 7

    capped = BooleanPhaseProblem('ccz-cap', 3, frozenset({7}), Limits(max_t=4))
    result = optimize_boolean_phase(capped)
    assert result['selected']['resources']['peak_aux'] == 1
    assert result['selected']['resources']['t_count'] == 4


def test_multiple_monomials_and_direct_path():
    p = BooleanPhaseProblem('combined', 4, frozenset({0, 1, 3, 7, 15}))
    result = optimize_boolean_phase(p)
    assert result['candidate_count'] == 4
    for row in result['pareto']:
        assert verify_phase(p, row['protocol'])['valid']


def test_ccz_word_is_exact_phase_on_all_eight_states():
    state = ExactMatrix.identity(3)
    for kind, wires in local_word('CCZ'):
        state = state.apply(kind, wires)
    rows = tuple(tuple((tuple(-v for v in ONE) if i == j == 7 else ONE)
                       if i == j else ZERO for j in range(8)) for i in range(8))
    assert state == ExactMatrix(rows)


def test_exact_menu_rejects_unstated_limit():
    p = BooleanPhaseProblem('many', 5, frozenset(m for m in range(32) if m.bit_count() >= 3))
    with pytest.raises(ValueError, match='exceeds'):
        optimize_boolean_phase(p, max_nonlinear_terms=1)
