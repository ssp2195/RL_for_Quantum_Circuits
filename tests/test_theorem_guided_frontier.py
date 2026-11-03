"""End-to-end M1-guided synthesis and M2-pruned deterministic frontier tests."""
from itertools import product
import pytest

from hybrid_qcs.cleanup.contextual import (
    LiveStage, MeasuredValue, necessary_peak_ancillas, correction_rank)
from hybrid_qcs.cleanup.contract import times
from hybrid_qcs.cleanup.general_boolean import BooleanPhaseProblem, compile_boolean_phase
from hybrid_qcs.cleanup.generic_search import (
    GenericFrontierSearch, train_generic_hierarchy)
from hybrid_qcs.cleanup.phase_protocol import verify_phase
from hybrid_qcs.cleanup.staged_frontier import (
    CausalCorrectionFrontier, compile_staged_corrections)


@pytest.mark.parametrize('degree', (3, 4, 5))
@pytest.mark.parametrize('mode', ('measured', 'direct'))
def test_m1_immediate_feedback_preserves_complete_oracle(degree, mode):
    problem = BooleanPhaseProblem('monomial', degree,
                                  frozenset({(1 << degree)-1}))
    baseline = compile_boolean_phase(problem, (mode,))
    causal = compile_boolean_phase(problem, (mode,), m1_guided=True)
    a = verify_phase(problem, baseline)
    b = verify_phase(problem, causal)
    assert a['valid'] and b['valid'], (a, b)
    assert a['resources']['peak_aux'] == b['resources']['peak_aux']
    assert a['resources']['t_count'] == b['resources']['t_count']
    assert a['resources']['cnot'] == b['resources']['cnot']
    causal.persistent_tail()
    if mode == 'measured' or degree > 3:
        assert any(op.stage == 'correct_m1' for op in causal.ops)


def test_m1_rejects_invalid_correction_grammar():
    from hybrid_qcs.cleanup.causal_protocol import causalize_immediate_cleanup
    from hybrid_qcs.cleanup.ir import Instruction
    with pytest.raises(ValueError, match='expected'):
        causalize_immediate_cleanup((
            Instruction('AND',(0,1,3),'prepare'),
            Instruction('MX',(3,),'measure',outcome=0),
            Instruction('Z',(2,),'incorrect')
        ),3)


def test_m1_generic_frontier_preserves_finite_library_results():
    problem=BooleanPhaseProblem('two-cubics',4,frozenset({7,15}))
    control=GenericFrontierSearch(problem,m1_guided=False).run()
    guided=GenericFrontierSearch(problem,m1_guided=True).run()
    assert control['complete_library_search'] and guided['complete_library_search']
    assert control['candidate_count'] == guided['candidate_count'] == 4
    assert control['selected']['resources']['peak_aux'] == guided['selected']['resources']['peak_aux']
    assert control['selected']['resources']['t_count'] == guided['selected']['resources']['t_count']
    assert verify_phase(problem,guided['selected']['protocol'])['valid']


def test_m1_generic_search_retains_sarsa_linucb_roles():
    train=(BooleanPhaseProblem('train-cubic',3,frozenset({7})),
           BooleanPhaseProblem('train-quartic',4,frozenset({15})))
    model=train_generic_hierarchy(train,seed=11,episodes=8)
    heldout=BooleanPhaseProblem('heldout',4,frozenset({7,15}))
    evaluation=GenericFrontierSearch(heldout,m1_guided=True).run(model)
    assert model.frozen and model.outer_updates and model.inner_updates
    assert evaluation['m1_guided'] and evaluation['complete_library_search']
    assert evaluation['candidate_count']==4


def test_m2_capacity_rejects_infeasible_root_before_expansion():
    # q=4 cubic byproducts have rank 16 modulo quadratic phases.
    # With t=12 fixed input wires, one helper in one stage has capacity 13.
    q=4
    baseline=LiveStage(tuple(frozenset({1<<i}) for i in range(3*q)))
    byproducts=tuple(MeasuredValue(frozenset({(1<<i) |
                       (1<<(q+2*j)) | (1<<(q+2*j+1))}),0)
                      for j in range(q) for i in range(q))
    assert correction_rank(baseline,byproducts) == 16
    assert necessary_peak_ancillas(baseline,byproducts,1) == 2
    guarded=CausalCorrectionFrontier(q,1,1,m2_enabled=True).run()
    unguarded=CausalCorrectionFrontier(q,1,1,m2_enabled=False).run()
    assert guarded['status']==unguarded['status']=='infeasible_in_stage_grammar'
    assert guarded['expanded_edges']==0 and guarded['m2_pruned_records']==1
    assert unguarded['expanded_edges']>0 and unguarded['m2_pruned_records']==0


@pytest.mark.parametrize('q,stages,cap,expected_peak', (
    (2,1,2,2),(2,2,1,1),(3,1,3,3),(3,3,1,1),
    (4,1,4,4),(4,2,2,2),(4,4,1,1),
))
def test_m1_m2_stage_schedule_and_independent_branch_verification(
    q,stages,cap,expected_peak
):
    result=CausalCorrectionFrontier(q,stages,cap).run()
    assert result['status']=='certified_finite_stage_optimum'
    assert result['selected']['peak_correction_ancillas']==expected_peak
    assert result['verified_schedules']>=1
    assert result['selected']['exact_all_branches']
    # The same accepted stage schedule is sound without M2.
    control=CausalCorrectionFrontier(q,stages,cap,m2_enabled=False).run()
    assert control['status']==result['status']
    assert control['selected']['resources']==result['selected']['resources']
    assert result['expanded_edges'] <= control['expanded_edges']
    problem,circuit=compile_staged_corrections(q,result['selected']['schedule'])
    assert verify_phase(problem,circuit)['valid']


def test_m2_is_necessary_not_sufficient():
    # For q=3, L=1 the M2 capacity bound accepts one helper,
    # yet the exact declared correction-stage grammar requires all three.
    p=CausalCorrectionFrontier(3,1,1).run()
    assert p['status']=='infeasible_in_stage_grammar'
    assert p['global_oracle_optimality'] is False


def test_reject_incomplete_or_reused_contexts():
    with pytest.raises(ValueError,match='every measured'):
        compile_staged_corrections(3,(1,2))
    with pytest.raises(ValueError,match='cannot be prepared'):
        compile_staged_corrections(2,(1,1,2))
