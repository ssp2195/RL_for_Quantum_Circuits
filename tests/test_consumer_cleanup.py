"""Exact algebra, all-outcome circuits, and actual frozen-policy generation.

Analytic constructions are deliberately not labelled learned discoveries.
The generated marker below requires an actual SARSA/LinUCB candidate as well
as the independently constructed and checked reduced candidate menu.
"""
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import copy
import random
import pytest

from hybrid_qcs.cleanup.contract import CleanupProblem, Consumer, Limits, plus, times
from hybrid_qcs.cleanup.benchmarks import make_problem, candidate_problems
from hybrid_qcs.cleanup.consumer import (bits, rank_factors, quadratic_form, factor_consumer,
                                         factor_forms, minimum_live_order, verify_live_certificate,
                                         order_peak, Interrupted)
from hybrid_qcs.cleanup.consumer_compile import compile_factors, verify_factor_receipt
from hybrid_qcs.cleanup.phase_protocol import PhaseProtocol, verify_phase, compose_blocks
from hybrid_qcs.cleanup.consumer_search import synthesize_oracle, optimize_materialization, RESOURCE_KEYS
from hybrid_qcs.cleanup.training import train_cleanup
from hybrid_qcs.cleanup.verify import verify_small_matrix, verify_truth_table_generators
from hybrid_qcs.cleanup.ir import Instruction


def test_binary_factorization_every_two_by_three_matrix():
    for matrix in range(64):
        rows = (matrix & 7, matrix >> 3)
        factors = rank_factors(rows, 3)
        reconstructed = [0, 0]
        for u, v in factors:
            for i in bits(u):
                reconstructed[i] ^= v
        assert tuple(reconstructed) == rows
        expected = 0 if not any(rows) else 1 if 0 in rows or rows[0] == rows[1] else 2
        assert len(factors) == expected


def test_all_four_variable_quadratic_forms():
    monomials = [1 << i for i in range(4)] + [(1 << i) | (1 << j) for i in range(4) for j in range(i+1, 4)]
    for word in range(1 << len(monomials)):
        poly = frozenset(m for i, m in enumerate(monomials) if word >> i & 1)
        form = quadratic_form(poly)
        assert form.polynomial() == poly and len(form.products) <= 2
        assert all(u & ~v and v & ~u for u, v in form.products)


@pytest.mark.generated
@pytest.mark.parametrize('shape', ((1,1), (2,1), (1,2), (2,2), (3,2), (2,3), (4,2)))
@pytest.mark.parametrize('salt', (0,1,2,3))
@pytest.mark.parametrize('storage', ('all','stream','recompute'))
def test_constructed_all_input_all_outcome_oracles(shape, salt, storage, fitted):
    p = make_problem(*shape, name=f'exact-{shape}-{salt}', salt=salt)
    model = fitted[11 if salt%2==0 else 19]
    before = model.digest
    study = synthesize_oracle(p, model, include_learned=True)
    learned = [item for item in study['candidates'] if item['learned_discovery']]
    assert len(learned)==1 and learned[0]['search']['edges']>0
    assert verify_phase(p, learned[0]['protocol'])['valid'] and model.digest==before
    c, receipt = compile_factors(p, storage=storage)
    assert verify_phase(p, c)['valid']
    assert verify_factor_receipt(p, c, receipt)
    assert verify_truth_table_generators(p, c)['valid']
    c.persistent_tail()
    assert PhaseProtocol.from_payload(c.payload()) == c
    assert c.qasm3().startswith('OPENQASM 3.0;')


@pytest.mark.generated
@pytest.mark.parametrize('salt', range(4))
@pytest.mark.parametrize('shape', ((1,1),(2,1),(1,2)))
def test_every_small_native_kraus_branch(shape, salt, fitted):
    p = make_problem(*shape, name='kraus', salt=salt)
    result = synthesize_oracle(p, fitted[11], include_learned=True)
    learned = next(item for item in result['candidates'] if item['learned_discovery'])
    assert learned['search']['edges']>0
    assert verify_small_matrix(p,PhaseProtocol.from_payload(learned['protocol']))['valid']
    c, _ = compile_factors(p)
    assert verify_small_matrix(p, c)['valid']


@pytest.mark.parametrize('seed', range(24))
def test_subset_optimum_against_all_orders(seed):
    rng = random.Random(seed)
    supports = tuple(frozenset(j for j in range(7) if rng.randrange(3)==0) for _ in range(5))
    receipt = minimum_live_order(supports)
    # Independent interval counting on complete orders.
    def brute(order):
        peak = 0
        features = set().union(*supports)
        for t in range(len(order)):
            count = sum(any(f in supports[order[i]] for i in range(t+1)) and
                        any(f in supports[order[i]] for i in range(t,len(order))) for f in features)
            peak = max(peak, 1+count)
        return peak
    assert receipt['peak'] == min(brute(order) for order in permutations(range(5)))
    assert verify_live_certificate(receipt)


def test_streaming_order_can_strictly_improve_peak():
    supports = (frozenset({0,1}), frozenset({2,3}), frozenset({0,1}), frozenset({2,3}))
    proof = minimum_live_order(supports)
    assert order_peak(supports,(0,1,2,3)) == 5
    assert proof['peak'] == 3


@pytest.mark.parametrize('change', ('phase','correction','restore','outcome','width','scope','read_before_measure','unclean_AND'))
def test_reject_bad_complete_protocol(change):
    p = candidate_problems()[0]
    c, _ = compile_factors(p)
    d = c.payload()
    if change == 'phase':
        d['ops'].append(Instruction('Z',(0,),'consumer').payload())
    elif change == 'correction':
        d['ops'].pop(next(i for i,o in enumerate(d['ops']) if o['guard'] is not None))
    elif change == 'restore':
        d['ops'].pop(next(i for i,o in enumerate(d['ops']) if o['kind']=='CX'))
    elif change == 'outcome':
        next(o for o in d['ops'] if o['guard'] is not None)['guard']=12345
    elif change == 'width':
        d['width'] = p.n
    elif change == 'scope':
        d['scope'] = 'globally optimal'
    elif change == 'read_before_measure':
        index = next(i for i,o in enumerate(d['ops']) if o['guard'] is not None)
        d['ops'].insert(0,d['ops'].pop(index))
    else:
        index = next(i for i,o in enumerate(d['ops']) if o['kind']=='AND')
        d['ops'].insert(index+1,copy.deepcopy(d['ops'][index]))
    assert not verify_phase(p,d)['valid']


@pytest.mark.parametrize('field', ('peak','values','order','supports','global_oracle_optimality'))
def test_reject_bad_order_certificate(field):
    c = minimum_live_order((frozenset({0}),frozenset({0,1}),frozenset({2})))
    if field=='peak': c[field]+=1
    elif field=='values': c[field][0]+=1
    elif field=='order': c[field]=[0,0,0]
    elif field=='supports': c[field]=[[0]]
    else: c[field]=True
    assert not verify_live_certificate(c)


@pytest.mark.parametrize('field', ('predicted_t','predicted_peak','feature_computations','global_oracle_optimality','schema','feature_table','forms'))
def test_reject_bad_factor_receipt(field):
    p=candidate_problems()[-1]; c,r=compile_factors(p)
    if field in ('predicted_t','predicted_peak','feature_computations'):r[field]+=1
    elif field=='global_oracle_optimality':r[field]=True
    elif field=='schema':r[field]='native_optimality'
    else:r[field]=[]
    assert not verify_factor_receipt(p,c,r)


@pytest.fixture(scope='module')
def fitted():
    result = {}
    for seed in (11,19):
        model, log = train_cleanup(seed, (8,12,4))
        assert model.frozen and model.stage_episodes=={'outer':12,'inner':12}
        assert model.outer_updates and min(model.updates)>0
        result[seed]=model
    return result


@pytest.mark.generated
@pytest.mark.parametrize('seed',(11,19))
@pytest.mark.parametrize('shape',((3,1),(4,2),(5,2),(6,3),(8,3)))
@pytest.mark.parametrize('salt',range(4))
def test_after_both_training_stages_real_circuits_and_learned_schedule(seed,shape,salt,fitted):
    p=make_problem(*shape,name=f'frozen-{shape}-{salt}',salt=salt)
    model=fitted[seed];before=model.digest
    out=synthesize_oracle(p,model,include_learned=True)
    assert out['status']=='certified_upper_bound' and out['menu_complete'] and out['timely']
    assert model.digest==before
    learned=[c for c in out['candidates'] if c['learned_discovery']]
    assert len(learned)==1 and learned[0]['search']['edges']>0
    assert learned[0]['search']['policy_digest']==before
    assert verify_phase(p,learned[0]['protocol'])['valid']
    assert verify_phase(p,out['selected']['protocol'])['valid']
    objective=out['objective']
    selected=tuple(out['selected']['verification']['resources'][k] for k in objective)
    for c in out['candidates']:
        assert selected<=tuple(c['verification']['resources'][k] for k in objective)
    assert verify_truth_table_generators(p,PhaseProtocol.from_payload(out['selected']['protocol']))['valid']


@pytest.mark.parametrize('problem',candidate_problems(),ids=lambda p:p.name)
def test_analytic_workspace_is_closed_without_policy_search(problem,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('the known optimum should not invoke search')
    monkeypatch.setattr('hybrid_qcs.cleanup.consumer_search.CleanupSearch',forbidden)
    out=optimize_materialization(problem)
    assert out['status']=='optimal_workspace_within_architecture'
    assert not out['learned_discovery'] and out['search']==[]


def test_full_bank_and_complete_oracle_scopes_are_different():
    p=candidate_problems()[1].with_aux_cap(1)
    assert optimize_materialization(p)['status']=='infeasible_within_theorem_architecture'
    out=synthesize_oracle(p)
    assert out['status']=='certified_upper_bound'
    assert out['selected']['verification']['resources']['peak_aux']==1
    assert not out['global_oracle_optimality']


def test_cancel_and_exact_domain_limits():
    supports=(frozenset({0}),)*17
    with pytest.raises(ValueError): minimum_live_order(supports)
    with pytest.raises(Interrupted): minimum_live_order((frozenset({0}),),cancel=lambda:True)
    p=candidate_problems()[0]
    assert synthesize_oracle(p,cancel=lambda:True)['status']=='unknown'
    assert optimize_materialization(p,cancel=lambda:True)['status']=='unknown'
    c,r=compile_factors(p,max_factors=0)
    assert r['schedule_certificate'] is None and 'upper_bound' in r['schedule_status']
    assert verify_factor_receipt(p,c,r)


def test_invalid_objectives_and_untrained_policy():
    from hybrid_qcs.cleanup.policy import CleanupHierarchy
    p=candidate_problems()[0]
    with pytest.raises(ValueError): synthesize_oracle(p,objective=('magic_guess',))
    with pytest.raises(ValueError): synthesize_oracle(p,include_learned=True)
    with pytest.raises(ValueError): synthesize_oracle(p,CleanupHierarchy())
    with pytest.raises(ValueError): synthesize_oracle(p,seconds=float('nan'))


def test_clean_block_composition_with_shared_original_inputs():
    # Blocks share the original a and b wires but not dirty auxiliary state.
    one=CleanupProblem('one',1,1,Consumer((('f0_0',),)))
    two=CleanupProblem('two',1,1,Consumer((('a','x0'),('f0_0',))))
    target=CleanupProblem('composed',1,2,Consumer((('f0_0',),('a','x1'),('f0_1',))))
    c1,_=compile_factors(one);c2,_=compile_factors(two)
    c=compose_blocks(target,((one,c1,(0,1,2)),(two,c2,(0,1,3))))
    v=verify_phase(target,c)
    assert v['valid'] and v['resources']['peak_aux']==1 and v['resources']['t_count']==8
    assert verify_small_matrix(target,c)['valid']
    assert verify_truth_table_generators(target,c)['valid']
    with pytest.raises(ValueError):compose_blocks(target,((one,c1,(0,0,2)),))


def test_clifford_only_consumer_needs_no_temporary_bank():
    p=CleanupProblem('clifford',4,3,Consumer((('a','x1'),('b0','b2'),())))
    c,r=compile_factors(p)
    v=verify_phase(p,c)
    assert v['valid'] and v['resources']['peak_aux']==0 and v['resources']['t_count']==0
    assert verify_factor_receipt(p,c,r)


def test_actual_overlapping_feature_circuits_need_fewer_slots():
    from hybrid_qcs.cleanup.consumer_diagnostics import overlapping_problem
    p=overlapping_problem()
    natural,_=compile_factors(p,order=(0,1,2,3))
    optimal,receipt=compile_factors(p)
    a,b=verify_phase(p,natural),verify_phase(p,optimal)
    assert a['valid'] and b['valid']
    assert a['resources']['peak_aux']==5 and b['resources']['peak_aux']==3
    assert a['resources']['t_count']==b['resources']['t_count']==32
    assert verify_factor_receipt(p,optimal,receipt)
    assert verify_truth_table_generators(p,optimal)['valid']


def test_latency_first_keeps_larger_workspace_alternative():
    p=candidate_problems()[-1]
    compact=synthesize_oracle(p)
    fast=synthesize_oracle(p,objective=('worst_case_ticks','t_count','peak_aux'))
    c=compact['selected']['verification']['resources']
    f=fast['selected']['verification']['resources']
    assert f['worst_case_ticks']<c['worst_case_ticks']
    assert f['peak_aux']>c['peak_aux']


def test_tight_round_cap_can_prefer_all_features():
    p=replace(candidate_problems()[3],limits=Limits(max_rounds=5,max_t=36))
    out=synthesize_oracle(p)
    assert out['status']=='certified_upper_bound'
    assert out['selected']['method']=='rank_all'
    assert out['selected']['verification']['resources']['measurement_rounds']==5


def test_no_candidate_is_not_an_unrestricted_infeasibility_proof():
    p=replace(candidate_problems()[3],limits=Limits(max_rounds=0,max_t=0,max_aux=0))
    out=synthesize_oracle(p)
    assert out['status']=='unknown' and not out['global_oracle_optimality']


def test_analytic_depth_failure_is_not_infeasibility():
    p=replace(candidate_problems()[0],limits=Limits(max_depth=0))
    out=optimize_materialization(p)
    assert out['status']=='unknown' and 'cap' in out['reason']
