"""Post-training generation dominates this new module; safety tests are retained.

These are protocol-planner tests, not evidence that a generic 39-qubit native
search converges. Every generated result is checked as a complete instrument.
"""
from dataclasses import replace
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from hybrid_qcs.model import HybridState, Budget, Gate
from hybrid_qcs.native_policy import NativeHierarchy
from hybrid_qcs.resource_search import WorkLimits
from hybrid_qcs.cleanup.contract import CleanupProblem, Consumer, Limits, Layout, ASSUMPTIONS
from hybrid_qcs.cleanup.benchmarks import candidate_problems, training_problems, make_problem
from hybrid_qcs.cleanup.ir import (Protocol, Instruction, ResourceState, local_hybrid, local_word,
                                  append_event, instructions, resources)
from hybrid_qcs.cleanup.plan import Plan, Action, eligible, transition, compile_deterministic
from hybrid_qcs.cleanup.policy import CleanupHierarchy
from hybrid_qcs.cleanup.training import train_cleanup
from hybrid_qcs.cleanup.search import CleanupSearch, optimize_cleanup
from hybrid_qcs.cleanup.verify import (verify_protocol, verify_small_matrix, verify_truth_table_generators,
                                      primitive_receipts, workspace_certificate, verify_workspace_certificate)
from hybrid_qcs.cleanup.reference import compile_bank_free, verify_reference, PhaseReference

PS=candidate_problems()


@pytest.fixture(scope='module')
def fitted():
    models={}
    for seed in (11,19,23,31,47):
        model,log=train_cleanup(seed,(8,12,4))
        assert model.frozen and model.episodes==24 and model.outer_updates>0 and min(model.updates)>0
        assert model.stage_episodes=={'outer':12,'inner':12}
        assert not(set(log['training_oracle_digests'])&set(log['test_oracle_digests']))
        models[seed]=model
    return models


@pytest.mark.generated
@pytest.mark.parametrize('problem',PS,ids=lambda p:p.name)
@pytest.mark.parametrize('seed',(11,19,23,31,47))
@pytest.mark.parametrize('variant',('theorem','control_only','coherent'))
def test_post_training_complete_oracle(problem,seed,variant,fitted):
    model=fitted[seed];before=model.digest
    sides=('row',)if variant=='control_only'else('row','column')
    mode='coherent'if variant=='coherent'else'measured'
    out=CleanupSearch(problem,WorkLimits(4096,20000,5,5),sides=sides,mode=mode).run(model)
    assert out['status']=='feasible' and out['timely']
    assert not out['audit_witness_used'] and out['policy_frozen'] and model.digest==before
    prot=Protocol.from_payload(out['protocol']);checked=verify_protocol(problem,prot)
    assert checked['valid'] and checked['resources']==out['verification']['resources']
    assert checked['resources']['aux_at_end']==0
    truth=verify_truth_table_generators(problem,prot)
    assert truth['valid']
    if mode=='measured':
        assert checked['resources']['cleanup_t']==0 and checked['resources']['measurement_rounds']==2
        cert=workspace_certificate(problem,prot)
        assert verify_workspace_certificate(problem,prot,cert)['valid']
        assert not cert['global_oracle_optimality']
    else:assert checked['resources']['measurements']==0


@pytest.mark.generated
@pytest.mark.parametrize('problem',PS,ids=lambda p:p.name)
@pytest.mark.parametrize('seed',(11,19,23))
@pytest.mark.parametrize('side',('row','column'))
def test_post_training_layout_tradeoff(problem,seed,side,fitted):
    out=CleanupSearch(problem,WorkLimits(2048,10000,5,5),sides=(side,)).run(fitted[seed])
    assert out['status']=='feasible'
    r=out['verification']['resources'];k=problem.r if side=='row'else problem.m
    assert r['peak_aux']==problem.r*problem.m+k
    assert r['t_count']==4*(problem.r*problem.m+k)
    assert verify_protocol(problem,out['protocol'])['valid']


@pytest.mark.generated
@pytest.mark.parametrize('problem',PS,ids=lambda p:p.name)
@pytest.mark.parametrize('seed',(11,19,23))
@pytest.mark.parametrize('primitive',('and4','ccx7'))
def test_post_training_native_primitive_lowering(problem,seed,primitive,fitted):
    out=CleanupSearch(problem,WorkLimits(2048,10000,5,5),primitive=primitive).run(fitted[seed])
    assert out['status']=='feasible'
    prot=Protocol.from_payload(out['protocol'])
    checked=verify_protocol(problem,prot)
    assert checked['valid'] and checked['resources']['t_count']==(4 if primitive=='and4'else 7)*checked['resources']['forward_and']
    assert prot.qasm3().count('= measure')==checked['resources']['measurements']
    assert 'reset ' not in prot.qasm3()  # reset is explicitly H S S H under the result


@pytest.mark.generated
@pytest.mark.parametrize('problem',PS,ids=lambda p:p.name)
@pytest.mark.parametrize('seed',(11,19,23,31,47))
def test_post_training_tight_workspace_optimum(problem,seed,fitted):
    p=problem.with_aux_cap(problem.workspace_lower_bound)
    out=optimize_cleanup(p,fitted[seed],limits=WorkLimits(2048,10000,5,5))
    assert out['status']=='optimal_workspace_within_architecture'
    assert out['verification']['resources']['peak_aux']==problem.workspace_lower_bound
    assert verify_workspace_certificate(p,out['protocol'],out['certificate'])['valid']
    assert eligible(p,Plan(),('row',))==()  # all five r>m; restricted row baseline cannot fit


@pytest.mark.generated
@pytest.mark.parametrize('shape',((1,1),(2,1),(1,2)))
@pytest.mark.parametrize('primitive',('and4','ccx7'))
def test_post_training_all_dense_kraus_branches(shape,primitive,fitted):
    p=make_problem(*shape,name=f'matrix-{shape}-{primitive}',salt=1)
    out=CleanupSearch(p,primitive=primitive).run(fitted[11])
    assert out['status']=='feasible'
    prot=Protocol.from_payload(out['protocol']);checked=verify_small_matrix(p,prot)
    assert checked['valid'] and checked['branches']==1<<(p.r*p.m+prot.layout.k)


@pytest.mark.generated
@pytest.mark.parametrize('problem',PS,ids=lambda p:p.name)
@pytest.mark.parametrize('seed',(11,19,23))
def test_post_training_optimizer_frozen_rounds(problem,seed,fitted):
    m=fitted[seed];before=m.digest
    out=optimize_cleanup(problem,m)
    assert out['status']=='optimal_workspace_within_architecture' and out['timely']
    assert out['policy_digest']==before==m.digest
    assert not out['global_oracle_optimality']
    assert all(not r['audit_witness_used']for r in out['rounds'])
    assert verify_protocol(problem,out['protocol'])['valid']


@pytest.mark.parametrize('primitive',('and4','ccx7'))
def test_exact_primitive_and_scalar_premises(primitive):
    assert primitive_receipts(primitive)['MX_reset_exact']
    h=local_hybrid('AND',primitive)
    assert type(h)is HybridState and h.num_qubits==3
    assert h.t_count==(4 if primitive=='and4'else 7)
    assert h.materialize_dag().gates
    # Four-T primitive is NOT represented as a full arbitrary-target Toffoli.
    assert primitive_receipts(primitive)['scope'].startswith('exact integer')


@pytest.mark.parametrize('shape',((1,1),(2,2),(3,2),(8,3)))
def test_persistent_quantum_classical_dag(shape):
    p=make_problem(*shape,name='dag',salt=2);prot=compile_deterministic(p);tail=None
    for op in prot.ops:tail=append_event(tail,op,prot.primitive)
    assert instructions(tail)==prot.ops
    seen=0
    while tail:
        assert all(i<tail.index for i in tail.parents+tail.classical_parents)
        assert all(isinstance(b,HybridState)and b.num_qubits<=3 for b in tail.coherent_blocks)
        if tail.op.guard is not None:assert tail.classical_parents;seen+=1
        tail=tail.previous
    assert seen==p.r*p.m+min(p.r,p.m)


@pytest.mark.parametrize('change',('omit_correction','wrong_guard','early_helper','wrong_control','omit_reset','bad_primitive','wrong_consumer','extra_phase'))
def test_reject_invalid_protocol(change):
    p=PS[0];d=compile_deterministic(p).payload();ops=d['ops']
    if change=='omit_correction':ops.pop(next(i for i,x in enumerate(ops)if x['stage']=='correct_products'))
    elif change=='wrong_guard':next(x for x in ops if x['guard']is not None)['guard']=1000
    elif change=='early_helper':ops.insert(0,ops.pop(next(i for i,x in enumerate(ops)if x['stage']=='measure_helpers')))
    elif change=='wrong_control':next(x for x in ops if x['kind']=='AND')['wires'][0]=2
    elif change=='omit_reset':next(x for x in ops if x['kind']=='MX')['kind']='MZ'
    elif change=='bad_primitive':d['primitive']='relative_phase_unverified'
    elif change=='wrong_consumer':next(x for x in ops if x['stage']=='consumer')['wires'][0]=0
    elif change=='extra_phase':ops.append(Instruction('Z',(0,),'undeclared').payload())
    assert not verify_protocol(p,d)['valid']


@pytest.mark.parametrize('field,value',(('global_oracle_optimality',True),('lower_bound',0),('upper_bound',0),
                                        ('optimal_within_architecture',False),('assumptions',[]),('kind','native-hybrid-closed-cover-v1')))
def test_theorem_scope_is_not_forgeable(field,value):
    p=PS[0];pr=compile_deterministic(p);c=workspace_certificate(p,pr);c[field]=value
    assert not verify_workspace_certificate(p,pr,c)['valid']


@pytest.mark.parametrize('bad',('partial_cube','phase_mode','assumptions','measurement_permission'))
def test_invalid_source_contract(bad):
    d=PS[0].manifest()
    if bad=='partial_cube':d['input_domain']='x0=a'
    elif bad=='phase_mode':d['phase_mode']='projective'
    elif bad=='assumptions':d['assumptions']=[]
    else:d['measurement_permission']=False
    with pytest.raises(ValueError):CleanupProblem.from_manifest(d)


def test_reject_future_measurement_and_dirty_target():
    with pytest.raises(ValueError):append_event(None,Instruction('CZ',(0,1),'correct',guard=0),'and4')
    p=PS[0];pr=compile_deterministic(p);ops=list(pr.ops)
    product_pos=next(i for i,o in enumerate(ops)if o.kind=='AND'and o.wires[2]in pr.layout.products)
    ops.insert(0,ops.pop(product_pos))
    assert not verify_protocol(p,replace(pr,ops=tuple(ops)))['valid']


def test_no_measurement_in_HybridState():
    from hybrid_qcs.model import HybridState
    assert HybridState.identity(1,Budget(20,20,20,20)).apply(Gate('MX',(0,))) is None


def test_no_dense_global_allocation(monkeypatch):
    from hybrid_qcs.clifford_lift import CliffordColumn
    original=CliffordColumn.identity.__func__
    def guarded(cls,n):
        assert n<=3, 'structured cleanup must not allocate a global Clifford column'
        return original(cls,n)
    monkeypatch.setattr(CliffordColumn,'identity',classmethod(guarded))
    for p in PS:
        out=CleanupSearch(p).run(scheduler='untrained')
        assert out['status']=='feasible' and not out['profile']['global_dense_arrays_allocated']


def test_independent_checker_does_not_invoke_search_or_HybridState(monkeypatch):
    p=PS[-1];prot=compile_deterministic(p)
    def forbidden(*a,**k):raise AssertionError('checker used producer implementation')
    monkeypatch.setattr(HybridState,'apply',forbidden);monkeypatch.setattr(CleanupSearch,'run',forbidden)
    assert verify_protocol(p,prot)['valid']
    assert verify_truth_table_generators(p,prot)['valid']


def test_checkpoint_contract_and_read_only(fitted,tmp_path):
    m=fitted[11];path=tmp_path/'m.json';m.save(path);loaded=CleanupHierarchy.load(path)
    assert loaded.digest==m.digest
    with pytest.raises(ValueError):loaded.update_outer(np.zeros(20),1)
    with pytest.raises(ValueError):loaded.w[0]=1
    data=json.loads(path.read_text());data['schema']='native-hybrid-frontier-v1';path.write_text(json.dumps(data))
    with pytest.raises(ValueError):CleanupHierarchy.load(path)
    with pytest.raises(TypeError):CleanupSearch(PS[0]).run(NativeHierarchy().freeze())
    with pytest.raises(ValueError):CleanupSearch(PS[0]).run(CleanupHierarchy().freeze())


def test_complete_training_and_actual_next_context():
    m,log=train_cleanup(101,(3,4,2))
    assert m.episodes==9 and m.outer_updates>0 and min(m.updates)>0
    assert any(t['next_record']!=t['record']+1 for e in log['episodes']for t in e['transitions']if t['next_record']is not None)
    for e in log['episodes']:
        for t in e['transitions']:
            if t['terminal']:assert t['potential_after']==0 and t['next_outer_context']is None
            if e['stage']=='inner':assert t['inner_response']is not None
    assert not log['test_inputs_used_for_training'] and not log['reference_circuits_used_for_training']


@pytest.mark.parametrize('kind',('cancel','zero_edges','records','workspace','t_count'))
def test_limits_never_become_unrestricted_infeasibility(kind):
    p=PS[0]
    if kind=='cancel':out=CleanupSearch(p,cancel=lambda:True).run(scheduler='untrained')
    elif kind=='zero_edges':out=CleanupSearch(p,WorkLimits(0,10,1,1)).run(scheduler='untrained')
    elif kind=='records':out=CleanupSearch(p,WorkLimits(100,1,1,1)).run(scheduler='untrained')
    elif kind=='t_count':out=CleanupSearch(replace(p,limits=Limits(max_t=0)),WorkLimits(50,100,1,1)).run(scheduler='untrained')
    else:
        out=optimize_cleanup(p.with_aux_cap(p.workspace_lower_bound-1),scheduler='untrained')
        assert out['status']=='infeasible_within_theorem_architecture' and not out['global_oracle_optimality'];return
    assert out['status']=='unknown' and out['protocol']is None


def test_fairness_and_panel_do_not_discard_frontier(fitted):
    e=CleanupSearch(PS[-1],WorkLimits(100,1000,5,5),panel_size=2,fairness=3)
    out=e.run(fitted[11]);assert out['profile']['fairness_steps']>0
    assert out['profile']['max_scored_panel']<=2
    assert out['profile']['peak_frontier']>2


@pytest.mark.parametrize('problem',PS,ids=lambda p:p.name)
def test_stronger_bank_free_control_and_scope(problem):
    pr=compile_bank_free(problem);v=verify_reference(problem,pr)
    assert v['valid'] and not v['global_oracle_optimality']
    assert PhaseReference.from_payload(pr.payload()).digest==pr.digest
    assert not verify_protocol(problem,pr.payload())['valid']
    assert not verify_workspace_certificate(problem,pr.payload(),{})['valid']


def test_parity_consumer_counterexample_not_globally_pruned():
    p=CleanupProblem('bankfree-counterexample',3,2,Consumer(tuple((f'f{i}_{j}',)for i in range(3)for j in range(2))))
    ref=compile_bank_free(p);v=verify_reference(p,ref)
    assert v['valid'] and v['resources']['peak_aux']<p.workspace_lower_bound
    bound=workspace_certificate(p,compile_deterministic(p))
    assert bound['optimal_within_architecture'] and not bound['global_oracle_optimality']


def test_hardware_latency_is_separate_from_gate_count():
    from hybrid_qcs.cleanup.contract import Hardware
    p=PS[0];r=verify_protocol(p,compile_deterministic(p))['resources']
    q=replace(p,hardware=Hardware(1,100,20));s=verify_protocol(q,compile_deterministic(q))['resources']
    assert r['t_count']==s['t_count'] and r['native_gates']==s['native_gates']
    assert s['worst_case_ticks']>r['worst_case_ticks']


def test_expected_reset_cost_is_explicit():
    p=PS[0];pr=compile_deterministic(p);r=verify_protocol(p,pr)['resources'];m=r['measurements']
    from fractions import Fraction
    assert Fraction(r['expected_native_gates'])==r['native_gates']-Fraction(7*m,2)
    # Conditional CZ = 3 gates and reset X = 4 gates, each activated with p=1/2.


def test_empty_and_constant_consumers_are_exact():
    for terms in ((),((),)):
        p=CleanupProblem('constant',1,1,Consumer(terms));pr=compile_deterministic(p)
        assert verify_protocol(p,pr)['valid'] and verify_small_matrix(p,pr)['valid']
