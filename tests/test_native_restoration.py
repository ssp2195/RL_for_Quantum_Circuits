"""Restoration tests: most cases actually generate native circuits AFTER training.

Hard benchmark attempts have separate tests and do not masquerade as synthesis
successes. Constructive reference replay is explicitly NOT a learned outcome.
"""
from __future__ import annotations
from dataclasses import replace
from itertools import product
import json
import random
import numpy as np
import pytest
from hybrid_qcs.model import Budget, Gate, HybridState, generate_gates
from hybrid_qcs.ancilla_contract import AncillaContract, PhaseMode
from hybrid_qcs.certify import unitary_from_gates
from hybrid_qcs.clifford_lift import symbolic_unitary
from hybrid_qcs.native_domain import NativeProblem, archive_key, certify_native, phase_target
from hybrid_qcs.native_policy import NativeHierarchy
from hybrid_qcs.native_search import NativeSearch, optimize_native
from hybrid_qcs.native_benchmarks import (named_benchmarks, prior_phase_benchmarks,
                                         regression_problems, contract, reference_control)
from hybrid_qcs.native_runner import train_native
from hybrid_qcs.native_audit import audit_native, verify_native_cover
from hybrid_qcs.resource_search import WorkLimits


@pytest.fixture(scope='session')
def native_models(tmp_path_factory):
    path=tmp_path_factory.mktemp('native-trained-models');models={}
    for seed in (0,1,2):
        model,log=train_native(seed)
        assert model.episodes==48 and model.outer_updates>0
        assert np.all(model.updates>0),log['family_updates']
        model.save(path/f'{seed}.json');loaded=NativeHierarchy.load(path/f'{seed}.json')
        assert loaded.frozen and loaded.digest==model.digest
        models[seed]=loaded
    return models


@pytest.fixture(scope='session')
def native_regressions():return regression_problems(32)


@pytest.mark.generated
@pytest.mark.parametrize('case,seed',product(range(32),range(3)))
def test_posttraining_native_mixed_circuit(case,seed,native_models,native_regressions):
    p=native_regressions[case];m=native_models[seed];before=m.digest
    engine=NativeSearch(p,WorkLimits(8192,20000,15,15))
    r=engine.run(m)
    assert r['status']=='feasible',(p.name,seed,r['reason'],r['edges'])
    assert r['witness']['success'] and r['witness']['exact_symbolic_replay']
    assert r['witness']['native'] and not r['audit_witness_used']
    assert m.digest==before
    assert all(type(v.state) is HybridState for v in engine.records.values())
    for v in engine.records.values():
        assert hasattr(v.state,'tableau') and hasattr(v.state,'rotations')
        assert hasattr(v.state,'tail') and hasattr(v.state,'global_phase_eighths')
    assert r['profile']['phase_obligation_transitions']==0


@pytest.mark.generated
@pytest.mark.parametrize('n,a,seed',product((1,2),(0,1,2),range(3)))
def test_posttraining_entangling_and_clean_registers(n,a,seed,native_models):
    gates=(Gate('H',(0,)),Gate('T',(0,))) if n==1 else (Gate('H',(0,)),Gate('CNOT',(0,1)))
    p=NativeProblem('coherent-generation',contract(n,a),Budget(1,1,3,3),unitary_from_gates(n,gates))
    r=NativeSearch(p,WorkLimits(8192,20000,20,20)).run(native_models[seed])
    assert r['status']=='feasible',(n,a,seed,r['reason'])
    assert r['witness']['workspace_leakage']<1e-18
    assert r['witness']['isometry_error']<1e-9


@pytest.mark.generated
@pytest.mark.parametrize('n,seed',product((2,3,4),range(3)))
def test_posttraining_swap(n,seed,native_models):
    p=next(p for p in named_benchmarks() if p.name==f'swap-{n}-q0-q{n-1}')
    r=NativeSearch(p,WorkLimits(12000,24000,30,30)).run(native_models[seed])
    assert r['status']=='feasible',(n,seed,r['reason'],r['edges'])
    assert r['witness']['resources']['cnot']==3


@pytest.mark.generated
@pytest.mark.parametrize('seed',range(3))
def test_posttraining_global_minus_identity(seed,native_models):
    p=NativeProblem('-I',contract(1),Budget(0,0,12,12),-np.eye(2))
    r=NativeSearch(p,WorkLimits(4096,10000,15,15)).run(native_models[seed])
    assert r['status']=='feasible'
    assert r['witness']['resources']['gates']>0
    assert r['witness']['global_phase_eighths']==8


@pytest.mark.parametrize('case',range(18))
def test_all_named_benchmarks_retain_native_frontiers(case,native_models):
    p=named_benchmarks()[case]
    engine=NativeSearch(p,WorkLimits(8,1000,10,10))
    r=engine.run(native_models[0])
    assert r['edges']<=8
    assert all(type(v.state) is HybridState for v in engine.records.values())
    assert r['status'] in ('feasible','unknown')
    assert not r['audit_witness_used']
    if r['status']=='feasible':assert r['witness']['success']
    else:assert r['optimality']=='not_established'


@pytest.mark.parametrize('case',range(18))
def test_named_reference_control_is_not_rl_discovery(case):
    r=reference_control(named_benchmarks()[case])
    if r['status']=='certified':
        assert r['certificate']['success']
        assert r['certificate']['source']=='constructive_reference_not_RL'
    else:assert named_benchmarks()[case].name in ('qft-3-clean-0','toffoli-4-clean-0')


def test_preserves_all_25_phase_specifications_and_caps():
    from pathlib import Path
    rows=json.loads((Path(__file__).resolve().parents[1]/'experiments/publication_v1/protocol.json').read_text())['split']['test']
    targets=prior_phase_benchmarks();assert len(targets)==25
    for p,row in zip(targets,rows):
        spec=row['problem'];c=spec['coefficients']
        exponents=[(spec['constant']+sum(k*((m&x).bit_count()%2) for m,k in c))%8 for x in range(1<<spec['n'])]
        assert np.allclose(np.diag(p.unitary),np.exp(1j*np.pi*np.array(exponents)/4))
        assert p.budget.max_cnot_count==spec['max_cnot']
        assert p.budget.max_gates==spec['max_gates'] and p.budget.max_depth==spec['max_depth']
        assert p.max_t_depth==spec['max_t_depth'] and len(p.contract.clean_ancillas)==spec['ancillas']
        assert any(g.name=='H' for g in p.actions)
        assert not hasattr(p,'remaining') and not hasattr(p,'phase_bits')


@pytest.mark.parametrize('n',range(1,5))
def test_exact_hybrid_factorization_not_just_projective(n):
    rng=random.Random(119+n)
    for _ in range(10):
        s=HybridState.identity(n,Budget(12,12,12,12));gates=[]
        for _ in range(12):
            g=rng.choice(generate_gates(n));gates.append(g)
            s=s.apply(g,partial_order_reduction=False)
        assert np.max(np.abs(symbolic_unitary(s)-unitary_from_gates(n,gates)))<1e-10
        s.validate()


def test_phase_lift_separates_equal_tableaux_and_corruption():
    root=HybridState.identity(1,Budget(0,0,12,12));s=root
    for name in ('H','S','S','H','S','S')*2:s=s.apply(Gate(name,(0,)),partial_order_reduction=False)
    assert s.tableau==root.tableau and s.canonical_key==root.canonical_key
    assert s.global_phase_eighths==8 and root.global_phase_eighths==0
    p=NativeProblem('I',contract(1),root.budget,np.eye(2))
    assert archive_key(p,s)!=archive_key(p,root)
    assert archive_key(replace(p,contract=contract(1,mode=PhaseMode.PROJECTIVE)),s)==archive_key(replace(p,contract=contract(1,mode=PhaseMode.PROJECTIVE)),root)
    with pytest.raises(AssertionError):replace(s,global_phase_eighths=0).validate()
    assert not certify_native(p,s)['success']
    assert certify_native(replace(p,contract=contract(1,mode=PhaseMode.PROJECTIVE)),s)['success']


def test_ordered_noncommuting_rotations_and_persistent_dag():
    root=HybridState.identity(1,Budget(4,0,8,8));s=root
    for name in ('T','H','T'):
        prev=s;s=s.apply(Gate(name,(0,)),partial_order_reduction=False)
        assert s.tail.previous is prev.tail
    assert s.anticommuting_pairs==1
    flipped=replace(s,rotations=tuple(reversed(s.rotations)))
    assert np.max(np.abs(symbolic_unitary(s)-symbolic_unitary(flipped)))>.1
    with pytest.raises(AssertionError):flipped.validate()


def test_native_pending_actions_do_not_use_a_phase_obligation_mask():
    p=phase_target('T0',2,((1,1),),budget=Budget(4,3,6,6))
    e=NativeSearch(p,panel_size=1);root=e.records[0]
    assert root.pending.bit_count()==12
    t=e.p.actions.index(Gate('H',(0,)));child=e.step(0,t)
    assert child is not None and child.state.tableau!=root.state.tableau
    assert root.pending.bit_count()==11 and len(e.frontier)>e.panel_size


def test_no_phase_solver_called(monkeypatch,native_models):
    import hybrid_qcs.phase_search as old
    def fail(*args,**kwargs):raise AssertionError('specialized phase engine was invoked')
    monkeypatch.setattr(old,'search_phase',fail)
    p=phase_target('T',1,((1,1),),budget=Budget(1,0,2,2))
    assert NativeSearch(p).run(native_models[0])['status']=='feasible'


def test_ancilla_restoration_is_coherent():
    p=NativeProblem('logical-I',contract(1,1),Budget(0,0,2,2),np.eye(2))
    s=HybridState.identity(2,p.budget).apply(Gate('H',(1,)),partial_order_reduction=False)
    r=certify_native(p,s);assert not r['success'] and r['workspace_leakage']>.49


def test_old_checkpoint_cannot_be_loaded_as_native(tmp_path):
    path=tmp_path/'old.json';path.write_text(json.dumps({'schema':'phase-hierarchy-v1'}))
    with pytest.raises(ValueError):NativeHierarchy.load(path)


@pytest.mark.parametrize('which',('edge','record','cancel','wall'))
def test_interruption_is_not_infeasibility(which,native_models):
    p=NativeProblem('H',contract(1),Budget(1,0,3,3),unitary_from_gates(1,[Gate('H',(0,))]))
    limits=WorkLimits(0 if which=='edge' else 100,1 if which=='record' else 1000,0 if which=='wall' else 10,10)
    r=NativeSearch(p,limits,cancel=(lambda:True) if which=='cancel' else None).run(native_models[0])
    assert r['status']=='unknown' and r['optimality']=='not_established'


def test_terminal_shaping_and_sarsa_bootstrap():
    p=NativeProblem('hard',contract(1),Budget(2,0,4,4),unitary_from_gates(1,[Gate('H',(0,)),Gate('T',(0,)),Gate('H',(0,))]))
    m=NativeHierarchy();r=NativeSearch(p,WorkLimits(1,100,10,10)).run(m,train='outer')
    assert r['training_transitions'][-1]['terminal']
    assert r['training_transitions'][-1]['potential_after']==0
    assert m.outer_updates==1


def test_native_cover_checks_schema_closure_and_digest():
    p=NativeProblem('H',contract(1),Budget(0,0,0,0),unitary_from_gates(1,[Gate('H',(0,))]))
    r=audit_native(p);assert r['status']=='infeasible_under_numerical_contract'
    assert verify_native_cover(p,r['certificate'])['valid']
    invalid=dict(r['certificate'],schema='phase-cover-v1')
    assert not verify_native_cover(p,invalid)['valid']
    invalid=dict(r['certificate'],labels=[])
    assert not verify_native_cover(p,invalid)['valid']
    assert not verify_native_cover(p,r['certificate'],limits=WorkLimits(0,100,10,10))['valid']


def test_native_bound_tightening_zero_certificate(native_models):
    p=NativeProblem('H',contract(1),Budget(0,0,1,1),unitary_from_gates(1,[Gate('H',(0,))]))
    r=optimize_native(p,native_models[0],objective='cnot')
    assert r['status']=='optimal_nonnegative_resource_bound' and r['witness']['success']
