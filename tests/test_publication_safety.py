"""Necessary supporting tests for the proof and experiment trust boundary."""
from copy import deepcopy
from dataclasses import replace
import json
import random
import numpy as np
import pytest
from hybrid_qcs.phase_contract import (PhaseProblem,root_state,successor,certify_phase,
    problem_from_manifest,boolean_phase_coefficients,completion_possible)
from hybrid_qcs.phase_policy import PhaseHierarchy,features
from hybrid_qcs.phase_index import CandidateIndex
from hybrid_qcs.phase_search import search_phase
from hybrid_qcs.phase_audit import audit_phase,verify_phase_cover,reference_root,reference_step,_within
from hybrid_qcs.phase_baselines import rank_tdepth_certificate,verify_rank_bound,uniform_cost_phase,rank_partition_baseline
from hybrid_qcs.phase_graysynth import graysynth_baseline
from hybrid_qcs.publication_pipeline import anytime_discovery
from hybrid_qcs.publication_corpus import corpus,key,bnn_truth,bnn_problem,regression_cases
from hybrid_qcs.resource_search import WorkLimits
from hybrid_qcs.resource_domain import canonical_digest


def test_all_target_orbits_are_separated():
    train,val,test=corpus()
    sets=[{key(p) for p in split} for split in (train,val,test,regression_cases())]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i+1,4))
    assert len(sets[2])==25 and len(sets[3])==60


def test_phase_specification_matches_all_boolean_truth_tables_three_bits():
    # The converter, not any reference circuit, defines Boolean phase tasks.
    for bits in range(256):
        truth=tuple((bits>>x)&1 for x in range(8))
        n,coef,constant=boolean_phase_coefficients(truth)
        p=PhaseProblem(n,coef,constant=constant)
        assert p.target_exponents()==tuple(4*v for v in truth)


def test_exact_reference_steps_match_fast_transitions_without_sharing_code():
    rng=random.Random(177)
    for p in corpus()[0]:
        s,_=root_state(p)
        assert s==reference_root(p)
        for _ in range(80):
            valid=[i for i,(c,t) in enumerate(p.actions) if c!=t or s.remaining&p.phase_bits.get(s.rows[t],0)]
            token=rng.choice(valid)
            s2,_=successor(p,s,token)
            assert reference_step(p,s,token)==s2
            assert _within(p,s2)==completion_possible(p,s2)
            s=s2


def test_indexed_panel_matches_sort_under_insert_and_delete():
    rng=random.Random(5);index=CandidateIndex();present={}
    for rid in range(1000):
        if not present or rng.random()<.6:
            value=rng.random();present[rid]=value;index.add(value,rid)
        else:
            old=rng.choice(list(present));del present[old];index.discard(old)
        assert index.smallest(32)==tuple(sorted(present,key=lambda i:(present[i],i))[:32])
        assert len(index)==len(present)


def test_independent_closed_cover_excludes_tighter_cnot_cap():
    p=PhaseProblem(2,((1,1),(2,1),(3,7)),max_cnot=1,max_t_depth=2)
    r=audit_phase(p)
    assert r['status']=='infeasible' and r['witness'] is None
    assert verify_phase_cover(r['proof'],p)['valid']


@pytest.mark.parametrize('tamper',('hash','root','wrong_contract','target','missing_successor'))
def test_corrupt_exclusion_certificate_is_not_a_proof(tamper):
    p=PhaseProblem(2,((1,1),(2,1),(3,7)),max_cnot=1)
    proof=deepcopy(audit_phase(p)['proof'])
    if tamper=='hash':
        proof['digest']='invalid'
    elif tamper=='root':
        proof['cover']=[]
    elif tamper=='target':
        proof['cover'][0]['remaining']=0
    elif tamper=='missing_successor':
        from dataclasses import asdict
        proof['cover']=[asdict(reference_root(p))]
    if tamper!='hash':
        proof['digest']=canonical_digest({k:v for k,v in proof.items() if k!='digest'})
    expected=replace(p,max_cnot=2) if tamper=='wrong_contract' else p
    assert not verify_phase_cover(proof,expected)['valid']


def test_timeout_and_cancellation_never_export_infeasibility():
    p=bnn_problem()
    r=audit_phase(p,limits=WorkLimits(0,100,1,1))
    assert r['status']=='unknown' and r['proof'] is None
    r=audit_phase(p,cancel=lambda:True)
    assert r['status']=='unknown' and r['proof'] is None


def test_untrained_evaluation_requires_explicit_baseline_label():
    with pytest.raises(ValueError):
        search_phase(bnn_problem(),PhaseHierarchy())


def test_frozen_model_rejects_updates_and_roundtrips(tmp_path):
    m=PhaseHierarchy().freeze()
    with pytest.raises(RuntimeError):m.update_outer(np.zeros(20),0,None)
    with pytest.raises(RuntimeError):m.update_inner(np.zeros(20),'data',0)
    m.save(tmp_path/'model.json')
    assert PhaseHierarchy.load(tmp_path/'model.json').digest==m.digest


def test_ablation_masks_apply_to_training_as_well_as_scores():
    m=PhaseHierarchy();m.ablation='no_budget';before=m.outer.copy()
    x=np.ones(20);m.update_outer(x,1,None)
    assert np.array_equal(m.outer[[5,6,7,9,10,13,14,16,17,19]],before[[5,6,7,9,10,13,14,16,17,19]])
    m.update_inner(x,'data',1)
    assert np.all(m.responses['data'][[7,8,9,12,13,14,16,17,19]]==0)


def test_manifest_rejects_changed_native_cost_model():
    d=bnn_problem().manifest();d['lowering']['1']=['S']
    with pytest.raises(ValueError):problem_from_manifest(d)


def test_boolean_converter_rejects_degree_four():
    with pytest.raises(ValueError):boolean_phase_coefficients(tuple(int(x==15) for x in range(16)))


def test_rank_certificate_matches_native_ancilla_tradeoff():
    for a,minimum in ((0,2),(1,1),(2,1)):
        p=PhaseProblem(2,((1,1),(2,1),(3,7)),a,max_t_depth=minimum)
        proof=rank_tdepth_certificate(p)
        assert verify_rank_bound(proof,p) and proof['lower_t_depth']==minimum
        cert=rank_partition_baseline(p)
        assert cert['success'] and cert['resources']['t_depth']==minimum


def test_rank_certificate_cannot_be_promoted_to_larger_lower_bound():
    p=bnn_problem();proof=rank_tdepth_certificate(p);proof['lower_t_depth']+=1
    proof['digest']=canonical_digest({k:v for k,v in proof.items() if k!='digest'})
    assert not verify_rank_bound(proof,p)


def test_eager_phase_restriction_cannot_claim_deferred_infeasibility():
    p=PhaseProblem(2,((1,1),(2,1),(3,7)),1,max_cnot=4,max_t_depth=1,emission='eager')
    r=audit_phase(p)
    assert r['proof'] is not None and verify_phase_cover(r['proof'],p)['valid']
    assert not verify_phase_cover(r['proof'],replace(p,emission='deferred'))['valid']


def test_uniform_cost_accepts_a_settled_goal_and_does_not_use_a_policy():
    p=PhaseProblem(2,((3,1),),max_cnot=4)
    r=uniform_cost_phase(p)
    assert r['reason']=='settled_goal' and r['witness']['resources']['cnot']==2
    assert r['witness_source']=='uniform_cost' and not r['audit_witness_used']


def test_discovery_module_does_not_call_auditor(monkeypatch):
    import hybrid_qcs.phase_audit as audit
    def fail(*args,**kwargs):raise AssertionError('auditor entered discovery')
    monkeypatch.setattr(audit,'audit_phase',fail)
    p=PhaseProblem(2,((3,1),))
    r=anytime_discovery(p,PhaseHierarchy().freeze(),scheduler='untrained',limits=WorkLimits(200,1000,1,1))
    assert r['best'] and r['audit_calls']==0 and not r['audit_witness_used']


def test_graysynth_control_is_exact_and_independently_identified():
    for p in corpus()[1]:
        cert=graysynth_baseline(p)
        assert cert['success'] and cert['dag_validated']
        assert cert['source'].startswith('GraySynth')


def test_global_phase_is_physically_counted_not_dropped():
    p=PhaseProblem(1,(),constant=4)
    cert=certify_phase(p,())
    assert cert['success'] and cert['resources']['gates']==12
    assert cert['exact_basis_phase_match']


def test_global_fairness_preserves_small_problem_under_adversarial_scores():
    p=PhaseProblem(2,((3,1),),max_cnot=2)
    m=PhaseHierarchy();m.outer[:]=100;m.prior[:]=-100;m.freeze()
    r=search_phase(p,m,panel_size=1,fairness=1,limits=WorkLimits(1000,2000,2,2))
    assert r['status']=='feasible' and r['profile']['fairness_steps']>0


def test_selected_zero_residual_is_labelled_untrained_not_learned_success():
    from pathlib import Path
    from hybrid_qcs.publication_guard import shrink
    path = Path('experiments/publication_v1/checkpoints/full-seed-0.json')
    model = PhaseHierarchy.load(path)
    candidate = shrink(model, 0.)
    prior = PhaseHierarchy().freeze()
    assert np.array_equal(candidate.outer, prior.outer)
    assert all(np.all(candidate.responses[arm] == 0) for arm in candidate.responses)
    before = model.digest
    shrink(model, .1)
    assert model.digest == before


def test_amendment_confirmatory_targets_exclude_every_original_split():
    from hybrid_qcs.publication_guard import confirmation_targets
    original = {key(p) for group in corpus() for p in group} | {key(p) for p in regression_cases()}
    new = {key(p) for p in confirmation_targets()}
    assert len(new) == 12 and not new & original
