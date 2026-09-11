"""Native publication qualification: generate AFTER a complete frozen training run.

The fixed 24-episode regression training schedule is deliberately separate from
184-episode performance training. These tests are not success-rate estimates.
The publication evidence verifier checks the actual performance checkpoints and
all saved successful performance witnesses after their campaign.
"""
from __future__ import annotations
from dataclasses import replace
from itertools import product
import json
import numpy as np
import pytest
from hybrid_qcs.model import Gate,Budget,HybridState,generate_gates
from hybrid_qcs.native_exact import (ExactMatrix,exact_word,exact_qft,exact_mcx,orbit_digest,
    verify_exact_word,determinant_obstruction,verify_determinant_certificate,verify_exact_cover,verify_exact_optimization)
from hybrid_qcs.native_study_corpus import make_corpus,load_problem,dump,lock
from hybrid_qcs.native_study_training import train_model
from hybrid_qcs.native_study_search import StudyPolicy,StudySearch,run_anytime,uniform_cost,meet_in_middle
from hybrid_qcs.native_study_analysis import paired_interval
from hybrid_qcs.native_benchmarks import contract,named_benchmarks,reference_gates
from hybrid_qcs.native_domain import NativeProblem,certify_native,digest
from hybrid_qcs.native_audit import audit_native
from hybrid_qcs.native_optimize import optimize_native_resources,OPTIMAL
from hybrid_qcs.native_study_runner import run_reachability
from hybrid_qcs.resource_search import WorkLimits
from hybrid_qcs.certify import unitary_from_gates


@pytest.fixture(scope='session')
def study_protocol(tmp_path_factory):
    return lock(tmp_path_factory.mktemp('native-pub-protocol'))


@pytest.fixture(scope='session')
def study_models(study_protocol,tmp_path_factory):
    out=tmp_path_factory.mktemp('native-pub-trained');models={}
    for seed in (11,23,47):
        model,log=train_model(study_protocol,seed,.01,stages=(8,12,4),output=out)
        assert model.frozen and model.episodes==24 and all(model.updates>0)
        assert log['audit_calls']==0 and not log['references_supplied']
        loaded=StudyPolicy.load(out/'checkpoints/full'/f'seed-{seed}.json')
        assert loaded.digest==model.digest
        models[seed]=loaded
    return models


@pytest.mark.generated
@pytest.mark.parametrize('case,seed',list(product(range(12),(11,23,47))))
def test_trained_native_exact_mixed_generation(case,seed,study_protocol,study_models):
    row=[r for r in study_protocol['splits']['regression'] if r['n']==1][case]
    p=load_problem(row);model=study_models[seed];before=model.digest
    engine=StudySearch(p,WorkLimits(30000,60000,30.,30.))
    result=engine.run(model)
    assert result['witness'] is not None,(row['name'],seed,result['reason'])
    assert verify_exact_word(p.contract,ExactMatrix.from_payload(row['exact_target']),result['witness']['native'])['valid']
    assert model.digest==before and all(isinstance(r.state,HybridState) for r in engine.records.values())
    assert result['profile']['phase_obligation_transitions']==0


@pytest.mark.generated
@pytest.mark.parametrize('n,seed,phase',list(product((2,3),(11,23,47),('T','TDG'))))
def test_trained_native_entangling_generation(n,seed,phase,study_models):
    word=(Gate('H',(0,)),Gate('CNOT',(0,1)),Gate(phase,(0,)))
    exact=exact_word(n,word)
    p=NativeProblem('mixed-entangling-unit-regression',contract(n),Budget(1,1,3,3),exact.numerical())
    r=StudySearch(p,WorkLimits(30000,60000,30.,30.)).run(study_models[seed])
    assert r['witness'] is not None,(n,seed,phase)
    assert verify_exact_word(p.contract,exact,r['witness']['native'])['valid']


@pytest.mark.generated
@pytest.mark.parametrize('ancilla,seed,phase',list(product((0,1,2),(11,23,47),('T','TDG'))))
def test_trained_native_clean_isometry_generation(ancilla,seed,phase,study_models):
    exact=exact_word(1,[Gate('H',(0,)),Gate(phase,(0,))])
    p=NativeProblem('clean-generation',contract(1,ancilla),Budget(1,1,3,3),exact.numerical())
    r=StudySearch(p,WorkLimits(20000,60000,30.,30.)).run(study_models[seed])
    assert r['witness'] is not None
    assert verify_exact_word(p.contract,exact,r['witness']['native'])['valid']
    assert r['witness']['workspace_leakage']<1e-20


@pytest.mark.generated
@pytest.mark.parametrize('seed,word',list(product((11,23,47),(('H','T'),('T','H'),('H','T','H')))))
def test_trained_generation_and_exact_optimality(seed,word,study_models):
    exact=exact_word(1,[Gate(g,(0,)) for g in word])
    p=NativeProblem('exact-opt-regression',contract(1),Budget(1,0,len(word),len(word)),exact.numerical())
    r=optimize_native_resources(p,study_models[seed],objectives=('gates','t_count','depth'),
       limits=WorkLimits(30000,60000,30.,30.),discovery_edges=2048,allow_audit_witness=False)
    assert r['status'] in OPTIMAL and not r['audit_witness_used']
    checked=verify_exact_optimization(p,exact,r)
    assert checked['valid'] and checked['exact_optimality_verified']


@pytest.mark.parametrize('n,seed',list(product((1,2,3),(1,17,42))))
def test_cyclotomic_reference_differential(n,seed):
    rng=np.random.default_rng(seed);gs=generate_gates(n)
    word=[gs[int(rng.integers(len(gs)))] for _ in range(30)]
    exact=exact_word(n,word)
    assert np.max(np.abs(exact.numerical()-unitary_from_gates(n,word)))<1e-12
    assert ExactMatrix.from_payload(json.loads(json.dumps(exact.payload())))==exact
    assert np.max(np.abs(exact.dagger().numerical()-exact.numerical().conj().T))<1e-12


@pytest.mark.parametrize('kind,n,excluded', [('QFT',2,False),('QFT',3,True),('Toffoli',3,False),('Toffoli',4,True)])
def test_exact_determinant_scope(kind,n,excluded):
    proof=determinant_obstruction(kind,n)
    assert proof['excluded']==excluded
    assert verify_determinant_certificate(proof)==excluded
    mutated=dict(proof,clean_ancillas=1)
    assert not verify_determinant_certificate(mutated)
    mutated=dict(proof,phase_mode='projective')
    assert not verify_determinant_certificate(mutated)


def test_exact_one_ancilla_witnesses(tmp_path):
    rows=run_reachability(tmp_path)
    assert len(rows)==2
    assert all(r['minimum_clean_ancillas']==1 and not r['discovered_by_learning'] for r in rows)


def test_exact_orbit_disjointness(study_protocol):
    seen=set()
    for split,rows in study_protocol['splits'].items():
        for row in rows:
            exact=ExactMatrix.from_payload(row['exact_target'])
            assert orbit_digest(exact)==row['orbit']
            assert row['orbit'] not in seen
            seen.add(row['orbit'])
    assert len(study_protocol['retained'])==43


@pytest.mark.parametrize('method', ['uniform_cost','mitm'])
def test_reference_free_baselines(method):
    exact=exact_word(2,[Gate('H',(0,)),Gate('CNOT',(0,1)),Gate('T',(0,))])
    p=NativeProblem('baseline',contract(2),Budget(1,1,3,3),exact.numerical())
    solver=uniform_cost if method=='uniform_cost' else meet_in_middle
    r=solver(p,WorkLimits(10000,30000,10.,10.))
    assert r['witness'] and verify_exact_word(p.contract,exact,r['witness']['native'])['valid']


def test_no_late_success(study_models):
    exact=exact_word(1,[Gate('H',(0,)),Gate('T',(0,)),Gate('H',(0,))])
    p=NativeProblem('zero-budget',contract(1),Budget(1,0,3,3),exact.numerical())
    r=run_anytime(p,study_models[11],'hierarchy',axis='wall_seconds',budget=0.)
    assert not r['timely_success'] and r['witness'] is None


def test_ablation_masks_and_checkpoint(tmp_path,study_protocol):
    model,log=train_model(study_protocol,77,.01,variant='no_structure',stages=(2,4,2))
    x=np.arange(20,dtype=float);y=x.copy();y[6:10]+=1000
    assert model.score_outer(x)==model.score_outer(y)
    model.save(tmp_path/'a.json');copy=StudyPolicy.load(tmp_path/'a.json')
    assert copy.digest==model.digest


def test_exact_cover_rejects_target_and_digest():
    exact=exact_word(1,[Gate('T',(0,))]);p=NativeProblem('excluded',contract(1),Budget(0,0,2,2),exact.numerical())
    audited=audit_native(p,limits=WorkLimits(1000,1000,10.,10.))
    cert=audited['certificate'];assert cert
    assert verify_exact_cover(p,exact,cert)['valid']
    changed=dict(cert,digest='0'*64)
    assert not verify_exact_cover(p,exact,changed)['valid']
    assert not verify_exact_cover(p,ExactMatrix.identity(1),cert)['valid']


def test_pairing_does_not_count_repetitions_as_targets():
    rows=[]
    for target in range(4):
        for _ in range(30):
            rows += [{'name':str(target),'method':'h','success':1.}, {'name':str(target),'method':'u','success':0.}]
    p=paired_interval(rows,'h','u',resamples=200)
    assert p['clusters']==4 and p['mean']==1. and p['lower_95']==1.
