"""Most new tests generate, optimize, and verify circuits AFTER native training."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json

import numpy as np
import pytest

from hybrid_qcs.model import Budget, Gate, HybridState
from hybrid_qcs.certify import unitary_from_gates
from hybrid_qcs.native_benchmarks import contract, named_benchmarks, prior_phase_benchmarks
from hybrid_qcs.native_domain import NativeProblem, digest
from hybrid_qcs.native_runner import train_native
from hybrid_qcs.native_policy import NativeHierarchy
from hybrid_qcs.native_search import NativeSearch, optimize_native
from hybrid_qcs.native_audit import audit_native, verify_native_cover
from hybrid_qcs.native_optimize import (optimize_native_resources, verify_native_optimization,
                                       optimize_native_ancillas, OPTIMAL)
from hybrid_qcs.native_optimality_runner import calibration_problems, problem_from_manifest
from hybrid_qcs.resource_search import WorkLimits

LIMITS = WorkLimits(25000, 50000, 15., 15.)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('optimality-trained')
    models = []
    for seed in range(3):
        model, log = train_native(seed)
        assert model.episodes == 48 and model.outer_updates > 0
        assert all(log['family_updates'].values())
        model.save(root/f'{seed}.json')
        loaded = NativeHierarchy.load(root/f'{seed}.json')
        assert loaded.frozen and loaded.digest == model.digest
        models.append(loaded)
    return models


@pytest.fixture(scope='module')
def specs():
    base = calibration_problems()
    out = list(base)
    for word in [('S',), ('TDG',), ('T','H')]:
        out.append(NativeProblem(''.join(word), contract(1), Budget(1,0,3,3),
                   unitary_from_gates(1,[Gate(g,(0,)) for g in word])))
    out.append(NativeProblem('CNOT',contract(2),Budget(0,1,1,1),
               unitary_from_gates(2,[Gate('CNOT',(0,1))])))
    out.append(replace(base[1],name='T-clean',contract=contract(1,1),budget=Budget(1,0,1,1)))
    return out


@pytest.mark.generated
@pytest.mark.parametrize('case,seed,order',product(range(10),range(3),[('gates',),('t_count','cnot','depth','t_depth','gates')]))
def test_posttraining_generated_optima_and_standalone_proofs(case,seed,order,trained,specs):
    p, model = specs[case], trained[seed]
    before = model.digest
    result = optimize_native_resources(p,model,objectives=order,limits=LIMITS)
    assert result['status'] in OPTIMAL,(p.name,seed,result['reason'])
    assert result['witness']['success'] and result['discovery_incumbent'] is not None
    assert not result['audit_witness_used']
    assert model.digest == before
    # The archived JSON, not only the in-memory tuple representation, must verify.
    archived = json.loads(json.dumps(result))
    verified = verify_native_optimization(p,archived)
    assert verified['valid'] and verified['optimality_verified'], verified
    assert verified['proved_objectives'] == list(order)
    assert all(s['proved'] and s['lower_bound']==s['upper_bound'] for s in result['stages'])
    if p.name=='opt-cal-SWAP':
        assert result['witness']['resources']['cnot']==3
        assert result['witness']['resources']['gates']==3
    if p.name=='opt-cal-HTH':
        assert result['witness']['resources']['t_count']==1
        assert result['witness']['resources']['gates']==3
    work=result['work']
    assert work['edges']==sum(work['phase_edges'].values())<=LIMITS.max_edges
    assert work['records']<=LIMITS.max_records
    assert all(r['discovery']['profile']['phase_obligation_transitions']==0 for r in result['journal'])
    assert all(r['discovery']['policy_frozen'] for r in result['journal'])


@pytest.fixture(scope='module')
def proven_t(trained):
    p=calibration_problems()[1]
    return p,optimize_native_resources(p,trained[0],objectives=('t_count',),limits=LIMITS)


@pytest.mark.parametrize('damage',('cost','target','cap','missing_stage','zero_proof','phase_schema','label','trial','phase','word'))
def test_tampered_optimality_receipt_rejected(damage,proven_t):
    p,original=proven_t
    r=deepcopy(original)
    if damage=='cost':r['witness']['resources']['t_count']=0
    elif damage=='target':p=replace(p,unitary=np.eye(2))
    elif damage=='cap':p=p.cap('gates',4)
    elif damage=='missing_stage':r['stages']=[]
    elif damage=='zero_proof':r['stages'][0]['proof']={'kind':'nonnegative_integer_resource'}
    elif damage=='phase_schema':r['stages'][0]['proof']['certificate']['schema']='phase-closed-cover-v1'
    elif damage=='label':r['stages'][0]['proof']['certificate']['labels']=[]
    elif damage=='trial':r['stages'][0]['proof']['trial_digest']=p.digest
    elif damage=='phase':
        from hybrid_qcs.ancilla_contract import PhaseMode
        p=replace(p,contract=replace(p.contract,phase_mode=PhaseMode.PROJECTIVE))
    elif damage=='word':r['witness']['native']=[['H',[0]]]
    assert not verify_native_optimization(p,r)['valid']


@pytest.mark.parametrize('which',('edge','record','wall','cpu','cancel'))
def test_interrupted_optimizer_never_proves_infeasibility(which,trained):
    p=calibration_problems()[2]
    kw={}
    limits=WorkLimits(0 if which=='edge' else 200,1 if which=='record' else 500,
                      0 if which=='wall' else 10,0 if which=='cpu' else 10)
    if which=='cancel':kw['cancel']=lambda:True
    r=optimize_native_resources(p,trained[0],limits=limits,**kw)
    assert r['status']=='unknown' and not r['witness']
    assert all(not s['proved'] for s in r['stages'])
    assert r['work']['edges']<=limits.max_edges


@pytest.mark.parametrize('which',('audit_disabled','verification_limit','round_limit'))
def test_incumbent_without_completed_proof_remains_upper_bound(which,trained):
    p=calibration_problems()[1]
    kw={'audit':False} if which=='audit_disabled' else {'verification_edges':1} if which=='verification_limit' else {'max_rounds':1}
    r=optimize_native_resources(p,trained[0],objectives=('t_count',),limits=LIMITS,**kw)
    assert r['status']=='upper_bound' and r['witness']['success']
    checked=verify_native_optimization(p,r)
    assert checked['valid'] and not checked['optimality_verified']


@pytest.mark.parametrize('adopt',(False,True))
def test_audit_generated_circuits_never_count_as_rl(adopt,trained):
    p=calibration_problems()[2]
    r=optimize_native_resources(p,trained[0],objectives=('gates',),limits=LIMITS,
                               discovery_edges=1,allow_audit_witness=adopt)
    assert r['discovery_incumbent'] is None
    if adopt:
        assert r['status'] in OPTIMAL and r['audit_witness_used']
        assert r['witness']['source']=='deterministic_native_audit'
        assert verify_native_optimization(p,r)['optimality_verified']
    else:
        assert r['status']=='unknown' and not r['witness'] and not r['audit_witness_used']
        assert r['reason']=='audit_found_witness_not_adopted'


def test_actual_infeasibility_requires_verified_native_cover(trained):
    p=calibration_problems()[0].cap('gates',0)
    r=optimize_native_resources(p,trained[0],limits=LIMITS)
    assert r['status']=='infeasible_under_numerical_contract'
    assert r['journal'][0]['verification']['valid']
    assert verify_native_optimization(p,json.loads(json.dumps(r)))['infeasibility_verified']


def test_native_verifier_cancellation_and_budget(proven_t):
    p,r=proven_t
    assert not verify_native_optimization(p,r,cancel=lambda:True)['valid']
    assert not verify_native_optimization(p,r,limits=WorkLimits(1,100,10,10))['valid']
    cert=r['stages'][0]['proof']['certificate'];trial=p.cap('t_count',0)
    assert not verify_native_cover(trial,cert,cancel=lambda:True)['valid']
    assert audit_native(trial,cancel=lambda:True)['status']=='unknown'


def test_checker_does_not_call_discovery_or_audit(monkeypatch,proven_t):
    p,r=proven_t
    def reject(*a,**k):raise AssertionError('independent verification called search')
    monkeypatch.setattr(NativeSearch,'run',reject)
    import hybrid_qcs.native_optimize as opt
    monkeypatch.setattr(opt,'audit_native',reject)
    assert verify_native_optimization(p,r)['optimality_verified']


@pytest.mark.generated
@pytest.mark.parametrize('seed',range(3))
def test_posttraining_clean_width_sweep_without_cross_width_merges(seed,trained):
    p=calibration_problems()[0]
    r=optimize_native_ancillas(p,trained[seed],ancilla_budgets=(0,1),objectives=('gates',),limits=LIMITS)
    assert r['minimum_proved'] and r['minimum_clean_ancillas']==0
    assert all(row['result']['status'] in OPTIMAL for row in r['results'])
    assert r['results'][0]['result']['problem_digest']!=r['results'][1]['result']['problem_digest']
    # Omitting width zero cannot prove that one clean ancilla is necessary.
    other=optimize_native_ancillas(p,trained[seed],ancilla_budgets=(1,),objectives=('gates',),limits=LIMITS)
    assert not other['minimum_proved'] and other['minimum_clean_ancillas'] is None


def test_manifest_roundtrip_and_no_altered_grammar(specs):
    for p in specs:
        doc=json.loads(json.dumps(p.manifest()))
        assert problem_from_manifest(doc).digest==p.digest
        doc['grammar']=[]
        with pytest.raises(ValueError):problem_from_manifest(doc)


def test_benchmark_registry_unchanged():
    assert len(named_benchmarks())==18 and len(prior_phase_benchmarks())==25
    assert {p.family for p in named_benchmarks()}=={'QFT','Toffoli','SWAP','ancilla-mixed-Pauli'}


def test_invalid_objectives_do_not_run_search(trained):
    p=calibration_problems()[0]
    for order in [(),('cnot','cnot'),('not_a_resource',)]:
        with pytest.raises(ValueError):optimize_native_resources(p,trained[0],objectives=order)
    with pytest.raises(ValueError):optimize_native_resources(p,NativeHierarchy())
    with pytest.raises(ValueError):optimize_native_resources(p,trained[0],audit_edges=0)


def test_original_api_now_checks_nonzero_optimum(trained):
    p=calibration_problems()[1]
    r=optimize_native(p,trained[0],objective='t_count',limits=LIMITS)
    assert r['status']=='optimal_under_numerical_contract'
    assert verify_native_optimization(p,r)['optimality_verified']
