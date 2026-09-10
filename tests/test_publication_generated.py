"""Primary tests: actually synthesize circuits with an already trained hierarchy.

Sixty distinct target orbits x five independently trained seeds = 300 generated
circuits. These are not hand supplied witnesses or auditor-found replacements.
Every test replays ALL promised basis columns and the native DAG. Training must
complete before this module's circuit generation begins (the CI trains first).
"""
import json
import os
from pathlib import Path
import pytest
from hybrid_qcs.phase_contract import certify_phase
from hybrid_qcs.phase_policy import PhaseHierarchy
from hybrid_qcs.phase_search import search_phase
from hybrid_qcs.publication_corpus import regression_cases
from hybrid_qcs.resource_search import WorkLimits

CASES = regression_cases()


@pytest.fixture(scope='session',params=range(5),ids=lambda seed:f'trained-seed-{seed}')
def frozen_publication_hierarchy(request):
    directory=Path(os.environ.get('QCS_PUBLICATION_MODELS','experiments/publication_v1/checkpoints'))
    path=directory/f'full-seed-{request.param}.json'
    if not path.exists():
        pytest.fail(f'Training must complete first; missing frozen checkpoint {path}. Run publication_runner train.')
    model=PhaseHierarchy.load(path)
    assert model.frozen and model.outer_updates>0 and model.inner_updates>0
    assert all(n>0 for n in model.arm_updates.values())
    return model


@pytest.mark.generated
@pytest.mark.parametrize('problem',CASES,ids=lambda p:p.name)
def test_post_training_hierarchy_generates_exact_native_circuit(problem,frozen_publication_hierarchy):
    model=frozen_publication_hierarchy
    before=model.digest
    result=search_phase(problem,model,limits=WorkLimits(8192,30000,5.,5.),dag=True)
    assert result['status']=='feasible', (problem.name,model.seed,result['reason'])
    assert result['witness_source']=='frozen_policy_discovery'
    assert result['profile']['audit_calls']==0 and not result['audit_witness_used']
    assert result['expanded_tokens']
    witness=result['witness']
    assert witness['tokens'] and witness['native']
    assert witness['resources']['t_count']>=1 and witness['resources']['cnot']>=1
    assert witness['resources']['t_count']==problem.t_count
    assert witness['dag_validated'] is True
    assert witness['exact_basis_phase_match']
    assert witness['native_isometry_error']<1e-9 and witness['workspace_leakage']<1e-20
    independent=certify_phase(problem,witness['tokens'],dag=True)
    assert independent['success'] and independent['resources']==witness['resources']
    assert result['profile']['max_scored_panel']<=32
    assert result['profile']['full_dag_policy_reconstructions']==0
    assert model.digest==before


@pytest.mark.generated
def test_trained_hierarchy_generates_bnn_oracle_for_coherent_amplification(frozen_publication_hierarchy):
    """Run the actual learned native circuit, not the desired diagonal oracle."""
    import numpy as np
    from hybrid_qcs.certify import unitary_from_gates
    from hybrid_qcs.model import Gate
    from hybrid_qcs.publication_corpus import bnn_problem, bnn_truth
    p = bnn_problem()
    model = frozen_publication_hierarchy
    digest = model.digest
    result = search_phase(p, model, limits=WorkLimits(20000, 60000, 5., 5.), dag=True)
    assert result['witness_source'] == 'frozen_policy_discovery'
    assert result['profile']['audit_calls'] == 0 and not result['audit_witness_used']
    native = tuple(Gate(name, tuple(qs)) for name, qs in result['witness']['native'])
    state = unitary_from_gates(p.width, native) @ (np.ones(8) / np.sqrt(8))
    amplified = 2 * state.mean() - state
    marked = [x for x, value in enumerate(bnn_truth()) if value]
    assert np.isclose(np.sum(np.abs(amplified[marked]) ** 2), 27/32, atol=1e-12)
    assert model.digest == digest


@pytest.mark.generated
@pytest.mark.parametrize('ancillas,t_depth', [(0, 2), (1, 1), (2, 1)])
def test_trained_generation_attains_a_checked_ancilla_tradeoff(ancillas, t_depth, frozen_publication_hierarchy):
    from hybrid_qcs.phase_contract import PhaseProblem
    from hybrid_qcs.phase_baselines import rank_tdepth_certificate, verify_rank_bound
    from hybrid_qcs.phase_audit import audit_phase, verify_phase_cover
    from hybrid_qcs.publication_pipeline import anytime_discovery
    p = PhaseProblem(2, ((1, 1), (2, 1), (3, 7)), ancillas, 6, 20, 32, max_t_depth=t_depth)
    digest = frozen_publication_hierarchy.digest
    result = anytime_discovery(p, frozen_publication_hierarchy, limits=WorkLimits(8192, 30000, 3., 3.))
    # Discovery must finish before auditing begins; no auditor witness is usable.
    assert result['best'] and result['audit_calls'] == 0 and not result['audit_witness_used']
    w = result['best']['witness']
    assert w['resources']['t_depth'] == t_depth and w['resources']['t_count'] == 3
    assert w['resources']['cnot'] == (2 if ancillas == 0 else 4)
    assert certify_phase(p, w['tokens'], dag=True)['success']
    rank = rank_tdepth_certificate(p)
    assert verify_rank_bound(rank, p) and rank['lower_t_depth'] == t_depth
    tighter = p.cap('cnot', w['resources']['cnot'] - 1)
    audit = audit_phase(tighter, limits=WorkLimits(200000, 60000, 3., 3.))
    assert audit['witness'] is None and verify_phase_cover(audit['proof'], tighter)['valid']
    assert frozen_publication_hierarchy.digest == digest
