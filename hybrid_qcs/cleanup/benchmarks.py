"""Prospective five-problem registry and separate training specifications.

These are synthetic complete common-enable feature-bank phase oracles, not
claims of full trained BNN implementations. r>m makes the helper-size comparison
with the data-independent row construction relevant. Consumers include shared
quadratic terms; no blanket global optimality is inferred from materialization.
"""
from __future__ import annotations
from .contract import CleanupProblem, Consumer, Limits, digest

SHAPES = ((3,1),(4,2),(5,2),(6,3),(8,3))
NAMES = ('enabled-three-feature-coupling','two-channel-feature-ring',
         'mixed-five-row-predicate','three-channel-cross-feature-bank',
         'wide-shared-mask-oracle')


def make_problem(r,m,*,name,salt=0,split='test',limits=None):
    terms=set();n=r*m
    for k in range(n):
        i,j=divmod(k,m)
        # Distinct row/column pair, plus occasional direct product consumers.
        other=((i+1+salt%max(1,r-1))%r)*m+(j+1)%m
        if other!=k:terms.add(tuple(sorted((f'f{i}_{j}',f'f{other//m}_{other%m}'))))
        if (k+salt)%3==0:terms.add((f'f{i}_{j}',))
        if m>1 and (k+salt)%4==0:terms.add((f'b{(i+2)%r}',f'f{i}_{j}'))
    if salt%2:terms.add(('a',f'x{salt%m}'))
    return CleanupProblem(name,r,m,Consumer(tuple(sorted(terms))),limits or Limits(),split=split)


def candidate_problems():
    return tuple(make_problem(r,m,name=name,salt=j,split='test') for j,((r,m),name) in enumerate(zip(SHAPES,NAMES)))


def training_problems():
    # Training widths 3,4,6 are disjoint from test widths 5,7,8,10,12.
    shapes=((1,1),(1,2),(2,1),(1,4),(2,3),(3,2),(4,1))
    problems=tuple(make_problem(r,m,name=f'train-{r}-{m}-{salt}',salt=salt,split='train')
                   for r,m in shapes for salt in (7,11,16))
    test={p.oracle_digest for p in candidate_problems()}
    unique={}
    for p in problems:
        if p.oracle_digest not in test:unique.setdefault(p.oracle_digest,p)
    return tuple(unique.values())


def protocol_manifest():
    return {'schema':'cleanup-study-lock-v1','seeds':[11,19,23,31,47],
            'training_stages':[64,96,24], 'training_edges':512,
            'test_repetitions':3,'test_edges':4096,'test_seconds':10.,
            'primary_resources':['peak_aux','t_count','cnot','native_gates'],
            'secondary':['native_depth','worst_case_ticks','wall_seconds','edges'],
            'primitive':'and4', 'measurement_latency_is_model_not_measurement':True,
            'candidates':[p.manifest() for p in candidate_problems()],
            'training':[p.manifest() for p in training_problems()],
            'controls':['theorem_untrained','control_helpers_trained','coherent_inverse_trained',
                        'deterministic_smaller_side','deterministic_control_side','bank_free_anf'],
            'claim_boundary':'restricted materialization comparison; neither global oracle nor RL superiority is presumed'}
