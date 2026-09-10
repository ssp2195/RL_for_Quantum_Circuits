"""Specification-only train/validation/test splits and Boolean BNN application.

No function in this module constructs a target circuit. Orbit keys exclude
input-wire permutations, input complements and an input-independent phase.
Held-out results must be clustered by target, not treated as independent per
seed/ancilla/timing repetitions.
"""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
from itertools import permutations
import numpy as np
from .phase_contract import PhaseProblem, boolean_phase_coefficients
from .resource_domain import canonical_digest


@lru_cache(maxsize=2048)
def orbit_key(n, exponents):
    array = np.asarray(exponents, dtype=np.int16)
    bases = np.arange(1 << n)
    best = None
    for perm in permutations(range(n)):
        transformed = sum(((bases >> q) & 1) << perm[q] for q in range(n))
        for flip in range(1 << n):
            values = array[transformed ^ flip]
            candidate = tuple(int(v) for v in (values - values[0]) % 8)
            if best is None or candidate < best:
                best = candidate
    return canonical_digest({'n': n, 'projective_x_permutation_orbit': best})


def key(p):
    return orbit_key(p.n, p.target_exponents())


def _random_problem(rng, n, count, name, *, ancillas=0, cnot=None, t_depth=64):
    masks = rng.choice(np.arange(1, 1 << n), size=min(count, (1 << n)-1), replace=False)
    coefficients = tuple((int(m), int(rng.choice([1,2,3,5,6,7]))) for m in masks)
    # Ensure that a test actually requires non-Clifford entangling synthesis.
    if not any(m.bit_count() >= 2 and c % 2 for m,c in coefficients):
        coefficients = (*coefficients, ((1<<n)-1, 1))
    p = PhaseProblem(n, coefficients, ancillas, max_cnot=cnot or 4*n*count,
                     max_depth=128, max_gates=256, max_t_depth=t_depth, name=name)
    return p


def corpus():
    rng = np.random.default_rng(19092026)
    training = []
    # Nonlinear/phase-placement curriculum includes tight workspace T-layer tasks.
    for n, count, anc, td in [(2,3,1,1),(2,3,2,1),(2,3,0,2),
                              (3,5,0,4),(3,6,1,4),(3,7,2,4)]:
        for i in range(4):
            training.append(_random_problem(rng,n,count,f'train-{n}-{count}-{anc}-{i}',ancillas=anc,t_depth=td))
    # The exact controlled-S tradeoff is a named calibration, not held-out data.
    training.append(PhaseProblem(2,((1,1),(2,1),(3,7)),1,8,20,32,max_t_depth=1,name='controlled-S-calibration'))
    training.append(PhaseProblem(3,((1,1),(2,1),(3,7),(4,1),(5,7),(6,7),(7,1)),0,8,24,32,max_t_depth=3,name='cubic-calibration'))
    used = {key(p) for p in training}
    validation, testing = [], []
    for label, result, widths, counts in [('validation',validation,(3,4),(3,4,5)),
                                           ('test',testing,(3,4,5),(3,4,5,6))]:
        wanted = 8 if label == 'validation' else 24
        attempts = 0
        while len(result) < wanted:
            i = len(result)
            n = widths[i % len(widths)]
            count = counts[(i // len(widths)) % len(counts)]
            p = _random_problem(rng,n,count,f'{label}-{i:02d}',ancillas=i%3)
            signature = key(p)
            attempts += 1
            if attempts > 10000:
                raise RuntimeError('unable to construct disjoint target orbits')
            if signature in used:
                continue
            used.add(signature)
            result.append(p)
    # A multi-marked, degree-three threshold-network predicate is held out.
    bnn = bnn_problem()
    if key(bnn) in {key(p) for p in training + validation + testing}:
        raise AssertionError('BNN target orbit leaked into another split')
    testing.append(bnn)
    return tuple(training), tuple(validation), tuple(testing)


def bnn_specification():
    # Binary input encoding. These weights/thresholds explicitly define the
    # binary threshold network; no hidden witness is supplied to the synthesizer.
    return {'input_encoding': 'x in {0,1}^3; q0 is least significant',
            'hidden_weights': [[1,1,1],[1,1,1]], 'hidden_thresholds': [1,2],
            'output_weights': [1,-1], 'output_threshold': 1,
            'predicate': 'network output differs from reference label 0',
            'reference_label': 0}


def bnn_truth():
    spec = bnn_specification()
    out=[]
    for x in range(8):
        bits=[(x>>q)&1 for q in range(3)]
        hidden=[int(sum(w*b for w,b in zip(weights,bits))>=threshold)
                for weights,threshold in zip(spec['hidden_weights'],spec['hidden_thresholds'])]
        value=int(sum(w*h for w,h in zip(spec['output_weights'],hidden))>=spec['output_threshold'])
        out.append(value ^ spec['reference_label'])
    return tuple(out)


def bnn_problem():
    n, c, constant=boolean_phase_coefficients(bnn_truth())
    return PhaseProblem(n,c,0,14,40,64,constant,'heldout-bnn-exactly-one',max_t_depth=5)


def regression_cases(count=60):
    """Distinct small target orbits for actual post-training circuit generation."""
    rng=np.random.default_rng(91927)
    train, val, test=corpus()
    forbidden={key(p) for p in train+val+test}
    cases=[]
    attempts = 0
    while len(cases)<count:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("insufficient distinct regression orbits")
        i=len(cases)
        n=2 if i<4 else 3+(i%2)
        p=_random_problem(rng,n,2+(i//3)%3,f'generated-regression-{i:02d}',ancillas=(i//9)%3)
        signature=key(p)
        if signature in forbidden:continue
        forbidden.add(signature);cases.append(p)
    return tuple(cases)


def split_manifest():
    train,val,test=corpus()
    return {'schema':'qcs-publication-split-v1',
            'construction_seed':19092026,
            'disjointness':'exact diagonal target modulo input permutations, complements, and global phase',
            'training':[{'name':p.name,'orbit':key(p),'problem':p.manifest()} for p in train],
            'validation':[{'name':p.name,'orbit':key(p),'problem':p.manifest()} for p in val],
            'test':[{'name':p.name,'orbit':key(p),'problem':p.manifest()} for p in test],
            'bnn':bnn_specification()}
