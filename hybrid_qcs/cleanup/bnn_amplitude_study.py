"""Exact BNN robustness phase-oracle benchmarks for amplitude amplification.

Each phase predicate marks x inside a Hamming ball around 0^n if its
binarized neural network prediction differs from the nominal prediction.
The compiler and the independent verifier operate on the entire input cube.
Preparation, reflection, and Grover repetitions are not included.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from .contract import evaluate
from .general_boolean import BooleanPhaseProblem, compile_boolean_phase, optimize_boolean_phase
from .phase_protocol import verify_phase
from .benchmarks import candidate_problems
from .consumer_search import synthesize_oracle


@dataclass(frozen=True)
class BNNCase:
    name: str
    description: str
    radius: int
    weights: tuple
    biases: tuple
    reference: int = 0

    @property
    def n(self):
        return len(self.weights[0][0])


CASES = (
    BNNCase('B1', '3-input single threshold neuron; one radius-1 violation', 1,
            (((1,-1,-1),),), ((-2,),)),
    BNNCase('B2', '4-input threshold pattern detector; one radius-2 violation', 2,
            (((1,1,-1,-1),),), ((-3,),)),
    BNNCase('B3', '5-input BNN with 3 hidden neurons and majority output', 2,
            (((1,1,1,-1,-1),(-1,1,-1,1,1),(1,-1,1,1,-1)),
             ((1,1,1),)), ((-1,1,0),(-1,)), 14),
)


def predict(case: BNNCase, x: int) -> int:
    values = tuple(2*((x>>i)&1)-1 for i in range(case.n))
    for layer,biases in zip(case.weights,case.biases,strict=True):
        values = tuple(1 if sum(w*z for w,z in zip(weights,values,strict=True))+bias>=0 else -1
                       for weights,bias in zip(layer,biases,strict=True))
    assert len(values)==1
    return int(values[0]==1)


def bnn_result(case: BNNCase) -> dict:
    n=case.n
    nominal=predict(case,case.reference)
    truth=tuple(int((x ^ case.reference).bit_count()<=case.radius and predict(case,x)!=nominal)
                for x in range(1<<n))
    assert not truth[case.reference] and any(truth)
    problem=BooleanPhaseProblem.from_truth_table(case.name,truth)
    assert all(evaluate(problem.target_polynomial,x)==truth[x] for x in range(1<<n))
    nonlinear=sum(m.bit_count()>=3 for m in problem.target_polynomial)
    endpoints={}
    for mode in ('measured','direct'):
        protocol=compile_boolean_phase(problem,(mode,)*nonlinear)
        report=verify_phase(problem,protocol)
        assert report['valid'], (case.name,mode,report)
        protocol.persistent_tail()
        endpoints[mode]=report['resources']
    optimum=optimize_boolean_phase(problem,max_nonlinear_terms=12)
    assert optimum['status']=='certified_finite_menu_optimum'
    selected=optimum['selected']['resources']
    for endpoint in endpoints.values():
        assert (selected['peak_aux'],selected['t_count'])<=(endpoint['peak_aux'],endpoint['t_count'])
    return {
        'name':case.name,'description':case.description,'n':n,
        'radius':case.radius,'nominal_class':nominal,
        'reference_input':''.join(str((case.reference>>i)&1) for i in range(n)),
        'weights':[[list(row) for row in layer] for layer in case.weights],
        'biases':[list(row) for row in case.biases],
        'search_size':sum((x ^ case.reference).bit_count()<=case.radius for x in range(1<<n)),
        'marked_count':sum(truth),
        'marked_inputs':[''.join(str((x>>i)&1) for i in range(n))
                         for x,v in enumerate(truth) if v],
        'anf_monomial_masks':sorted(problem.target_polynomial),
        'nonlinear_terms':nonlinear,
        'menu_candidates':optimum['candidate_count'],
        'menu_modes':list(optimum['selected']['modes']),
        'measured':endpoints['measured'],
        'direct':endpoints['direct'],
        'ancilla_first_menu':selected,
        'verified_on_full_cube':True,
        'globally_optimal':False,
    }


def run() -> dict:
    bnns=[bnn_result(case) for case in CASES]
    descriptions=(
        'Three enabled feature-product couplings',
        'Two-channel ring of cross-feature products',
        'Mixed five-row product and input-product terms',
        'Three-channel cross-feature bank',
        'Eight-row shared-mask feature couplings',
    )
    reference=[]
    for label,p,desc in zip(('E1','E2','E3','E4','E5'),
                            candidate_problems(),descriptions,strict=True):
        objective=('peak_aux','t_count','cnot','native_depth')
        before=synthesize_oracle(p,objective=objective,seconds=20.)
        after=synthesize_oracle(p,objective=objective,include_generic=True,seconds=20.)
        assert before.get('selected') and after.get('selected'),label
        reference.append({
            'label':label,'name':p.name,'description':desc,
            'r':p.r,'m':p.m,'synthetic_not_bnn':True,
            'previous':before['selected']['verification']['resources'],
            'extended':after['selected']['verification']['resources'],
            'method':after['selected']['method'],
        })
    return {
        'schema':'bnn-aa-oracle-v1',
        'source_branch':'stable-generic-phase-cleanup-20261102',
        'goal':'HammingBall(x,x0,r) AND (BNN(x)!=BNN(x0))',
        'scope':'phase oracle only; excludes AA state preparation and reflection',
        'objective':['peak_aux','t_count','cnot','native_depth'],
        'bnn_oracles':bnns,'reference_synthetic_benchmarks':reference,
        'menu_optimality':'exact finite measured/direct ANF menu only, not global',
        'search_uses_heuristics':False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    for row in result['bnn_oracles']:
        print(row['name'],row['marked_count'],row['menu_candidates'],
              row['measured']['peak_aux'],row['ancilla_first_menu']['peak_aux'],
              row['ancilla_first_menu']['t_count'])


if __name__=='__main__':
    main()
