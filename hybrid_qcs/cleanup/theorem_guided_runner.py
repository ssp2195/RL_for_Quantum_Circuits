"""Reproducible M1/M2 integration ablation on BNN and staged correction circuits.

All resource vectors are produced by the independent branch-complete verifier.
No external simulator estimates, inferred speedups, or trained-policy advantages
are ascribed to the theorem-guided construction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .bnn_amplitude_study import CASES, bnn_result
from .contract import evaluate
from .general_boolean import BooleanPhaseProblem, compile_boolean_phase
from .phase_protocol import verify_phase
from .staged_frontier import CausalCorrectionFrontier
from .generic_search import GenericFrontierSearch, train_generic_hierarchy


def study_bnn():
    results=[]
    for case in CASES:
        ref=bnn_result(case)
        n=case.n
        nominal=ref['nominal_class']
        truth=tuple(int((x ^ case.reference).bit_count()<=case.radius
                        and int(__predict(case,x))!=nominal) for x in range(1<<n))
        p=BooleanPhaseProblem.from_truth_table(case.name,truth)
        number=sum(m.bit_count()>=3 for m in p.target_polynomial)
        modes=tuple('measured' for _ in range(number))
        m1=compile_boolean_phase(p,modes,m1_guided=True)
        checked=verify_phase(p,m1)
        if not checked['valid']:
            raise AssertionError('M1 BNN exact oracle verification failed: '+checked['reason'])
        raw=ref['measured']
        new=checked['resources']
        if (raw['peak_aux'],raw['t_count'],raw['cnot']) != (
            new['peak_aux'],new['t_count'],new['cnot']):
            raise AssertionError('M1 changed declared primitive resource counts')
        results.append({
            'name':ref['name'],'description':ref['description'],
            'n':n,'hamming_radius':case.radius,
            'search_size':ref['search_size'],'violations':ref['marked_count'],
            'measured_control':raw,
            'm1_generated_measured':new,
            'previous_ancilla_first':ref['ancilla_first_menu'],
            'verified_all_outcomes':checked['all_inputs_and_outcomes'],
        })
    return results


def __predict(case,x):
    from .bnn_amplitude_study import predict
    return predict(case,x)


def study_stages():
    parameters=(
        ('one_round_q3',3,1,3),
        ('three_round_q3',3,3,1),
        ('one_round_q4',4,1,4),
        ('two_round_q4',4,2,2),
        ('four_round_q4',4,4,1),
        ('infeasible_q4',4,1,1),
        ('infeasible_q6',6,2,1),
    )
    rows=[]
    for name,q,rounds,cap in parameters:
        guarded=CausalCorrectionFrontier(q,rounds,cap,m2_enabled=True).run()
        original=CausalCorrectionFrontier(q,rounds,cap,m2_enabled=False).run()
        if bool(guarded['selected']) != bool(original['selected']):
            raise AssertionError('M2 pruning changed feasibility')
        if guarded['selected'] and guarded['selected']['resources'] != original['selected']['resources']:
            raise AssertionError('M2 pruning changed the exact optimum')
        rows.append({
            'case':name,'q':q,'max_correction_stages':rounds,
            'correction_ancilla_cap':cap,
            'status':guarded['status'],
            'm1_generated':bool(guarded['selected']),
            'selected_correction_ancillas':(
                guarded['selected']['peak_correction_ancillas'] if guarded['selected'] else None),
            'resources':(guarded['selected']['resources'] if guarded['selected'] else None),
            'search_with_m2':{k:guarded[k] for k in (
                'generated_records','expanded_records','expanded_edges',
                'm2_pruned_records','verified_schedules','m2_rank_checks')},
            'search_without_m2':{k:original[k] for k in (
                'generated_records','expanded_records','expanded_edges',
                'm2_pruned_records','verified_schedules','m2_rank_checks')},
            'same_verified_optimum':True,
        })
    return rows


def study_learning():
    training=(BooleanPhaseProblem('train-cubic',3,frozenset({7})),
              BooleanPhaseProblem('train-quartic',4,frozenset({15})))
    trained=train_generic_hierarchy(training,seed=11,episodes=8)
    target=BooleanPhaseProblem('heldout',4,frozenset({7,15}))
    original=GenericFrontierSearch(target,m1_guided=False).run(trained)
    guided=GenericFrontierSearch(target,m1_guided=True).run(trained)
    if not original['complete_library_search'] or not guided['complete_library_search']:
        raise AssertionError('incomplete exact generic policy evaluation')
    return {
        'policy':'frozen outer SARSA, inner LinUCB',
        'old':{'expansions':original['expansions'],'candidates':original['candidate_count'],
               'peak_aux':original['selected']['resources']['peak_aux'],
               't_count':original['selected']['resources']['t_count']},
        'm1_guided':{'expansions':guided['expansions'],'candidates':guided['candidate_count'],
                     'peak_aux':guided['selected']['resources']['peak_aux'],
                     't_count':guided['selected']['resources']['t_count']},
        'm2_applied_to_generic_frontier':False,
        'reason':'generic ANF grammar performs immediate local cleanup; M2 prunes the separate staged correction frontier',
    }


def run():
    return {
        'schema':'M1-M2-frontier-integration-v1',
        'scope':'exact phase oracle and staged correction architecture, not global optimum',
        'bnn_oracles':study_bnn(),
        'staged_correction_frontier':study_stages(),
        'learning_control':study_learning(),
        'limitations':[
            'BNN circuits use immediate local garbage cleanup; M2 does not reduce their resource counts',
            'M2 rejects only provably inadequate correction-stage capacity',
            'M3 test identity instruments include actual original garbage preparation',
            'finite gate grammar and stage model; no global ancilla optimum',
        ],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for item in result['staged_correction_frontier']:
        before=item['search_without_m2']['expanded_edges']
        after=item['search_with_m2']['expanded_edges']
        print(item['case'],item['status'],'edges',before,'->',after,
              'M2 pruned',item['search_with_m2']['m2_pruned_records'])
    print('BNN phase oracles verified:',len(result['bnn_oracles']))


if __name__ == '__main__':
    main()
