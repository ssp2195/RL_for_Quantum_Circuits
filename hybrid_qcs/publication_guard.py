"""Validation-gated shrinkage of learned linear residuals.

This is an explicitly dated amendment after the original primary campaign found
negative transfer. Its validation uses only the original validation set; a NEW
logical-target-orbit holdout is generated only after selection is frozen. The
original primary results remain unchanged. Shrinkage is not a theorem of
out-of-distribution non-regression and beta=0 must be labelled untrained.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from .phase_policy import PhaseHierarchy, ARMS, outer_prior
from .phase_contract import PhaseProblem
from .publication_corpus import corpus, key, regression_cases, _random_problem
from .publication_pipeline import anytime_discovery, public_parity_cnot_bound
from .publication_runner import read_lock, write_json, environment, test_contract
from .resource_domain import canonical_digest
from .resource_search import WorkLimits

BETAS = (0., .01, .03, .1, .25, .5, 1.)


def shrink(model, beta):
    if not model.frozen or beta not in BETAS:
        raise ValueError('require a frozen model and declared shrinkage coefficient')
    candidate = copy.deepcopy(model)
    candidate.outer = outer_prior() + beta * (candidate.outer - outer_prior())
    for arm in ARMS:
        candidate.responses[arm] *= beta
    candidate._theta.clear()
    return candidate.freeze()


def validation_select(out):
    out = Path(out)
    plan = read_lock(out)
    directory = out / 'guard_amendment'
    directory.mkdir(exist_ok=True)
    amendment = {
        'schema': 'validation-shrinkage-amendment-v1',
        'parent_protocol_digest': plan['digest'], 'betas': list(BETAS),
        'seeds': plan['seeds'], 'validation_repeats': 2,
        'selection': 'highest validation mean CNOT savings among coefficients with success at least beta=0; ties prefer smaller beta',
        'validation_only': True, 'primary_results_reused_for_selection': False,
        'trigger': 'original locked primary experiment showed negative transfer; preserve that evidence',
        'confirmation': '12 new logical target orbits, three widths of clean workspace each, five seeds, three timings; paired bootstrap by logical target',
        'confirmatory_generation_seed': 39092026,
        'limits': plan['discovery_limits'],
        'no_test_non_regression_guarantee': True,
    }
    amendment['digest'] = canonical_digest(amendment)
    lock = directory / 'protocol.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == amendment
    else:
        write_json(lock, amendment)
        write_json(directory / 'lock_event.json', {'utc': datetime.now(timezone.utc).isoformat(), 'environment': environment()})
    selection_path = directory / 'selection.json'
    if selection_path.exists():
        return json.loads(selection_path.read_text())
    jobs = [(seed, rep, i, beta) for seed in plan['seeds'] for rep in range(2)
            for i in range(len(corpus()[1])) for beta in BETAS]
    np.random.default_rng(94273).shuffle(jobs)
    models = {seed: PhaseHierarchy.load(out / 'checkpoints' / f'full-seed-{seed}.json') for seed in plan['seeds']}
    candidates = {(seed, beta): shrink(model, beta) for seed, model in models.items() for beta in BETAS}
    rows = []
    for seed, rep, i, beta in jobs:
        p = test_contract(corpus()[1][i])
        r = anytime_discovery(p, candidates[seed, beta], limits=WorkLimits(**plan['discovery_limits']))
        success = r['best'] is not None
        rows.append({'seed': seed, 'repeat': rep, 'validation_index': i, 'beta': beta,
                     'success': success, 'savings': 1-r['best']['resources']['cnot']/p.max_cnot if success else 0.,
                     'result': r})
        if len(rows) % 80 == 0:
            print('shrinkage validation', len(rows), '/', len(jobs), flush=True)
    stats = {beta: {'successes': sum(r['success'] for r in rows if r['beta'] == beta),
                    'mean_savings': float(np.mean([r['savings'] for r in rows if r['beta'] == beta]))}
             for beta in BETAS}
    eligible = [beta for beta in BETAS if stats[beta]['successes'] >= stats[0.]['successes']]
    chosen = max(eligible, key=lambda beta: (stats[beta]['mean_savings'], -beta))
    selection = {'amendment_digest': amendment['digest'], 'selected_beta': chosen,
                 'stats': stats, 'validation_runs': len(rows),
                 'frozen_utc': datetime.now(timezone.utc).isoformat(),
                 'uses_learned_correction': chosen > 0,
                 'model_digests': {str(seed): candidates[seed, chosen].digest for seed in plan['seeds']}}
    for seed in plan['seeds']:
        candidates[seed, chosen].save(directory / 'checkpoints' / f'seed-{seed}.json')
    write_json(directory / 'validation.json', rows)
    write_json(selection_path, selection)
    print('selected shrinkage beta', chosen, stats, flush=True)
    return selection


def confirmation_targets():
    used = {key(p) for group in corpus() for p in group} | {key(p) for p in regression_cases()}
    rng = np.random.default_rng(39092026)
    targets = []
    while len(targets) < 12:
        i = len(targets)
        n = (3, 4, 5)[i % 3]
        p = _random_problem(rng, n, 3+i//3, f'confirmation-{i:02d}')
        signature = key(p)
        if signature in used:
            continue
        used.add(signature)
        targets.append(p)
    return tuple(targets)


def confirm(out):
    out = Path(out)
    plan = read_lock(out)
    directory = out / 'guard_amendment'
    selection = json.loads((directory / 'selection.json').read_text())
    targets = confirmation_targets()
    write_json(directory / 'targets.json', [{'orbit': key(p), 'problem': p.manifest()} for p in targets])
    models = {s: PhaseHierarchy.load(out/'checkpoints'/f'full-seed-{s}.json') for s in plan['seeds']}
    guarded = {s: PhaseHierarchy.load(directory/'checkpoints'/f'seed-{s}.json') for s in plan['seeds']}
    priors = {s: PhaseHierarchy(seed=s).freeze() for s in plan['seeds']}
    path = directory / 'confirmation.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r['amendment_digest'] == selection['amendment_digest']
            done.add((r['target_index'], r['ancillas'], r['seed'], r['repeat'], r['method']))
    jobs = [(i, a, seed, rep, method) for i in range(12) for a in (0,1,2)
            for seed in plan['seeds'] for rep in range(3) for method in ('untrained','hierarchy','guarded')]
    np.random.default_rng(638190).shuffle(jobs)
    with path.open('a') as stream:
        for i, a, seed, rep, method in jobs:
            if (i,a,seed,rep,method) in done:
                continue
            p = test_contract(replace(targets[i], ancillas=a))
            model = {'untrained': priors, 'hierarchy': models, 'guarded': guarded}[method][seed]
            r = anytime_discovery(p, model, limits=WorkLimits(**plan['discovery_limits']))
            success = r['best'] is not None
            row = {'amendment_digest': selection['amendment_digest'],
                   'target_index': i, 'target_orbit': key(targets[i]), 'ancillas': a,
                   'seed': seed, 'repeat': rep, 'method': method, 'problem': p.manifest(),
                   'beta': selection['selected_beta'] if method=='guarded' else int(method=='hierarchy'),
                   'success': success, 'savings': 1-r['best']['resources']['cnot']/p.max_cnot if success else 0.,
                   'result': r}
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            stream.flush()
            done.add((i,a,seed,rep,method))
            if len(done) % 180 == 0:
                print('confirmation', len(done), '/', len(jobs), flush=True)
    write_json(directory / 'complete.json', {'rows': len(done), 'utc': datetime.now(timezone.utc).isoformat()})


def analyze_guard(out):
    from .publication_analysis import paired_interval
    directory = Path(out) / 'guard_amendment'
    selection = json.loads((directory / 'selection.json').read_text())
    rows = [json.loads(line) for line in (directory / 'confirmation.jsonl').read_text().splitlines()]
    assert len(rows) == 1620
    values = {method: [float(np.mean([r['savings'] for r in rows
                                     if r['method']==method and r['target_index']==i])) for i in range(12)]
              for method in ('untrained','hierarchy','guarded')}
    result = {'selected_beta': selection['selected_beta'], 'runs': len(rows), 'logical_target_clusters': 12,
              'uses_learned_correction': selection['uses_learned_correction'],
              'methods': {method: {'runs': 540, 'successes': sum(r['success'] for r in rows if r['method']==method),
                                  'mean_savings': float(np.mean(values[method])),
                                  'strict_improvement_runs': sum(r['result']['improvements']>0 for r in rows if r['method']==method)}
                          for method in values},
              'guarded_minus_untrained': paired_interval(np.array(values['guarded'])-values['untrained']),
              'hierarchy_minus_untrained': paired_interval(np.array(values['hierarchy'])-values['untrained']),
              'no_test_based_reselection': True}
    write_json(directory / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('select','confirm','analyze'))
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/publication_v1'))
    args = parser.parse_args()
    result = {'select': validation_select, 'confirm': confirm, 'analyze': analyze_guard}[args.command](args.output_dir)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
