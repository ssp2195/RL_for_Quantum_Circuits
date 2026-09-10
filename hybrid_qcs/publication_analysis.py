"""Analyze the locked campaign without selecting policies from test results.

The independent experimental unit is a target (25), not a seed/repetition row
(2,625). Average all five seeds and three repeats within each target before a
paired target-cluster bootstrap. Discovery failures remain in every endpoint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .phase_contract import certify_phase, problem_from_manifest
from .phase_audit import verify_phase_cover
from .phase_baselines import verify_rank_bound
from .publication_runner import PRIMARY, read_lock, write_json

LABELS = {
    'hierarchy': 'Trained hierarchy', 'untrained': 'Untrained identical prior',
    'greedy': 'Greedy', 'outer': 'Outer only', 'inner': 'Inner only',
    'uniform_cost': 'Uniform cost', 'audit_only': 'Audit only',
    'no_budget': 'Retrained: no budget', 'no_workspace': 'Retrained: no workspace',
    'full_panel': 'Full-frontier scoring', 'graysynth': 'Adapted GraySynth',
    'rank_partition': 'Rank partition reference', 'parity_star': 'Parity star',
}


def load_rows(out, prefix='primary'):
    return [json.loads(line) for path in sorted((Path(out) / 'raw').glob(prefix + '*.jsonl'))
            for line in path.read_text().splitlines()]


def paired_interval(differences, *, seed=918, repeats=10000):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError('a finite nonempty vector of paired target effects is required')
    indices = np.random.default_rng(seed).integers(len(values), size=(repeats, len(values)))
    sampled = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [.025, .975])
    return {'mean': float(values.mean()), 'lower_95': float(low), 'upper_95': float(high),
            'clusters': len(values), 'resamples': repeats,
            'unit': 'target; seeds and timings averaged within target first',
            'scope': 'exploratory target-population interval; training-seed uncertainty also reported separately'}


def _endpoint(row, name, cap):
    result = row['result']
    if name == 'savings':
        return row['normalized_savings']
    if name == 'success':
        return float(row['success'])
    if name == 'first_correct_capped':
        t = result.get('first_correct_seconds')
        return min(cap, t) if t is not None else cap
    raise ValueError(name)


def analyze(out):
    out = Path(out)
    plan = read_lock(out)
    rows = load_rows(out)
    target_count = len(plan['split']['test'])
    expected = {(target, seed, rep, method)
                for target in range(target_count) for seed in plan['seeds']
                for rep in range(plan['timing_repeats']) for method in PRIMARY}
    found = set()
    for row in rows:
        key = (row['target_index'], row['seed'], row['repeat'], row['method'])
        if key in found or key not in expected or row['protocol_digest'] != plan['digest']:
            raise AssertionError('duplicate, unexpected, or mismatched primary record')
        found.add(key)
        if row['method'] != 'audit_only':
            assert row['result']['audit_calls'] == 0
            assert row['result']['audit_witness_used'] is False
        assert row['success'] == (row['result']['best'] is not None)
        if row['success']:
            assert row['result']['best']['witness']['success']
            assert row['cnot'] <= row['problem']['max_cnot']
    if found != expected:
        raise AssertionError(f'incomplete locked campaign: {len(found)} / {len(expected)}')
    cap = plan['discovery_limits']['wall_seconds']
    by_method = {m: [r for r in rows if r['method'] == m] for m in PRIMARY}
    summaries = {}
    target_values = {}
    for method, values in by_method.items():
        target_values[method] = {
            endpoint: [float(np.mean([_endpoint(r, endpoint, cap) for r in values
                                     if r['target_index'] == i])) for i in range(target_count)]
            for endpoint in ('savings', 'success', 'first_correct_capped')}
        successful = [r for r in values if r['success']]
        summaries[method] = {
            'runs': len(values), 'successes': len(successful),
            'success_rate': len(successful) / len(values),
            'mean_normalized_savings_all_runs': float(np.mean(target_values[method]['savings'])),
            'mean_wall_seconds': float(np.mean([r['result']['wall_seconds'] for r in values])),
            'mean_cpu_seconds': float(np.mean([r['result']['cpu_seconds'] for r in values])),
            'mean_first_correct_capped': float(np.mean(target_values[method]['first_correct_capped'])),
            'mean_cnot_successful_only': float(np.mean([r['cnot'] for r in successful])) if successful else None,
            'strict_improvement_runs': sum(r['result']['improvements'] > 0 for r in values),
            'strict_improvements_total': sum(r['result']['improvements'] for r in values),
            'mean_attempted_edges': float(np.mean([r['result']['edges'] for r in values])),
            'mean_lookahead_transitions': float(np.mean([r['result']['lookahead_transitions'] for r in values])),
            'mean_selection_seconds': float(np.mean([sum(x.get('profile', {}).get('feature_seconds', 0)
                                                       for x in r['result']['rounds']) for r in values])),
            'maximum_scored_panel': max((x.get('profile', {}).get('max_scored_panel', 0)
                                        for r in values for x in r['result']['rounds']), default=0),
        }
    contrasts = {}
    for control in PRIMARY:
        if control == 'hierarchy':
            continue
        contrasts[control] = {endpoint: paired_interval(
            np.array(target_values['hierarchy'][endpoint]) - np.array(target_values[control][endpoint]))
            for endpoint in ('savings', 'success', 'first_correct_capped')}
    seed_effects = {}
    for seed in plan['seeds']:
        selected = [r for r in rows if r['seed'] == seed]
        seed_effects[str(seed)] = {
            m: float(np.mean([r['normalized_savings'] for r in selected if r['method'] == m]))
            for m in PRIMARY}
    secondary = load_rows(out, 'secondary')
    secondary_summary = {}
    for method in ('no_budget', 'no_workspace', 'full_panel'):
        values = [r for r in secondary if r['method'] == method]
        if values:
            # Match the secondary's one timing repeat with the primary repeat 0.
            paired = []
            for i in range(target_count):
                treatment = [r['normalized_savings'] for r in rows
                             if r['target_index'] == i and r['method'] == 'hierarchy' and r['repeat'] == 0]
                control = [r['normalized_savings'] for r in values if r['target_index'] == i]
                if not control:
                    raise AssertionError('incomplete secondary target')
                paired.append(np.mean(treatment) - np.mean(control))
            secondary_summary[method] = {
                'runs': len(values), 'successes': sum(r['success'] for r in values),
                'mean_savings': float(np.mean([r['normalized_savings'] for r in values])),
                'hierarchy_minus_ablation': paired_interval(paired),
                'max_panel': max((x.get('profile', {}).get('max_scored_panel', 0)
                                 for r in values for x in r['result']['rounds']), default=0)}
    constructive = json.loads((out / 'constructive_controls.json').read_text()) if (out / 'constructive_controls.json').exists() else []
    controls = {}
    for method in ('parity_star', 'rank_partition', 'graysynth'):
        values = [r for r in constructive if r['method'] == method]
        if values:
            control_targets = [float(np.mean([r['normalized_savings'] for r in values if r['target_index'] == i])) for i in range(target_count)]
            controls[method] = {'runs': len(values), 'successes': sum(r['success'] for r in values),
                               'mean_savings': float(np.mean(control_targets)),
                               'mean_wall_seconds': float(np.mean([r['wall_seconds'] for r in values])),
                               'hierarchy_minus_control': paired_interval(np.array(target_values['hierarchy']['savings']) - control_targets)}
    training = [json.loads((out / 'training' / f'full-seed-{seed}.json').read_text()) for seed in plan['seeds']]
    result = {
        'protocol_digest': plan['digest'], 'primary_rows': len(rows), 'target_clusters': target_count,
        'models_frozen_before_test': True, 'audit_replacement_count': 0,
        'methods': summaries, 'paired_hierarchy_minus_control': contrasts,
        'seed_savings': seed_effects, 'target_values': target_values,
        'secondary': secondary_summary, 'constructive': controls,
        'mean_training_cpu_seconds': float(np.mean([r['cpu_seconds'] for r in training])),
        'mean_training_wall_seconds': float(np.mean([r['wall_seconds'] for r in training])),
        'training_amortization_cpu_seconds_per_target': {
            str(n): float(np.mean([r['cpu_seconds'] for r in training])) / n for n in (25, 100, 1000)},
        'multiple_comparisons': 'hierarchy minus untrained is the primary contrast; other intervals are descriptive, not multiplicity-adjusted significance tests',
        'deadline_qualification': 'cooperative caps; full observed time includes certification and possible overrun',
        'no_global_t_count_or_unrestricted_optimality_claim': True,
    }
    write_json(out / 'analysis.json', result)
    with (out / 'methods.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', *next(iter(summaries.values())).keys()], lineterminator='\n')
        writer.writeheader()
        for method, stats in summaries.items():
            writer.writerow({'method': method, **stats})
    with (out / 'paired_target_effects.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['target_index', *PRIMARY, 'hierarchy_minus_untrained'])
        for i in range(target_count):
            writer.writerow([i, *[target_values[m]['savings'][i] for m in PRIMARY],
                             target_values['hierarchy']['savings'][i] - target_values['untrained']['savings'][i]])
    return result


def verify_saved_campaign(out):
    """Replay every unique saved primary/secondary witness and all proof families."""
    out = Path(out)
    plan = read_lock(out)
    checked = set()
    witnessed_runs = 0
    all_rows = load_rows(out) + load_rows(out, 'secondary')
    for row in all_rows:
        r = row['result']
        if row['method'] != 'audit_only':
            assert r['audit_calls'] == 0 and not r['audit_witness_used']
        if row['success']:
            witnessed_runs += 1
        circuits = [(run['problem'], run['witness']) for run in r.get('rounds', []) if run.get('witness')]
        if row['method'] == 'audit_only' and r['best']:
            circuits = [(row['problem'], r['best']['witness'])]
        for manifest, witness in circuits:
            p = problem_from_manifest(manifest)
            key = (p.digest, tuple(witness['tokens']))
            if key in checked:
                continue
            cert = certify_phase(p, witness['tokens'], dag=True)
            assert cert['success'] and cert['native'] == witness['native']
            assert json.dumps(cert['resources'], sort_keys=True) == json.dumps(witness['resources'], sort_keys=True)
            checked.add(key)
    covers = []
    for path in sorted((out / 'proofs').glob('*.json')):
        verdict = verify_phase_cover(json.loads(path.read_text()))
        assert verdict['valid'], verdict
        covers.append({'path': str(path.relative_to(out)), 'check': verdict})
    calibration_path = out / 'ancilla_calibration.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else []
    rank_count = 0
    for row in calibration:
        assert verify_rank_bound(row['rank_bound'])
        rank_count += 1
        if row['cnot_exclusion'] is not None:
            verdict = verify_phase_cover(row['cnot_exclusion'])
            assert verdict['valid'], verdict
            covers.append({'path': 'ancilla_calibration.json', 'seed': row['seed'], 'ancillas': row['ancillas'], 'check': verdict})
        for run in row['discovery']['rounds']:
            if run['witness']:
                assert certify_phase(problem_from_manifest(run['problem']), run['witness']['tokens'], dag=True)['success']
    post_path = out / 'post_audit.json'
    post = json.loads(post_path.read_text()) if post_path.exists() else []
    for row in post:
        assert verify_rank_bound(row['rank_bound'])
        rank_count += 1
    result = {'protocol_digest': plan['digest'], 'unique_native_witnesses_replayed': len(checked),
              'successful_primary_secondary_runs': witnessed_runs, 'native_witnesses_valid': True,
              'closed_covers_verified': len(covers), 'rank_bounds_verified': rank_count,
              'cover_checks': covers, 'auditor_replacement_of_discovery': False}
    write_json(out / 'independent_replay.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/publication_v1'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify_saved_campaign(args.output_dir) if args.verify else analyze(args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ('target_values', 'cover_checks')}, indent=2))


if __name__ == '__main__':
    main()
