"""Verify complete evidence and export a compact, lossless outcome/circuit index.

The full profiler/training JSON is distributed as a checksummed archive. The Git
capsule preserves every evaluated outcome, incumbent path, native witness and
contract, rather than duplicating large matrices in every repetition. Profiler
fields not in its explicit schema remain in the full archive, not reconstructed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from .phase_contract import certify_phase, problem_from_manifest
from .phase_baselines import verify_rank_bound
from .publication_corpus import corpus
from .publication_runner import read_lock, test_contract, write_json
from .publication_analysis import load_rows, verify_saved_campaign
from .resource_domain import canonical_digest


def export_and_verify(out: Path):
    out = Path(out)
    plan = read_lock(out)
    capsule = {'schema': 'publication-outcome-capsule-v1', 'protocol_digest': plan['digest'],
               'contracts': {}, 'witnesses': {}, 'runs': [],
               'omitted_from_capsule': 'Full per-round profiling and training trajectories remain in the full raw archive.'}
    checked = set()

    def contract(manifest):
        p = problem_from_manifest(manifest)
        capsule['contracts'][p.digest] = manifest
        return p

    def witness(p, w, *, relaxed=False):
        # Relaxed constructive controls are certified as such, never promoted to
        # feasible outcomes under the original comparison contract.
        if relaxed:
            r = w['resources']
            p = replace(p, max_cnot=max(p.max_cnot, r['cnot']),
                        max_depth=max(p.max_depth, r['depth']),
                        max_gates=max(p.max_gates, r['gates']),
                        max_t_depth=max(p.max_t_depth, r['t_depth']))
        capsule['contracts'][p.digest] = p.manifest()
        payload = {'problem_digest': p.digest, 'tokens': w['tokens'],
                   'native': w['native'], 'resources': w['resources']}
        identifier = canonical_digest(payload)
        if identifier not in checked:
            cert = certify_phase(p, w['tokens'], dag=True)
            if not cert['success'] or cert['native'] != w['native'] or cert['resources'] != w['resources']:
                raise AssertionError('saved witness failed independent native replay')
            payload['verification'] = {k: cert[k] for k in (
                'exact_basis_phase_match', 'native_isometry_error', 'workspace_leakage', 'dag_validated')}
            capsule['witnesses'][identifier] = payload
            checked.add(identifier)
        return identifier

    def record(kind, row, result):
        p = contract(result.get('problem') or row['problem'])
        item = {k: row[k] for k in ('method', 'seed', 'repeat', 'target_index', 'validation_index',
                                   'ancillas', 'beta', 'target_orbit', 'normalized_savings', 'savings') if k in row}
        item.update({'kind': kind, 'problem_digest': p.digest,
                     'policy_digest': result.get('policy_digest'), 'incumbents': []})
        for name in ('status', 'wall_seconds', 'cpu_seconds', 'edges', 'lookahead_transitions',
                     'first_correct_seconds', 'improvements', 'audit_calls', 'audit_witness_used'):
            item[name] = result.get(name)
        if item.get('method') != 'audit_only':
            assert not result.get('audit_witness_used')
            assert result.get('audit_calls', result.get('profile', {}).get('audit_calls', 0)) == 0
        for run in result.get('rounds', []):
            if run.get('witness'):
                p_round = contract(run['problem'])
                item['incumbents'].append(witness(p_round, run['witness']))
        best = result.get('best')
        if best:
            # The original (weaker) contract is sufficient for independently
            # replaying the best witness; its tightened round is stored above.
            item['best_witness'] = witness(p, best['witness'])
        elif result.get('witness'):
            item['best_witness'] = witness(p, result['witness'])
        else:
            item['best_witness'] = None
        item['success'] = item['best_witness'] is not None
        if 'success' in row:
            assert item['success'] == row['success']
        capsule['runs'].append(item)

    primary = load_rows(out)
    secondary = load_rows(out, 'secondary')
    assert len(primary) == 2625 and len(secondary) == 375
    for kind, rows in (('primary', primary), ('secondary', secondary)):
        for row in rows:
            record(kind, row, row['result'])
    directory = out / 'guard_amendment'
    confirmation = [json.loads(line) for line in (directory/'confirmation.jsonl').read_text().splitlines()]
    expected = {(i, a, s, rep, m) for i in range(12) for a in (0, 1, 2) for s in range(5)
                for rep in range(3) for m in ('hierarchy', 'untrained', 'guarded')}
    seen = set()
    for row in confirmation:
        key = tuple(row[k] for k in ('target_index', 'ancillas', 'seed', 'repeat', 'method'))
        assert key in expected and key not in seen
        seen.add(key)
        record('confirmation', row, row['result'])
    assert seen == expected
    for row in json.loads((directory/'validation.json').read_text()):
        record('shrinkage_validation', row, row['result'])
    for row in json.loads((out/'ancilla_calibration.json').read_text()):
        record('ancilla_calibration', {**row, 'method': 'hierarchy'}, row['discovery'])
        assert verify_rank_bound(row['rank_bound'])
    for row in json.loads((out/'boundary_evaluation.json').read_text()):
        record('rank_tight', row, row['result'])
        assert verify_rank_bound(row['rank_bound'])
        w = row['nonlearned_control']
        witness(problem_from_manifest(row['result']['problem']), w, relaxed=True)
    for row in json.loads((out/'constructive_controls.json').read_text()):
        p = test_contract(corpus()[2][row['target_index']])
        contract(p.manifest())
        wid = witness(p, row['certificate'], relaxed=True)
        capsule['runs'].append({k: row[k] for k in (
            'method', 'repeat', 'target_index', 'success', 'normalized_savings', 'wall_seconds', 'cpu_seconds')} |
            {'kind': 'constructive', 'problem_digest': p.digest, 'best_witness': wid,
             'certificate_correct_but_outside_original_bounds': not row['success']})
    app = json.loads((out/'bnn_application.json').read_text())
    record('application', {'method': 'hierarchy', 'seed': 0}, app['discovery'])
    assert app['success'] and abs(app['one_iteration_marked_probability'] - 27/32) < 1e-9
    from .publication_application import evaluator_phase_reference
    reference = evaluator_phase_reference()
    assert reference['native'] == app['nonlearned_compute_phase_uncompute_reference']['native']
    assert reference['resources'] == app['nonlearned_compute_phase_uncompute_reference']['resources']
    initial = verify_saved_campaign(out)
    summary = {'all_declared_campaigns_complete': True, 'outcome_records': len(capsule['runs']),
               'unique_native_witness_contract_pairs': len(checked), 'native_witnesses_valid': True,
               'primary_rows': len(primary), 'secondary_rows': len(secondary), 'confirmation_rows': len(confirmation),
               'closed_covers_verified': initial['closed_covers_verified'],
               'rank_bounds_verified': initial['rank_bounds_verified'] + 80,
               'auditor_replacement_of_discovery': False,
               'application_reference_replayed': True, 'protocol_digest': plan['digest']}
    capsule['summary'] = summary
    capsule['digest'] = canonical_digest(capsule)
    write_json(out/'outcome_capsule.json', capsule)
    write_json(out/'complete_verification.json', summary)
    return summary


def verify_capsule(path: Path):
    c = json.loads(Path(path).read_text())
    assert c['schema'] == 'publication-outcome-capsule-v1'
    assert c['digest'] == canonical_digest({k: v for k, v in c.items() if k != 'digest'})
    for identifier, w in c['witnesses'].items():
        payload = {k: w[k] for k in ('problem_digest', 'tokens', 'native', 'resources')}
        assert canonical_digest(payload) == identifier
        p = problem_from_manifest(c['contracts'][w['problem_digest']])
        assert p.digest == w['problem_digest']
        fresh = certify_phase(p, w['tokens'], dag=True)
        assert fresh['success'] and fresh['native'] == w['native'] and fresh['resources'] == w['resources']
    for row in c['runs']:
        if row['best_witness']:
            assert row['best_witness'] in c['witnesses']
        if row['kind'] != 'constructive':
            assert row['success'] == bool(row['best_witness'])
        if row.get('method') != 'audit_only':
            assert not row.get('audit_witness_used')
    return {'valid': True, 'outcomes': len(c['runs']), 'witnesses': len(c['witnesses'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/publication_v1'))
    parser.add_argument('--capsule-only', action='store_true')
    args = parser.parse_args()
    result = verify_capsule(args.output_dir/'outcome_capsule.json') if args.capsule_only else export_and_verify(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
