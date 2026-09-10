"""Reproducible native optimality runs, not first-feasible-only qualification."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from .ancilla_contract import AncillaContract
from .model import Budget
from .native_domain import SCHEMA, NativeProblem, digest
from .native_benchmarks import named_benchmarks, prior_phase_benchmarks, contract, swap_matrix
from .native_optimize import (OPT_SCHEMA, DEFAULT_ORDER, optimize_native_resources,
                              verify_native_optimization, OPTIMAL)
from .native_runner import train_native, write
from .resource_search import WorkLimits


def calibration_problems():
    """Specifications only: no reference gate word enters discovery or training."""
    h = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    t = np.diag([1, np.exp(1j*np.pi/4)])
    return (
        NativeProblem('opt-cal-H', contract(1), Budget(1, 0, 3, 3), h),
        NativeProblem('opt-cal-T', contract(1), Budget(1, 0, 3, 3), t),
        NativeProblem('opt-cal-HTH', contract(1), Budget(1, 0, 3, 3), h@t@h),
        NativeProblem('opt-cal-SWAP', contract(2), Budget(0, 3, 3, 3), swap_matrix(2)),
        NativeProblem('opt-cal-H-clean-1', contract(1, 1), Budget(0, 0, 1, 1), h),
    )


def problem_from_manifest(manifest):
    """Deserialize and validate the fixed grammar/contract; do not trust extra fields."""
    c = manifest['contract']
    if c[0] != 'ancilla-contract-v1' or manifest['schema'] != SCHEMA:
        raise ValueError('not a native hybrid manifest')
    roles = AncillaContract(c[1], tuple(c[2]), tuple(c[3]), tuple(c[4]), c[5])
    p = NativeProblem('verified-specification', roles, Budget(**manifest['budget']),
                      np.asarray(manifest['unitary_real']) + 1j*np.asarray(manifest['unitary_imag']),
                      max_t_depth=manifest['max_t_depth'], tolerance=manifest['tolerance'])
    if p.digest != digest(manifest):
        raise ValueError('manifest differs from the supported native grammar')
    return p


def run_study(output, *, targets='all', seeds=(0, 1, 2, 3, 4),
              objectives=DEFAULT_ORDER, limits=WorkLimits(20000, 50000, 3., 3.),
              stages=(64, 96, 24), discovery_edges=512, audit_edges=4096,
              verification_edges=12000, selected_names=()):
    output = Path(output)
    registry = {'all': lambda: named_benchmarks()+prior_phase_benchmarks(),
                'named': named_benchmarks, 'calibration': calibration_problems}
    problems = registry[targets]()
    if selected_names:
        unknown = set(selected_names) - {p.name for p in problems}
        if unknown:
            raise ValueError(f'unknown targets: {sorted(unknown)}')
        problems = tuple(p for p in problems if p.name in selected_names)
    src = Path(__file__).parent
    files = sorted(src.glob('native_*.py')) + [src/'model.py', src/'clifford_lift.py']
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=src, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write(output/'manifest.json', {
        'schema': OPT_SCHEMA, 'commit': commit, 'python': platform.python_version(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'objectives': list(objectives), 'limits_per_target_scheduler': asdict(limits),
        'seeds': list(seeds), 'training_stages': list(stages),
        'phase_quotas': [discovery_edges, audit_edges, verification_edges],
        'targets': [{'name': p.name, 'digest': p.digest, 'problem': p.manifest()} for p in problems],
        'historical_data_reused': False, 'witness_injection': False,
        'standalone_verification': 'separate replay budget, not included in optimization work',
    })
    rows = []
    for seed in seeds:
        model, training = train_native(seed, stages=stages)
        model.save(output/'checkpoints'/f'seed-{seed}.json')
        write(output/'training'/f'seed-{seed}.json', training)
        for p in problems:
            for scheduler in ('hierarchy', 'untrained'):
                r = optimize_native_resources(p, model, objectives=objectives, limits=limits,
                    scheduler=scheduler, discovery_edges=discovery_edges, audit_edges=audit_edges,
                    verification_edges=verification_edges)
                check = (verify_native_optimization(p, r) if r['witness'] is not None or
                         r['status']=='infeasible_under_numerical_contract' else None)
                if r['status'] in OPTIMAL and (not check or not check['optimality_verified']):
                    raise AssertionError('standalone verifier rejected a claimed optimum')
                if r['status']=='infeasible_under_numerical_contract' and (not check or not check['infeasibility_verified']):
                    raise AssertionError('standalone verifier rejected an exclusion')
                rows.append({'name': p.name, 'seed': seed, 'scheduler': scheduler,
                             'result': r, 'standalone_verification': check})
                write(output/'outcomes.json', rows)
    summary = {'schema': OPT_SCHEMA, 'target_count': len(problems), 'outcomes': len(rows),
               'objectives': list(objectives),
               'by_scheduler': {s: {
                   'attempts': sum(r['scheduler']==s for r in rows),
                   'optimal': sum(r['scheduler']==s and r['result']['status'] in OPTIMAL for r in rows),
                   'upper_bound': sum(r['scheduler']==s and r['result']['status']=='upper_bound' for r in rows),
                   'unknown': sum(r['scheduler']==s and r['result']['status']=='unknown' for r in rows),
                   'infeasible': sum(r['scheduler']==s and r['result']['status']=='infeasible_under_numerical_contract' for r in rows),
                   'discovery_witnesses': sum(r['scheduler']==s and r['result']['discovery_incumbent'] is not None for r in rows),
                   'audit_assisted_runs': sum(r['scheduler']==s and r['result']['audit_witness_used'] for r in rows),
               } for s in ('hierarchy','untrained')},
               'scope': 'bounded native tolerance-domain optimization, not unrestricted exact synthesis',
               'claim': 'functional optimization and certification qualification; not learned superiority'}
    write(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/native-optimality')
    parser.add_argument('--targets', choices=('all','named','calibration'), default='all')
    parser.add_argument('--names', nargs='*', default=[])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--objectives', nargs='+', default=list(DEFAULT_ORDER))
    parser.add_argument('--edge-limit', type=int, default=20000)
    parser.add_argument('--record-limit', type=int, default=50000)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--stages', type=int, nargs=3, default=[64,96,24])
    parser.add_argument('--discovery-edges', type=int, default=512)
    parser.add_argument('--audit-edges', type=int, default=4096)
    parser.add_argument('--verification-edges', type=int, default=12000)
    args = parser.parse_args()
    result = run_study(args.output_dir, targets=args.targets, selected_names=args.names,
                      seeds=args.seeds, objectives=args.objectives, stages=args.stages,
                      limits=WorkLimits(args.edge_limit,args.record_limit,args.seconds,args.seconds),
                      discovery_edges=args.discovery_edges, audit_edges=args.audit_edges,
                      verification_edges=args.verification_edges)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
