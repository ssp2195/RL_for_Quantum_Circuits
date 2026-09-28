"""Five-problem, fresh-training comparison of proof-first complete oracles.

Run: python -m hybrid_qcs.cleanup.consumer_runner --output-dir outputs/consumer
Verify: append --stage verify. No auditor supplies discovery circuits.
The original five problems are development benchmarks, not a fresh population
superiority test. Analytic winners are never counted as learned discoveries.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

from ..resource_search import WorkLimits
from .contract import CleanupProblem, Limits
from .benchmarks import candidate_problems, make_problem
from .training import train_cleanup
from .ir import Protocol, local_hybrid
from .policy import CleanupHierarchy
from .plan import compile_deterministic
from .search import CleanupSearch, optimize_cleanup
from .reference import compile_bank_free
from .consumer_compile import compile_factors, verify_factor_receipt
from .consumer_search import synthesize_oracle, optimize_materialization, DEFAULT_OBJECTIVE
from .phase_protocol import PhaseProtocol, verify_phase
from .verify import primitive_receipts, verify_truth_table_generators

BASE = '84c55c94f4b4c33ca55cbdbb9c7738222d6b32b8'
SEEDS = (11,19,23,31,47)
LEARNED = ('previous_hierarchy', 'fixed_side_hierarchy', 'consumer_hierarchy_menu')
DIRECT = ('analytic_materialization', 'fixed_side_untrained', 'bank_free_monomials',
          'rank_monomials_stream', 'rank_all', 'rank_stream', 'rank_recompute', 'consumer_exact_menu')


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def adapt(p, c, source):
    return PhaseProtocol(p.digest, c.layout.width, c.ops, source, c.primitive)


def run_one(p, method, model=None):
    start = time.perf_counter()
    before = None if model is None else model.digest
    receipt = None
    search = None
    analytic_winner = True
    if method == 'previous_hierarchy':
        result = optimize_cleanup(p, model)
        if not result['protocol']:
            return None, {'valid': False, 'reason': result['status'], 'wall_seconds': time.perf_counter()-start}
        c = adapt(p, Protocol.from_payload(result['protocol']), 'previous_frozen_hierarchy')
        search = {'edges': result['edges'], 'rounds': len(result['rounds']),
                  'profiles': [r['profile'] for r in result['rounds']]}
        analytic_winner = False
    elif method.startswith('fixed_side_'):
        side = 'column' if p.m <= p.r else 'row'
        result = CleanupSearch(p.with_aux_cap(p.workspace_lower_bound), sides=(side,)).run(
            model, scheduler='untrained' if method.endswith('untrained') else 'hierarchy')
        if not result['protocol']:
            return None, {'valid': False, 'reason': result['reason'], 'wall_seconds': time.perf_counter()-start}
        c = adapt(p, Protocol.from_payload(result['protocol']), method)
        search = {'edges': result['edges'], 'rounds': 1, 'profiles': [result['profile']]}
        analytic_winner = False
    elif method == 'analytic_materialization':
        result = optimize_materialization(p)
        c = adapt(p, Protocol.from_payload(result['protocol']), method)
    elif method == 'bank_free_monomials':
        c = adapt(p, compile_bank_free(p), method)
    elif method.startswith('rank_'):
        storage = 'stream' if method=='rank_monomials_stream' else method.removeprefix('rank_')
        c, receipt = compile_factors(p, storage=storage, compress_quadratics=method!='rank_monomials_stream')
    else:
        result = synthesize_oracle(p, model, include_learned=method=='consumer_hierarchy_menu')
        if not result.get('selected'):
            return None, {'valid': False, 'reason': result.get('reason'), 'wall_seconds': time.perf_counter()-start}
        chosen = result['selected']
        c = PhaseProtocol.from_payload(chosen['protocol'])
        receipt = chosen['receipt']
        analytic_winner = not chosen['learned_discovery']
        search = {'selected_method': chosen['method'], 'menu_complete': result['menu_complete'],
                  'pareto_methods': result['pareto_methods'], 'failures': result['failures'],
                  'learning_candidates': sum(x['learned_discovery'] for x in result['candidates'])}
    checked = verify_phase(p, c)
    if not checked['valid']:
        raise AssertionError((p.name, method, checked))
    if receipt is not None and not verify_factor_receipt(p, c, receipt):
        raise AssertionError('factor receipt did not verify')
    # Match delivery requirements: every method delivers an actual persistent
    # hybrid-block witness, not just an instruction count or an algebraic bound.
    c.persistent_tail()
    if model is not None and model.digest != before:
        raise AssertionError('test-time parameter change')
    wall = time.perf_counter()-start
    return c, {'valid': True, 'verification': checked, 'receipt': receipt, 'search': search,
               'wall_seconds': wall, 'timely': wall<=10., 'policy_digest': before,
               'training_finished': model is None or model.frozen,
               'selected_by_algebra_not_learning': analytic_winner,
               'audit_witness_used': False}


def analysis(output):
    output = Path(output)
    rows = [json.loads(line) for line in (output/'runs.jsonl').read_text().splitlines()]
    summary = {'base_commit': BASE, 'total_runs': len(rows), 'by_method': {}, 'problems': [],
               'primary_objective': list(DEFAULT_OBJECTIVE),
               'scope': 'same five development problems; finite candidate constructions, no global optimum or new learning-superiority claim',
               'timing': 'same-machine fresh executions; complete construction plus checking and persistent witness, excluding later serialization; shared primitive caches prepared once',
               'sampling': 'five logical targets; seeds/repetitions are not independent tasks'}
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method']==method]
        summary['by_method'][method] = {'runs':len(group),'correct':sum(r['valid'] for r in group),
             'timely':sum(r.get('timely',False) for r in group),
             'median_ms':1000*statistics.median(r['wall_seconds'] for r in group),
             'mean_ms':1000*statistics.mean(r['wall_seconds'] for r in group),
             'analytic_winners':sum(r.get('selected_by_algebra_not_learning',False) for r in group)}
    keys = ('t_count','peak_aux','cnot','native_gates','native_depth','t_depth','measurement_rounds','worst_case_ticks')
    for p in candidate_problems():
        item = {'name':p.name,'r':p.r,'m':p.m,'methods':{}}
        for method in summary['by_method']:
            rs = [r for r in rows if r['problem_digest']==p.digest and r['method']==method and r['valid']]
            item['methods'][method] = {'correct':len(rs), 'median_ms':1000*statistics.median(r['wall_seconds'] for r in rs) if rs else None,
                'ranges':{k:[min(r['verification']['resources'][k] for r in rs),max(r['verification']['resources'][k] for r in rs)] for k in keys} if rs else {}}
        summary['problems'].append(item)
    write(output/'summary.json', summary)
    return summary


def verify(output, *, check_hashes=True):
    output = Path(output)
    manifest = json.loads((output/'PROTOCOL.json').read_text())
    expected = {(p['name'],m,seed,rep) for p in manifest['problems']
                for rep in range(manifest['repetitions'])
                for m in manifest['learned_methods']+manifest['deterministic_methods']
                for seed in (manifest['seeds'] if m in manifest['learned_methods'] else [None])}
    if check_hashes and (output/'EVIDENCE_SHA256.json').exists():
        for relative, wanted in json.loads((output/'EVIDENCE_SHA256.json').read_text()).items():
            if hashlib.sha256((output/relative).read_bytes()).hexdigest()!=wanted:
                raise AssertionError('archived evidence checksum mismatch: '+relative)
    records = [json.loads(line) for line in (output/'runs.jsonl').read_text().splitlines()]
    actual_keys = [(r['name'],r['method'],r['seed'],r['repetition']) for r in records]
    if len(actual_keys)!=len(set(actual_keys)) or set(actual_keys)!=expected:
        raise AssertionError('missing, duplicated, or unexpected experimental jobs')
    seen = set()
    truth_pairs = 0
    for row in records:
        if not row['valid']:
            continue
        receipt = json.loads((output/'protocols'/f"{row['protocol_digest']}.json").read_text())
        p = CleanupProblem.from_manifest(receipt['problem'])
        c = PhaseProtocol.from_payload(receipt['protocol'])
        checked = verify_phase(p, c)
        if not checked['valid'] or checked != row['verification']:
            raise AssertionError('archived outcome failed independent replay')
        if row['receipt'] is not None and not verify_factor_receipt(p, c, row['receipt']):
            raise AssertionError('archived mathematical receipt failed')
        if c.digest not in seen:
            check = verify_truth_table_generators(p, c)
            if not check['valid']:
                raise AssertionError('independent truth-table check failed')
            truth_pairs += check['input_generator_pairs']
            seen.add(c.digest)
    result = {'valid':True,'records':len(records),'distinct_protocols':len(seen),
              'truth_table_input_outcome_generator_pairs':truth_pairs,
              'scope':'exact all-outcome symbolic replay and independent all-input truth tables; no generator or policy called'}
    write(output/'verification.json', result)
    return result


def run(output, *, stages=(64,96,24), seeds=SEEDS, repeats=3):
    output = Path(output)
    if (output/'runs.jsonl').exists():
        raise ValueError('refusing to overwrite an experiment; choose a fresh output folder')
    output.mkdir(parents=True, exist_ok=True)
    files = list(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).parents[1]/n for n in ('model.py','clifford_lift.py','native_exact.py')]
    hashes = {str(p.relative_to(Path(__file__).parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write(output/'PROTOCOL.json', {'base_commit':BASE,'seeds':list(seeds),'stages':list(stages),
          'repetitions':repeats,'problems':[p.manifest() for p in candidate_problems()],
          'learned_methods':list(LEARNED),'deterministic_methods':list(DIRECT),
          'order':'fixed seed shuffle before evaluation','source_hashes':hashes,
          'objective':list(DEFAULT_OBJECTIVE),'platform':platform.platform(),'python':platform.python_version(),
          'primitive':'same verified four-T AND for every method',
          'limits':'4096 classical allocations / 10 seconds where search is used',
          'development_not_preregistered_confirmation':True,'parameter_selection_on_test':False,
          'analytic_outputs_are_not_learning_successes':True})
    cache_start = time.perf_counter()
    primitive_receipts('and4')
    for k in ('AND','UNAND','Z','CZ','X','H','CX','MINUS'):
        local_hybrid(k)
    write(output/'cache_preparation.json', {'seconds':time.perf_counter()-cache_start,'target_independent':True})
    models = {}
    for seed in seeds:
        model, trajectory = train_cleanup(seed, stages)
        models[seed] = model
        write(output/'models'/f'{seed}.json', model.payload())
        with gzip.open(output/f'training-{seed}.json.gz','wt') as f:
            json.dump(trajectory,f,separators=(',',':'))
        write(output/f'training-summary-{seed}.json',{k:v for k,v in trajectory.items() if k!='episodes'})
    jobs = [(p,m,s,rep) for p in candidate_problems() for rep in range(repeats)
            for m in LEARNED+DIRECT for s in (seeds if m in LEARNED else (None,))]
    random.Random(290928).shuffle(jobs)
    with (output/'runs.jsonl').open('w') as log:
        for index,(p,method,seed,rep) in enumerate(jobs):
            c,row = run_one(p,method,models.get(seed))
            row.update({'index':index,'method':method,'seed':seed,'repetition':rep,
                        'problem_digest':p.digest,'name':p.name,
                        'protocol_digest':None if c is None else c.digest})
            if c is not None:
                path=output/'protocols'/f'{c.digest}.json'
                if not path.exists():
                    write(path,{'problem':p.manifest(),'protocol':c.payload()})
                    path.with_suffix('.qasm').write_text(c.qasm3())
            log.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
            log.flush()
    summary=analysis(output)
    verification=verify(output,check_hashes=False)
    (output/'COMPLETE').write_text('All requested jobs and independent verification completed.\n')
    write(output/'EVIDENCE_SHA256.json',{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(output.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json'})
    return {'summary':summary['by_method'],'verification':verification}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=('run','verify'),default='run')
    parser.add_argument('--smoke',action='store_true',help='one trained seed and repetition; labelled smoke, not the full study')
    args=parser.parse_args()
    result=verify(args.output_dir) if args.stage=='verify' else run(args.output_dir,
             **({'stages':(8,12,4),'seeds':(11,),'repeats':1} if args.smoke else {}))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
