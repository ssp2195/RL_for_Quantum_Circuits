"""Reproducible five-candidate post-training protocol-generation experiment.

python -m hybrid_qcs.cleanup.runner --output-dir outputs/cleanup-study
python -m hybrid_qcs.cleanup.runner --stage verify --output-dir outputs/cleanup-study

Five fixed held-out specifications, five independently trained frozen policies,
three timing repetitions. No audit or constructive control repairs a learned
search failure. Hardware ticks and classical runtime are reported separately.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import sys
import numpy as np

from ..resource_search import WorkLimits
from .contract import CleanupProblem, BASE_COMMIT, digest
from .benchmarks import candidate_problems, protocol_manifest
from .training import train_cleanup
from .policy import CleanupHierarchy
from .search import CleanupSearch, optimize_cleanup
from .plan import compile_deterministic
from .reference import PhaseReference, compile_bank_free, verify_reference
from .ir import Protocol, local_hybrid
from .verify import verify_protocol, workspace_certificate, verify_workspace_certificate, primitive_receipts


def write_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/name for name in
           ('model.py','native_policy.py','native_search.py','native_domain.py','native_exact.py','native_audit.py','native_optimize.py','clifford_lift.py')]
    return {str(p.relative_to(Path(__file__).parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def save_protocol(output,p,prot):
    checked=verify_reference(p,prot) if isinstance(prot,PhaseReference) else verify_protocol(p,prot)
    if not checked['valid']:raise AssertionError('cannot save invalid protocol')
    path=Path(output)/'protocols'/f'{prot.digest}.json'
    if not path.exists():
        write_json(path,{'problem':p.manifest(),'protocol':prot.payload(),'verification':checked})
        path.with_suffix('.qasm').write_text(prot.qasm3())
    return prot.digest,checked


def run_job(p,method,model,limits):
    before=None if model is None else model.digest
    t0=time.perf_counter();certificate=None;trace=[];profile={};edges=0;records=0
    if method in ('theorem_trained','theorem_untrained'):
        result=optimize_cleanup(p,model,scheduler='untrained' if method.endswith('untrained') else 'hierarchy',limits=limits)
        prot=Protocol.from_payload(result['protocol']) if result['protocol'] else None
        checked=result.get('verification');certificate=result.get('certificate')
        trace=[{'status':r['status'],'edges':r['edges'],'allocations':r['allocations'],'profile':r['profile']} for r in result['rounds']]
        status=result['status'];edges=result['edges'];records=result['records'];profile={'rounds':len(trace)}
    elif method in ('control_helpers_trained','coherent_inverse_trained'):
        side='row' if method=='control_helpers_trained' else ('column' if p.m<=p.r else 'row')
        result=CleanupSearch(p,limits,sides=(side,),mode='coherent' if method=='coherent_inverse_trained' else 'measured').run(model)
        prot=Protocol.from_payload(result['protocol']) if result['protocol'] else None
        checked=result['verification'];status=result['status'];edges=result['edges'];records=result['records'];profile=result['profile']
        trace=[{'allocations':result['allocations']}]
    elif method=='bank_free_anf':
        prot=compile_bank_free(p);checked=verify_reference(p,prot);status='certified_bank_free_control'
        if not checked['valid']:raise AssertionError('bank-free reference failed')
    else:
        side='smaller' if method=='deterministic_smaller_side' else 'row'
        prot=compile_deterministic(p,side=side)
        checked=verify_protocol(p,prot);status='certified_constructive_control'
        if not checked['valid']:raise AssertionError('deterministic control failed verification')
        certificate=workspace_certificate(p,prot)
    wall=time.perf_counter()-t0
    if model is not None and before!=model.digest:raise AssertionError('test-time checkpoint mutation')
    return prot,{'status':status,'timely':bool(prot is not None and wall<=limits.wall_seconds),
                 'wall_seconds':wall,'edges':edges,'records':records,'profile':profile,'trace':trace,
                 'verification':checked,'certificate':certificate,'policy_digest':before,
                 'audit_witness_used':False,'resource_counts_are_from_generated_native_protocol':True}


def analyze(output):
    output=Path(output);rows=[json.loads(s) for s in (output/'runs.jsonl').read_text().splitlines()]
    methods=sorted({r['method'] for r in rows});by_method={}
    for m in methods:
        rs=[r for r in rows if r['method']==m]
        by_method[m]={'runs':len(rs),'correct':sum(r['verification'] is not None and r['verification']['valid'] for r in rs),
                      'timely':sum(r['timely'] for r in rs),'mean_wall_seconds':statistics.mean(r['wall_seconds'] for r in rs),
                      'median_wall_seconds':statistics.median(r['wall_seconds'] for r in rs)}
    candidate_rows=[]
    for p in candidate_problems():
        group=[r for r in rows if r['problem_digest']==p.digest]
        item={'name':p.name,'r':p.r,'m':p.m,'logical_qubits':p.n,
              'phase_anf_degree':max((m.bit_count() for m in p.target_polynomial),default=0),'methods':{}}
        for m in methods:
            rs=[r for r in group if r['method']==m and r['verification'] and r['verification']['valid']]
            item['methods'][m]={'correct':len(rs),
                'resource_ranges':{k:[min(r['verification']['resources'][k] for r in rs),max(r['verification']['resources'][k] for r in rs)] for k in
                                   ('peak_aux','t_count','cnot','native_gates','native_depth','t_depth','worst_case_ticks','measurement_rounds')}
                if rs else {},'median_wall_seconds':statistics.median(r['wall_seconds'] for r in rs) if rs else None}
        a=item['methods']['theorem_trained']['resource_ranges'];b=item['methods']['control_helpers_trained']['resource_ranges']
        item['theorem_vs_control_helpers']={k:{'conservative_improvement':b[k][0]-a[k][1],
                                             'conservative_percent':100*(b[k][0]-a[k][1])/b[k][0] if b[k][0] else 0}
                                             for k in ('peak_aux','t_count','cnot','native_gates')} if a and b else {}
        candidate_rows.append(item)
    summary={'schema':'cleanup-study-analysis-v1','candidate_count':5,'total_runs':len(rows),'by_method':by_method,
             'candidates':candidate_rows,
             'claim':'measured resources of five selected full-bank oracle problems; not unrestricted optimality or learner superiority',
             'sampling':'five problem instances; seeds and repetitions are not independent logical tasks',
             'hardware_ticks':'declared synthetic latency model; no hardware execution',
             'classical_search':'bounded structured macro search, not unrestricted native circuit enumeration',
             'primitive_cache':'shared verified <=3-qubit primitives; no target witness cache',
             'strong_control_has_same_theorem':True}
    write_json(output/'summary.json',summary)
    lines=['# Five-candidate cleanup study','',summary['claim']+'.','',
           '| Candidate | r x m | Aux: theorem / control | T: theorem / control | CNOT: theorem / control | Native gates: theorem / control |',
           '|---|---:|---:|---:|---:|---:|']
    for item in candidate_rows:
        a=item['methods']['theorem_trained']['resource_ranges'];b=item['methods']['control_helpers_trained']['resource_ranges']
        val=lambda k: f"{a[k][0]} / {b[k][0]}" if a and b else 'unresolved'
        lines.append(f"| {item['name']} | {item['r']} x {item['m']} | {val('peak_aux')} | {val('t_count')} | {val('cnot')} | {val('native_gates')} |")
    lines += ['', '| Method | Correct / runs | Timely | Median generation + verification time |', '|---|---:|---:|---:|']
    for m,r in by_method.items():lines.append(f"| {m} | {r['correct']} / {r['runs']} | {r['timely']} | {r['median_wall_seconds']:.6f} s |")
    lines += ['', 'The deterministic smaller-side control receives the SAME construction and may be faster than learned search. '
              'Any common quantum-resource improvement is attributable to the helper/cleanup construction, not training.',
              '', 'All costs include the consumer and the native H/measurement/conditional-X reset implementation. '
              'T=4 per exact clean-target AND; this is an established resource primitive, not a new four-T full Toffoli.',
              '', 'Measurement has an explicit latency; two measurement layers do not mean constant total runtime. '
              'CNOT and gate counts are worst-case execution counts. The archived expected gate counts require the certified uniform transcript.',
              '', 'The workspace theorem is trusted mathematics from the attached PDF. Its hypotheses and the witness are machine-checked; '
              'this is not proof-assistant validation of the converse, and it does not prove global oracle optimality.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary


def verify_evidence(output):
    output=Path(output);lock=json.loads((output/'PROTOCOL.json').read_text())
    if lock!=protocol_manifest():raise ValueError('study protocol changed')
    if (output/'PROTOCOL_SHA256').read_text().strip()!=digest(lock):raise ValueError('protocol lock digest mismatch')
    if json.loads((output/'SOURCE_SHA256.json').read_text())!=source_hashes():raise ValueError('executed source changed')
    if (output/'EVIDENCE_SHA256.json').exists():
        for name,h in json.loads((output/'EVIDENCE_SHA256.json').read_text()).items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=h:raise ValueError('evidence checksum mismatch: '+name)
    models={}
    for seed in lock['seeds']:
        m=CleanupHierarchy.load(output/'checkpoints'/f'seed-{seed}.json')
        if not m.frozen or m.episodes!=sum(lock['training_stages']) or min(m.updates)<1:raise ValueError('incomplete training checkpoint')
        with gzip.open(output/f'training-{seed}.json.gz','rt') as f:log=json.load(f)
        if log['policy_digest']!=m.digest or len(log['episodes'])!=m.episodes:raise ValueError('training provenance mismatch')
        outer=sum(len(e['transitions']) for e in log['episodes'] if e['stage']=='outer')
        inner=[sum(t['action'][0]==family for e in log['episodes'] if e['stage']=='inner' for t in e['transitions']) for family in m.payload()['families']]
        if outer!=m.outer_updates or inner!=m.updates.tolist():raise ValueError('recorded updates disagree with trajectories')
        models[seed]=m
    count=0;branches=0
    for path in sorted((output/'protocols').glob('*.json')):
        obj=json.loads(path.read_text());p=CleanupProblem.from_manifest(obj['problem']);prot=(PhaseReference if obj['protocol']['schema']=='bank-free-phase-reference-v1' else Protocol).from_payload(obj['protocol'])
        if prot.digest!=path.stem:raise ValueError('protocol file content hash mismatch')
        checked=verify_reference(p,prot) if isinstance(prot,PhaseReference) else verify_protocol(p,prot)
        if not checked['valid'] or checked!=obj['verification']:raise ValueError('independent protocol replay differs')
        if path.with_suffix('.qasm').read_text()!=prot.qasm3():raise ValueError('native export differs from witness')
        count+=1;branches+=checked['branch_count']
    rows=[json.loads(s) for s in (output/'runs.jsonl').read_text().splitlines()]
    expected=set()
    for p in candidate_problems():
        for rep in range(lock['test_repetitions']):
            for seed in lock['seeds']:
                for method in ('theorem_trained','control_helpers_trained','coherent_inverse_trained'):expected.add((p.digest,rep,seed,method))
            for method in ('theorem_untrained','deterministic_smaller_side','deterministic_control_side','bank_free_anf'):expected.add((p.digest,rep,None,method))
    actual=[(r['problem_digest'],r['repetition'],r['seed'],r['method']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('incomplete, duplicated or extra experimental jobs')
    for row in rows:
        if row['policy_digest'] is not None and row['policy_digest']!=models[row['seed']].digest:raise ValueError('wrong frozen policy provenance')
        if row['protocol_digest']:
            obj=json.loads((output/'protocols'/f"{row['protocol_digest']}.json").read_text())
            p=CleanupProblem.from_manifest(obj['problem']);pr=(PhaseReference if obj['protocol']['schema']=='bank-free-phase-reference-v1' else Protocol).from_payload(obj['protocol'])
            if row['problem_digest']!=p.digest or row['verification']!=obj['verification']:raise ValueError('run/witness contract mismatch')
            if row['certificate'] is not None and not verify_workspace_certificate(p,pr,row['certificate'])['valid']:
                raise ValueError('invalid workspace certificate')
        elif row['verification'] is not None:raise ValueError('verification without witness')
    result={'valid':True,'rows':len(rows),'unique_protocols':count,'symbolically_covered_transcripts_sum':branches,
            'exact_primitive_receipts':primitive_receipts('and4'),
            'scope':'full symbolic all-input/all-outcome checks; no dense global state and no policy/auditor invocation'}
    write_json(output/'verification.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=('all','verify','report'),default='all')
    args=parser.parse_args(argv);out=args.output_dir
    if args.stage=='verify':print(json.dumps(verify_evidence(out),indent=2));return
    if args.stage=='report':print(json.dumps(analyze(out),indent=2));return
    if out.exists() and any(out.iterdir()):raise ValueError('choose a new output directory; previous evidence is never overwritten')
    out.mkdir(parents=True,exist_ok=True);lock=protocol_manifest()
    write_json(out/'PROTOCOL.json',lock);(out/'PROTOCOL_SHA256').write_text(digest(lock)+'\n')
    write_json(out/'SOURCE_SHA256.json',source_hashes())
    write_json(out/'ENVIRONMENT.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
               'base_commit':BASE_COMMIT,'clock':time.get_clock_info('perf_counter')._asdict() if hasattr(time.get_clock_info('perf_counter'),'_asdict') else str(time.get_clock_info('perf_counter'))})
    start=time.perf_counter();primitive_receipts('and4');primitive_receipts('ccx7')
    for k in ('AND','UNAND','Z','CZ','X','H','MINUS'):local_hybrid(k,'and4')
    write_json(out/'PRIMITIVE_PREPARATION.json',{'seconds':time.perf_counter()-start,'shared_across_all_methods':True})
    models={};training=[]
    for seed in lock['seeds']:
        model,log=train_cleanup(seed,tuple(lock['training_stages']),edge_budget=lock['training_edges'])
        model.save(out/'checkpoints'/f'seed-{seed}.json');models[seed]=model
        with gzip.GzipFile(str(out/f'training-{seed}.json.gz'),'wb',mtime=0) as f:f.write(json.dumps(log,separators=(',',':')).encode())
        training.append({k:v for k,v in log.items() if k!='episodes'})
        print(f'trained {seed}: {model.episodes} episodes, {log["wall_seconds"]:.3f}s',flush=True)
    write_json(out/'TRAINING.json',training);(out/'TRAINING_COMPLETE').write_text('completed staged training, not a convergence claim\n')
    jobs=[]
    for pi,p in enumerate(candidate_problems()):
        for rep in range(lock['test_repetitions']):
            for seed in lock['seeds']:
                for method in ('theorem_trained','control_helpers_trained','coherent_inverse_trained'):
                    jobs.append((pi,rep,seed,method))
            for method in ('theorem_untrained','deterministic_smaller_side','deterministic_control_side','bank_free_anf'):
                jobs.append((pi,rep,None,method))
    # Fixed interleaving avoids timing all methods in separate blocks.
    np.random.default_rng(2026091601).shuffle(jobs)
    ps=candidate_problems();limits=WorkLimits(lock['test_edges'],20000,lock['test_seconds'],lock['test_seconds'])
    with (out/'runs.jsonl').open('w') as stream:
        for pos,(pi,rep,seed,method) in enumerate(jobs):
            p=ps[pi];model=models.get(seed)
            prot,row=run_job(p,method,model,limits)
            ident=None
            if prot is not None:ident,checked=save_protocol(out,p,prot)
            row.update(method=method,seed=seed,repetition=rep,problem_digest=p.digest,problem_name=p.name,protocol_digest=ident)
            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
            if (pos+1)%30==0:print(f'{pos+1}/{len(jobs)} generation jobs',flush=True)
    summary=analyze(out);verified=verify_evidence(out)
    (out/'ALL_COMPLETE').write_text(f'{len(jobs)} generation runs; {verified["unique_protocols"]} independent exact protocol replays\n')
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json'}
    write_json(out/'EVIDENCE_SHA256.json',hashes)
    print(json.dumps({'methods':summary['by_method'],'verification':verified},indent=2))


if __name__=='__main__':main()
