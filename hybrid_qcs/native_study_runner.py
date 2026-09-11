"""Execute the locked native publication campaign; never alter historical data.

Run `python -m hybrid_qcs.native_study_runner --output-dir ...`.
Source and protocol locks precede training; checkpoints precede test execution.
Primary discovery never calls an auditor or reads construction_references.json.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from .model import HybridState,Gate,Budget
from .native_study_corpus import lock,dump,load_problem,STUDY_SCHEMA
from .native_study_search import StudyPolicy,run_anytime,run_discovery,StudySearch
from .native_study_training import select_and_train
from .native_study_analysis import read_rows,analyze
from .native_exact import (ExactMatrix,exact_word,verify_exact_word,verify_exact_optimization,
                           exact_qft,exact_mcx,determinant_obstruction,verify_determinant_certificate,ONE,ZERO,omega_times)
from .native_optimize import optimize_native_resources,verify_native_optimization,OPTIMAL
from .native_domain import NativeProblem,certify_native
from .native_benchmarks import named_benchmarks,reference_gates,contract
from .resource_search import WorkLimits


def source_lock(output):
    root=Path(__file__).resolve().parents[1]
    files=sorted((root/'hybrid_qcs').glob('*.py'))
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=Path(output)/'source_lock.json'
    data={'schema':STUDY_SCHEMA,'source_sha256':hashes,'numpy':np.__version__,
          'python':platform.python_version(),'platform':platform.platform(),
          'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),'provenance':'current source files, not an inferred commit'}
    if path.exists():
        old=json.loads(path.read_text())
        if old['source_sha256']!=hashes:raise ValueError('source changed after lock; use a new output directory')
    else:dump(path,data)
    return data


def append_row(path,row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush()


def run_primary(protocol,output):
    output=Path(output)
    if not (output/'TRAINING_COMPLETE').exists():raise ValueError('training must finish before test execution')
    model_files=sorted((output/'checkpoints'/'full').glob('*.json'))
    selected={int(p.stem.split('-')[-1]):StudyPolicy.load(p) for p in model_files}
    jobs=[]
    for row in protocol['splits']['test']:
        for axis,budgets in protocol['primary_budget_axes'].items():
            for budget in budgets:
                repeats=protocol['timing_repetitions'] if axis=='wall_seconds' else protocol['edge_repetitions']
                for repeat in range(repeats):
                    for method in protocol['methods']:
                        seeds=protocol['primary_seeds'] if method in ('hierarchy','outer','inner') else protocol['deterministic_controls_seeds']
                        for seed in seeds:jobs.append((row,axis,budget,repeat,method,seed))
    order=np.random.default_rng(4551).permutation(len(jobs))
    existing=read_rows(output/'primary.jsonl')
    seen={r['key'] for r in existing}
    for number,index in enumerate(order):
        row,axis,budget,repeat,method,seed=jobs[int(index)]
        key=f"{row['name']}:{axis}:{budget}:{repeat}:{method}:{seed}"
        if key in seen:continue
        r=run_anytime(load_problem(row),selected[seed],method,axis=axis,budget=budget)
        append_row(output/'primary.jsonl',{'key':key,'name':row['name'],'orbit':row['orbit'],'seed':seed,
                   'method':method,'axis':axis,'budget':budget,'repeat':repeat,'result':r})
        if (number+1)%100==0:print(f'primary {number+1}/{len(jobs)}',flush=True)
    rows=read_rows(output/'primary.jsonl')
    if len(rows)!=len(jobs) or len({r['key'] for r in rows})!=len(jobs):raise AssertionError('incomplete/duplicate primary outcomes')
    dump(output/'primary_completion.json',{'expected_runs':len(jobs),'completed_runs':len(rows),'all_failures_preserved':True})
    (output/'PRIMARY_COMPLETE').write_text('complete\n')


def run_ablations(protocol,output):
    output=Path(output);path=output/'ablations.jsonl'
    seen={r['key'] for r in read_rows(path)}
    for variant in protocol['exploratory_ablations']:
        for seed in protocol['ablation_seeds']:
            m=StudyPolicy.load(output/'checkpoints'/variant/f'seed-{seed}.json')
            for row in protocol['splits']['test']:
                key=f'{variant}:{seed}:{row["name"]}'
                if key in seen:continue
                r=run_anytime(load_problem(row),m,'hierarchy',axis='edges',budget=1024)
                append_row(path,{'key':key,'name':row['name'],'variant':variant,'seed':seed,'result':r})
    expected=len(protocol['exploratory_ablations'])*len(protocol['ablation_seeds'])*len(protocol['splits']['test'])
    if len(read_rows(path))!=expected:raise AssertionError('incomplete ablations')


def _cert_reference(p,word):
    s=HybridState.identity(p.width,p.budget)
    for g in word:
        s=s.apply(g,partial_order_reduction=False)
        if s is None:raise ValueError('reference violates caps')
    return certify_native(p,s,provenance='constructive_reference_not_RL')


def run_reachability(output):
    """Exact domain classification plus separate known one-clean-wire witnesses."""
    rows=[]
    for kind,n in (('QFT',3),('Toffoli',4)):
        proof=determinant_obstruction(kind,n)
        if not verify_determinant_certificate(proof):raise AssertionError('analytic exclusion failed')
        p=next(p for p in named_benchmarks() if p.family==kind and len(p.contract.logical_qubits)==n
               and len(p.contract.clean_ancillas)==1)
        word=reference_gates(p);reference=_cert_reference(p,word)
        exact=exact_qft(n) if kind=='QFT' else exact_mcx(n)
        check=verify_exact_word(p.contract,exact,reference['native'])
        if not reference['success'] or not check['valid']:raise AssertionError('one-ancilla reference failed exact replay')
        rows.append({'name':p.name,'zero_ancilla_exclusion':proof,'one_ancilla_reference':reference,
                     'exact_reference_check':check,'minimum_clean_ancillas':1,'discovered_by_learning':False,
                     'scope':'known determinant lower bound and independently checked construction, not a novel circuit identity'})
    dump(Path(output)/'reachability.json',rows)
    return rows


def run_challenges(protocol,output):
    path=Path(output)/'challenges.jsonl';seen={r['key'] for r in read_rows(path)}
    m=StudyPolicy.load(Path(output)/'checkpoints/full/seed-11.json')
    for row in protocol['retained']:
        p=load_problem(row)
        for method in ('hierarchy','untrained','mitm'):
            key=f'{row["name"]}:{method}'
            if key in seen:continue
            r=run_anytime(p,m,method,axis='wall_seconds',budget=protocol['secondary_seconds'])
            append_row(path,{'key':key,'name':row['name'],'family':row['family'],'method':method,'result':r})
    if len(read_rows(path))!=len(protocol['retained'])*3:raise AssertionError('incomplete retained challenge campaign')


def run_proofs(protocol,output):
    """Fixed held-out sample; no proof outcome selects or retrains the policy."""
    path=Path(output)/'proofs.jsonl';seen={r['name'] for r in read_rows(path)}
    m=StudyPolicy.load(Path(output)/'checkpoints/full/seed-11.json')
    chosen=protocol['splits']['test'][:8]+[r for r in protocol['splits']['test'] if r['n']==2][:4]
    for row in chosen:
        if row['name'] in seen:continue
        p=load_problem(row)
        r=optimize_native_resources(p,m,objectives=('gates','t_count','cnot','depth'),
              limits=WorkLimits(50000,100000,15.,15.),discovery_edges=2048,audit_edges=10000,verification_edges=30000,
              allow_audit_witness=False)
        numerical=verify_native_optimization(p,r) if r['witness'] else None
        exact=verify_exact_optimization(p,ExactMatrix.from_payload(row['exact_target']),r) if r['status'] in OPTIMAL else None
        if exact is not None and not exact['valid']:raise AssertionError('claimed exact upgrade failed')
        append_row(path,{'name':row['name'],'result':r,'numerical_check':numerical,'exact_check':exact})


def run_workspace(protocol,output):
    """Crossed native ancilla/T-depth study. No reference trajectory enters search."""
    from .qft_guided import controlled_s_native
    path=Path(output)/'workspace.jsonl';seen={r['key'] for r in read_rows(path)}
    # Predetermined held-out one-qubit and entangling targets plus CS calibration.
    chosen=[protocol['splits']['test'][i] for i in (0,1,8,9,10,11)]
    cs=ExactMatrix(tuple(tuple(omega_times(ONE,2) if i==j==3 else ONE if i==j else ZERO for j in range(4)) for i in range(4)))
    for seed in protocol['ablation_seeds']:
        m=StudyPolicy.load(Path(output)/'checkpoints/full'/f'seed-{seed}.json')
        for row in chosen:
            for a in protocol['ancilla_budgets']:
                for method in ('hierarchy','untrained'):
                    key=f'{row["name"]}:{a}:{seed}:{method}'
                    if key in seen:continue
                    p=load_problem(row,a)
                    r=run_anytime(p,m,method,axis='edges',budget=1024)
                    append_row(path,{'key':key,'name':row['name'],'seed':seed,'ancillas':a,'method':method,'result':r})
    # Search the known CS trade-off without phase obligations; failures retained.
    m=StudyPolicy.load(Path(output)/'checkpoints/full/seed-11.json')
    calibration=[]
    for a in (0,1,2):
        for td in (1,2):
            p=NativeProblem(f'CS-a{a}-td{td}',contract(2,a),Budget(3,4,7,7),cs.numerical(),max_t_depth=td)
            for method in ('hierarchy','untrained'):
                r=run_anytime(p,m,method,axis='wall_seconds',budget=5.)
                calibration.append({'name':p.name,'problem':p.manifest(),'ancillas':a,'t_depth_cap':td,'method':method,'result':r})
    # Known references, checked separately after all discovery.
    references=[]
    for a in (0,1):
        if a==0:word=controlled_s_native(0,1)
        else:
            word=(Gate('CNOT',(0,2)),Gate('CNOT',(1,2)),Gate('T',(0,)),Gate('T',(1,)),
                  Gate('TDG',(2,)),Gate('CNOT',(1,2)),Gate('CNOT',(0,2)))
        p=NativeProblem(f'CS-reference-a{a}',contract(2,a),Budget(3,4,7,7),cs.numerical(),max_t_depth=2)
        cert=_cert_reference(p,word);exact=verify_exact_word(p.contract,cs,cert['native'])
        if not cert['success'] or not exact['valid']:raise AssertionError('CS reference not exact')
        references.append({'problem':p.manifest(),'ancillas':a,'certificate':cert,'exact_check':exact,'learned':False})
    dump(Path(output)/'workspace_calibration.json',{'exact_target':cs.payload(),'discovery':calibration,'constructive_references':references,
          'scope':'native discovery and independent known T-depth constructions, not a new identity'})


def run_application(protocol,output):
    """Coherent amplitude amplification with provenance-safe native oracle choice."""
    from .qft_guided import controlled_s_native
    # Two threshold neurons detecting x0=1 and x1=1, followed by a threshold-2
    # output neuron: f(x0,x1)=x0 AND x1. A compact BNN instance, not a large BNN.
    f=[0,0,0,1]
    exact=ExactMatrix(tuple(tuple(omega_times(ONE,4*f[i]) if i==j else ZERO for j in range(4)) for i in range(4)))
    p=NativeProblem('BNN-two-input-conjunction-phase',contract(2),Budget(3,4,8,8),exact.numerical(),'BNN')
    m=StudyPolicy.load(Path(output)/'checkpoints/full/seed-11.json')
    outcomes=[]
    for method in ('hierarchy','untrained','mitm'):
        r=run_anytime(p,m,method,axis='wall_seconds',budget=10.)
        if r['witness']:
            checked=verify_exact_word(p.contract,exact,r['witness']['native'])
            if not checked['valid']:raise AssertionError('BNN oracle exact failure')
            from .certify import unitary_from_gates
            gates=[Gate(g,tuple(qs)) for g,qs in r['witness']['native']]
            u=unitary_from_gates(2,gates);s=np.ones(4,complex)/2
            diffusion=2*np.outer(s,s.conj())-np.eye(4)
            final=diffusion@u@s
            r['amplitude_amplification']={'marked_probability':float(abs(final[3])**2),'ideal_marked_probability':1.,
                                         'oracle_exact_check':checked,'diffusion_cost_included':False}
        outcomes.append({'method':method,'result':r})
    dump(Path(output)/'application.json',{'exact_target':exact.payload(),'predicate':f,'network':'two identity threshold neurons, followed by threshold-2 output',
        'scope':'small coherent oracle integration; original three-input BNN remains in unchanged challenges',
        'problem':p.manifest(),'outcomes':outcomes,'construction_witness_supplied_to_search':False})


def verify_evidence(output):
    """Independently replay final, intermediate and late circuit witnesses.

    Exact reference witnesses and determinant arguments remain separate from
    native discovery. Reused semantic checks never bypass stored resource or
    current-contract bounds. The protocol is the external specification source.
    """
    from .native_optimality_runner import problem_from_manifest
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text())
    byname={r['name']:r for r in protocol['splits']['test']}
    byname.update({r['name']:r for r in protocol['retained']})
    unique={};discovery=set();reference=set()

    def check_word(p,exact,w,kind='discovery'):
        key=json.dumps([exact.payload(),p.manifest(),w['native']],sort_keys=True,separators=(',',':'))
        if key not in unique:
            check=verify_exact_word(p.contract,exact,w['native'])
            if not check['valid']:raise AssertionError(f'inexact archived circuit: {p.name}')
            state=HybridState.identity(p.width,p.budget)
            for g,qs in w['native']:
                state=state.apply(Gate(g,tuple(qs)),partial_order_reduction=False)
                if state is None:raise AssertionError('resource violation')
            cert=certify_native(p,state)
            if not cert['success']:raise AssertionError('native replay rejected circuit')
            unique[key]=cert['resources']
        if unique[key]!=w['resources']:raise AssertionError('stored circuit resources differ from replay')
        (reference if kind=='reference' else discovery).add(key)

    def all_words(p,exact,result):
        witnesses=([result['witness']] if result.get('witness') else [])
        witnesses += [t for t in result.get('incumbent_trace',[]) if 'native' in t]
        witnesses += [r['late_certificate'] for r in result.get('rounds',[]) if r.get('late_certificate')]
        for w in witnesses:check_word(p,exact,w)

    for filename in ('primary.jsonl','ablations.jsonl','workspace.jsonl','challenges.jsonl'):
        for row in read_rows(output/filename):
            spec=byname[row['name']];p=load_problem(spec,row.get('ancillas',0))
            all_words(p,ExactMatrix.from_payload(spec['exact_target']),row['result'])
    proofchecks=[]
    for row in read_rows(output/'proofs.jsonl'):
        spec=byname[row['name']];p=load_problem(spec);exact=ExactMatrix.from_payload(spec['exact_target'])
        all_words(p,exact,row['result'])
        if row['exact_check'] is not None:
            check=verify_exact_optimization(p,exact,row['result'])
            if not check['valid']:raise AssertionError('exact optimization receipt rejected')
            proofchecks.append({'name':row['name'],'valid':True})
    reachability=json.loads((output/'reachability.json').read_text())
    for row in reachability:
        if not verify_determinant_certificate(row['zero_ancilla_exclusion']):raise AssertionError('determinant certificate rejected')
        p=next(p for p in named_benchmarks() if p.name==row['name'])
        exact=exact_qft(3) if p.family=='QFT' else exact_mcx(4)
        check_word(p,exact,row['one_ancilla_reference'],'reference')
    workspace=json.loads((output/'workspace_calibration.json').read_text())
    cs=ExactMatrix(tuple(tuple(omega_times(ONE,2) if i==j==3 else ONE if i==j else ZERO for j in range(4)) for i in range(4)))
    if ExactMatrix.from_payload(workspace['exact_target'])!=cs:raise AssertionError('CS target changed')
    for row in workspace['discovery']:all_words(problem_from_manifest(row['problem']),cs,row['result'])
    for row in workspace['constructive_references']:check_word(problem_from_manifest(row['problem']),cs,row['certificate'],'reference')
    application=json.loads((output/'application.json').read_text());p=problem_from_manifest(application['problem'])
    exact=ExactMatrix(tuple(tuple(omega_times(ONE,4) if i==j==3 else ONE if i==j else ZERO for j in range(4)) for i in range(4)))
    if ExactMatrix.from_payload(application['exact_target'])!=exact:raise AssertionError('application target changed')
    for row in application['outcomes']:
        all_words(p,exact,row['result'])
        if row['result'].get('witness'):
            from .certify import unitary_from_gates
            word=[Gate(g,tuple(qs)) for g,qs in row['result']['witness']['native']]
            uniform=np.ones(4,complex)/2
            final=(2*np.outer(uniform,uniform.conj())-np.eye(4))@unitary_from_gates(2,word)@uniform
            actual=float(abs(final[3])**2)
            saved=row['result']['amplitude_amplification']['marked_probability']
            if abs(actual-saved)>1e-12 or abs(actual-1)>1e-9:raise AssertionError('coherent application mismatch')
    # Construction words are checked ONLY in this post-campaign verification.
    refs=json.loads((output/'construction_references.json').read_text())['words'];construction_checks=0
    for split in protocol['splits'].values():
        for row in split:
            exact=ExactMatrix.from_payload(row['exact_target'])
            if exact_word(row['n'],refs[row['name']])!=exact:raise AssertionError('constructor/exact target mismatch')
            construction_checks+=1
    result={'schema':STUDY_SCHEMA,'unique_exact_discovery_contracts':len(discovery),
            'unique_exact_reference_contracts':len(reference),'exact_optima_rechecked':proofchecks,
            'determinant_exclusions':len(reachability),'construction_specifications_checked':construction_checks,
            'includes_intermediate_and_late_witnesses':True,'counts_are_not_independent_test_targets':True}
    dump(output/'verification.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='experiments/native_publication_v1')
    parser.add_argument('--stage',choices=('all','lock','train','primary','secondary','verify','analyze'),default='all')
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    protocol=lock(out);source_lock(out)
    if hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()!=(out/'PROTOCOL_SHA256').read_text().strip():
        raise ValueError('protocol changed after lock')
    if args.stage=='lock':return
    if args.stage in ('all','train'):
        print('native training and validation selection',flush=True);select_and_train(protocol,out)
    if args.stage in ('all','primary'):
        print('native held-out primary campaign',flush=True);run_primary(protocol,out)
    if args.stage in ('all','secondary'):
        print('native secondary campaigns',flush=True)
        run_ablations(protocol,out);run_reachability(out);run_proofs(protocol,out)
        run_workspace(protocol,out);run_challenges(protocol,out);run_application(protocol,out)
        from .native_study_diagnostics import diagnose,ranking_profile
        diagnose(protocol,out);ranking_profile(protocol,out)
    if args.stage in ('all','verify'):print(json.dumps(verify_evidence(out)),flush=True)
    if args.stage in ('all','analyze'):print(json.dumps(analyze(out)),flush=True)
    if args.stage=='all':(out/'ALL_CAMPAIGNS_COMPLETE').write_text('complete\n')


if __name__=='__main__':main()
