"""Reproduce training, attributable discovery, controls and proof checks.

Examples:
    python -m hybrid_qcs.publication_runner lock --output-dir outputs/publication
    python -m hybrid_qcs.publication_runner train --output-dir outputs/publication
    python -m hybrid_qcs.publication_runner benchmark --output-dir outputs/publication --seed 0 --repeat 0
    python -m hybrid_qcs.publication_runner supplement --output-dir outputs/publication

A lock must precede the held-out campaign. Test outcomes never select a model.
A failed discovery stays failed even if the independent audit finds a circuit.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
from .resource_domain import canonical_digest
from .resource_search import WorkLimits
from .phase_contract import PhaseProblem, problem_from_manifest, certify_phase
from .phase_policy import PhaseHierarchy
from .phase_search import search_phase, constructive_phase
from .phase_baselines import rank_partition_baseline, rank_tdepth_certificate, verify_rank_bound
from .phase_audit import audit_phase, verify_phase_cover
from .publication_corpus import corpus, split_manifest, bnn_problem, bnn_truth, bnn_specification
from .publication_train import train_phase_hierarchy
from .phase_graysynth import graysynth_baseline
from .publication_pipeline import anytime_discovery, public_parity_cnot_bound

PRIMARY = ('hierarchy','untrained','greedy','outer','inner','uniform_cost','audit_only')
ENGINE = ('phase_contract.py','phase_policy.py','phase_index.py','phase_search.py',
          'phase_audit.py','phase_baselines.py','phase_graysynth.py','publication_corpus.py',
          'publication_train.py','publication_pipeline.py')


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def engine_hashes():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ENGINE}


def environment():
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit=None
    return {'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
            'processor':platform.processor(),'source_commit_at_execution':commit,
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
            'cpu_clock':'time.process_time','wall_clock':'time.perf_counter',
            'memory_units':'ru_maxrss KiB on Linux; process high-water, not per-search allocation',
            'deadline':'cooperative operation boundaries; includes certification overrun'}


def lock_protocol(out):
    out=Path(out)
    plan={'schema':'linear-certified-publication-protocol-v1',
          'locked_before_test':True,'seeds':[0,1,2,3,4],'timing_repeats':3,
          'training_episodes':[64,96,24],'rate':.01,
          'rate_selection':'validation-only seed 0: {0.001,0.003,0.01,0.03}; maximize mean normalized CNOT savings, then success count',
          'training_limits':asdict(WorkLimits(512,10000,20.,20.)),
          'discovery_limits':asdict(WorkLimits(2048,20000,.25,.25)),
          'primary_methods':list(PRIMARY),'panel_size':32,'fairness_period':32,
          'primary_endpoint':'mean normalized CNOT savings vs specification-derived parity-star count; failed discovery contributes zero savings',
          'co_primary':'certified discovery success fraction; failures retained',
          'confidence_interval':'paired target-cluster bootstrap 10000 resamples; average seeds and repeats within each target first',
          'secondary':['retrained no_budget','retrained no_workspace','full-frontier scoring with identical frozen weights',
                       'rank-tight T-depth contracts on first eight test targets','controlled-S ancilla calibration',
                       'BNN exactly-one phase oracle and one amplitude-amplification iteration'],
          'audit':'separate baseline and post-campaign proof job; never repairs a learned outcome',
          'scope':'fixed phase polynomial CNOT/required-phase synthesis; T count fixed, optimize CNOT under width/depth/gate constraints',
          'split':split_manifest(),'engine_sha256':engine_hashes()}
    plan['digest']=canonical_digest(plan)
    path=out/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=plan:
            raise RuntimeError('refusing to modify an existing held-out protocol')
    else:
        write_json(path,plan)
        write_json(out/'lock_event.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_digest':plan['digest'],'environment':environment()})
    return plan


def read_lock(out):
    path=Path(out)/'protocol.json'
    if not path.exists():
        raise RuntimeError('create protocol lock before training/evaluation')
    plan=json.loads(path.read_text())
    if plan['digest']!=canonical_digest({k:v for k,v in plan.items() if k!='digest'}):
        raise ValueError('protocol hash mismatch')
    if plan['engine_sha256']!=engine_hashes():
        raise ValueError('engine changed after protocol lock')
    return plan


def train_models(out, seeds=None, ablation=None):
    out=Path(out);plan=read_lock(out)
    seeds=plan['seeds'] if seeds is None else seeds
    results=[]
    for seed in seeds:
        stem=f'{ablation or "full"}-seed-{seed}'
        model_path=out/'checkpoints'/f'{stem}.json'
        report_path=out/'training'/f'{stem}.json'
        if model_path.exists() and report_path.exists():
            m=PhaseHierarchy.load(model_path);r=json.loads(report_path.read_text())
            if r['protocol_digest']!=plan['digest'] or r['checkpoint_digest']!=m.digest:
                raise ValueError('mismatched saved training/checkpoint')
        else:
            m,r=train_phase_hierarchy(seed,rate=plan['rate'],episodes=tuple(plan['training_episodes']),
                                     limits=WorkLimits(**plan['training_limits']),ablation=ablation)
            if not r['all_arms_trained']:
                raise AssertionError('curriculum failed to train every continuation family')
            m.save(model_path)
            r['protocol_digest']=plan['digest'];r['ablation']=ablation
            write_json(report_path,r)
        results.append(r)
        print('trained',stem,'successes',r['stage_successes'],'arms',r['bandit_arm_updates'],flush=True)
    return results


def test_contract(p):
    return replace(p,max_cnot=max(1,min(p.max_cnot,public_parity_cnot_bound(p))))


def _audit_baseline(p,limits):
    r=audit_phase(p,limits=limits)
    w=r['witness']
    best=None if w is None else {'resources':w['resources'],'witness':w,'source':'auditor',
                                 'wall_seconds':r['wall_seconds'],'edges':r['edges']}
    return {'schema':'audit-only-baseline-v1','status':'feasible' if w else r['status'],
            'best':best,'incumbents':[] if best is None else [best],
            'edges':r['edges'],'lookahead_transitions':r['edges'],
            'wall_seconds':r['wall_seconds'],'cpu_seconds':r['cpu_seconds'],
            'first_correct_seconds':r['wall_seconds'] if w else None,'improvements':0,
            'audit_calls':1,'audit_witness_used':bool(w),'rounds':[],
            'proof_generated':r['proof'] is not None,'optimality':'not asserted by baseline timing'}


def benchmark_batch(out, seed, repeat, secondary=False):
    out=Path(out);plan=read_lock(out)
    if seed not in plan['seeds'] or not 0<=repeat<plan['timing_repeats']:
        raise ValueError('seed/repetition outside locked protocol')
    m=PhaseHierarchy.load(out/'checkpoints'/f'full-seed-{seed}.json')
    snapshot=m.digest
    methods=('no_budget','no_workspace','full_panel') if secondary else PRIMARY
    untrained=PhaseHierarchy(seed=seed).freeze()
    models={name:PhaseHierarchy.load(out/'checkpoints'/f'{name}-seed-{seed}.json')
            for name in methods if name in ('no_budget','no_workspace')}
    jobs=[(i,p,method) for i,p in enumerate(corpus()[2]) for method in methods]
    rng=np.random.default_rng(887+seed*19+repeat)
    rng.shuffle(jobs)
    path=out/'raw'/f'{"secondary" if secondary else "primary"}-seed-{seed}-repeat-{repeat}.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['protocol_digest']!=plan['digest']:
                raise ValueError('raw record belongs to another protocol')
            completed.add((row['target_index'],row['method']))
    limits=WorkLimits(**plan['discovery_limits'])
    with path.open('a') as stream:
        for i,p,method in jobs:
            if (i,method) in completed:
                continue
            p=test_contract(p)
            if method=='audit_only':
                r=_audit_baseline(p,limits)
            else:
                selected=untrained if method=='untrained' else models.get(method,m)
                scheduler='hierarchy' if method in ('full_panel','untrained') else method
                r=anytime_discovery(p,selected,scheduler=scheduler,limits=limits,
                                    panel_size=0 if method=='full_panel' else plan['panel_size'])
                if method=='untrained':
                    r['scheduler']='untrained'
                    for step in r['incumbents']:
                        step['source']='untrained'
                    for run in r['rounds']:
                        run['scheduler']='untrained'
                        if run['witness']:
                            run['witness_source']='untrained'
            success=r['best'] is not None
            row={'protocol_digest':plan['digest'],'seed':seed,'repeat':repeat,
                 'target_index':i,'target_name':p.name,'problem':p.manifest(),
                 'method':method,'success':success,
                 'normalized_savings':1-r['best']['resources']['cnot']/p.max_cnot if success else 0.,
                 'cnot':r['best']['resources']['cnot'] if success else None,
                 'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'result':r}
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
    if m.digest!=snapshot:
        raise AssertionError('test generation mutated the checkpoint')
    print('finished',path.name,len(jobs),'jobs',flush=True)


def constructive_controls(out):
    out=Path(out);plan=read_lock(out)
    results=[]
    for repeat in range(plan['timing_repeats']):
        for i,p in enumerate(corpus()[2]):
            p=test_contract(p)
            for method in ('parity_star','rank_partition','graysynth'):
                started,cpu=time.perf_counter(),time.process_time()
                cert={'parity_star':constructive_phase,'rank_partition':rank_partition_baseline,'graysynth':graysynth_baseline}[method](p)
                success=cert['success'] and cert.get('within_original_contract',True)
                results.append({'target_index':i,'target_name':p.name,'repeat':repeat,'method':method,
                                'success':success,'cnot':cert['resources']['cnot'] if success else None,
                                'normalized_savings':1-cert['resources']['cnot']/p.max_cnot if success else 0,
                                'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
                                'certificate':cert,'protocol_digest':plan['digest']})
    write_json(out/'constructive_controls.json',results)


def calibration(out):
    out=Path(out);plan=read_lock(out)
    results=[]
    for seed in plan['seeds']:
        m=PhaseHierarchy.load(out/'checkpoints'/f'full-seed-{seed}.json')
        for a,td in ((0,2),(1,1),(2,1)):
            p=PhaseProblem(2,((1,1),(2,1),(3,7)),a,6,20,32,
                           name='controlled-S-calibration',max_t_depth=td)
            r=anytime_discovery(p,m,limits=WorkLimits(8192,30000,3.,3.))
            bound=rank_tdepth_certificate(p)
            checked=verify_rank_bound(bound,p)
            proof=None;verified=None
            if r['best']:
                lower=p.cap('cnot',r['best']['resources']['cnot']-1)
                audited=audit_phase(lower,limits=WorkLimits(200000,60000,3.,3.))
                if audited['proof'] is not None:
                    proof=audited['proof'];verified=verify_phase_cover(proof,lower)
            row={'seed':seed,'ancillas':a,'t_depth_cap':td,'discovery':r,
                 'rank_bound':bound,'rank_bound_verified':checked,
                 'cnot_exclusion':proof,'cnot_exclusion_verified':verified,
                 'calibration_not_heldout':True}
            results.append(row)
    write_json(out/'ancilla_calibration.json',results)


def boundary_evaluation(out):
    out=Path(out);plan=read_lock(out);rows=[]
    for i,p in enumerate(corpus()[2][:8]):
        rank_proof=rank_tdepth_certificate(p)
        p=replace(p,max_t_depth=rank_proof['lower_t_depth'],
                  max_cnot=6*p.width*p.width*max(1,len(p.coefficients)),
                  max_gates=2048,max_depth=2048)
        bound=rank_tdepth_certificate(p)
        control=rank_partition_baseline(p)
        for seed in plan['seeds']:
            m=PhaseHierarchy.load(out/'checkpoints'/f'full-seed-{seed}.json')
            untrained=PhaseHierarchy(seed=seed).freeze()
            for method in ('hierarchy','untrained'):
                selected=m if method=='hierarchy' else untrained
                r=search_phase(p,selected,scheduler='hierarchy',limits=WorkLimits(**plan['discovery_limits']))
                r['scheduler']=method
                r.pop('expanded_tokens',None)
                rows.append({'target_index':i,'seed':seed,'method':method,'result':r,
                             'rank_bound':bound,'rank_bound_verified':verify_rank_bound(bound,p),
                             'nonlearned_control':control})
    write_json(out/'boundary_evaluation.json',rows)


def application(out):
    out=Path(out);plan=read_lock(out)
    m=PhaseHierarchy.load(out/'checkpoints'/'full-seed-0.json')
    p=bnn_problem()
    r=anytime_discovery(p,m,limits=WorkLimits(20000,60000,5.,5.))
    result={'network':bnn_specification(),'truth_table':bnn_truth(),'discovery':r,
            'seed_selected_before_test':0,'success':r['best'] is not None,
            'application':'small binary threshold-network verification predicate, not a full BNN benchmark'}
    if r['best']:
        w=r['best']['witness'];width=p.width;size=1<<width
        psi=np.zeros(size,dtype=complex);psi[:1<<p.n]=1/np.sqrt(1<<p.n)
        # Apply the actual synthesized native gates, not a desired oracle matrix.
        indices=np.arange(size)
        for g,qs in w['native']:
            if g=='CNOT':
                c,t=qs;psi=psi[indices^(((indices>>c)&1)<<t)]
            elif g=='H':
                q=qs[0];lo=indices[(indices&(1<<q))==0];hi=lo|(1<<q)
                u,v=psi[lo].copy(),psi[hi].copy();psi[lo]=(u+v)/np.sqrt(2);psi[hi]=(u-v)/np.sqrt(2)
            else:
                turn={'S':2,'SDG':-2,'T':1,'TDG':-1}[g]
                psi[(indices&(1<<qs[0]))!=0]*=np.exp(1j*np.pi*turn/4)
        # Deterministic data-register diffusion, evaluated after clean restoration.
        block=psi[:1<<p.n];psi[:1<<p.n]=2*np.mean(block)-block
        marked=[x for x,v in enumerate(bnn_truth()) if v]
        probability=float(np.sum(np.abs(psi[marked])**2))
        expected=float(np.sin(3*np.arcsin(np.sqrt(len(marked)/(1<<p.n)))))**2
        result.update({'marked_inputs_lsb_indices':marked,'initial_marked_probability':len(marked)/(1<<p.n),
                       'one_iteration_marked_probability':probability,'expected_probability':expected,
                       'amplification_error':abs(probability-expected),
                       'workspace_leakage_after_iteration':float(np.sum(np.abs(psi[1<<p.n:])**2)),
                       'oracle_resources':w['resources'],'diffusion_resources_included_in_oracle_counts':False})
        if abs(probability-expected)>1e-9:
            raise AssertionError('generated BNN oracle failed amplitude-amplification test')
    from .publication_application import evaluator_phase_reference
    result['nonlearned_compute_phase_uncompute_reference']=evaluator_phase_reference()
    result['comparison_is_representation_not_learning_gain']=True
    write_json(out/'bnn_application.json',result)


def post_audit(out):
    out=Path(out);plan=read_lock(out)
    raw=[json.loads(line) for path in sorted((out/'raw').glob('primary*.jsonl')) for line in path.read_text().splitlines()]
    results=[]
    for i,p in enumerate(corpus()[2]):
        p=test_contract(p)
        successful=[r for r in raw if r['target_index']==i and r['success']]
        best=min((r['cnot'] for r in successful),default=None)
        rank_proof=rank_tdepth_certificate(p)
        entry={'target_index':i,'problem':p.manifest(),'best_cnot_from_completed_campaign':best,
               'rank_bound':rank_proof,'rank_bound_verified':verify_rank_bound(rank_proof,p),
               'lower_cnot':0,'upper_cnot':best,'status':'upper_bound' if best is not None else 'unknown'}
        if best is not None and best>0:
            lower=p.cap('cnot',best-1)
            result=audit_phase(lower,limits=WorkLimits(100000,40000,1.,1.))
            proof=result.pop('proof',None)
            entry['audit']=result
            if proof is not None:
                verified=verify_phase_cover(proof,lower,limits=WorkLimits(300000,80000,3.,3.))
                entry['verification']=verified
                write_json(out/'proofs'/f'target-{i:02d}.json',proof)
                entry['proof_file']=f'proofs/target-{i:02d}.json'
                if verified['valid']:
                    entry.update({'lower_cnot':best,'status':'bounded_optimal'})
        entry['discovery_records_modified']=False
        results.append(entry)
    write_json(out/'post_audit.json',results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('lock','train','benchmark','secondary','supplement','audit','verify'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/publication'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--ablation',choices=('no_budget','no_workspace'))
    args=parser.parse_args();out=args.output_dir
    if args.command=='lock':
        print(lock_protocol(out)['digest'])
    elif args.command=='train':
        train_models(out,None if args.seed is None else [args.seed],args.ablation)
    elif args.command in ('benchmark','secondary'):
        if args.seed is None:
            parser.error('--seed is required for a benchmark batch')
        benchmark_batch(out,args.seed,args.repeat,args.command=='secondary')
    elif args.command=='supplement':
        constructive_controls(out);calibration(out);boundary_evaluation(out);application(out)
    elif args.command=='audit':
        post_audit(out)
    else:
        plan=read_lock(out);checks=[]
        for path in sorted((out/'proofs').glob('*.json')):
            proof=json.loads(path.read_text());checks.append({'file':str(path),'check':verify_phase_cover(proof)})
        write_json(out/'standalone_verification.json',checks)
        if not all(row['check']['valid'] for row in checks):
            raise SystemExit('certificate verification failed')
        print('verified',len(checks),'closed covers')


if __name__=='__main__':
    main()
