"""Active native-hybrid qualification; historical phase-study data stay intact."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
from .native_domain import SCHEMA
from .native_policy import NativeHierarchy, FAMILIES
from .native_search import NativeSearch, optimize_native
from .native_benchmarks import training_problems, named_benchmarks, prior_phase_benchmarks, reference_control
from .resource_search import WorkLimits


def train_native(seed=0, *, stages=(16,24,8), limits=WorkLimits(128,5000,3.,3.)):
    if len(stages)!=3 or any(type(n) is not int or n<1 for n in stages):
        raise ValueError('three positive training stage lengths are required')
    model=NativeHierarchy(seed);problems=training_problems();rows=[]
    start,cpu=time.perf_counter(),time.process_time()
    for stage,episodes in zip(('outer','inner','outer'),stages,strict=True):
        for _ in range(episodes):
            p=problems[model.episodes%len(problems)]
            result=NativeSearch(p,limits).run(model,train=stage,epsilon=.25)
            rows.append({'target':p.name,'target_digest':p.digest,'stage':stage,
                         'status':result['status'],'edges':result['edges'],
                         'training_transitions':result['training_transitions']})
    model.freeze()
    return model,{'schema':SCHEMA,'seed':seed,'stages':list(stages),'episodes':rows,
                  'phase_problem_witnesses_supplied':False,'limits':asdict(limits),
                  'family_updates':dict(zip(FAMILIES,map(int,model.updates))),
                  'outer_updates':model.outer_updates,'wall_seconds':time.perf_counter()-start,
                  'cpu_seconds':time.process_time()-cpu,'checkpoint_digest':model.digest}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def qualify(output, *, seeds=(0,1,2,3,4), edge_limit=512, seconds=3., stages=(64,96,24), named_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    targets=named_benchmarks()+(() if named_only else prior_phase_benchmarks())
    limits=WorkLimits(edge_limit,20000,seconds,seconds)
    files=sorted(Path(__file__).parent.glob('native_*.py'))+[Path(__file__).with_name('model.py'),Path(__file__).with_name('clifford_lift.py')]
    manifest={'schema':SCHEMA,'base_commit':'4e5d1a429f9ec6514a0afbae7a29929ffe4845ef',
              'python':platform.python_version(),'platform':platform.platform(),'seeds':list(seeds),
              'stages':list(stages),'limits':asdict(limits),'comparison':'representation restoration qualification, not a new publication performance campaign',
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'targets':[{'name':p.name,'family':p.family,'problem':p.manifest()} for p in targets],
              'legacy_phase_evidence':'unchanged; not evidence for the restored native engine'}
    write(output/'manifest.json',manifest)
    results=[];models=[]
    for seed in seeds:
        model,training=train_native(seed,stages=stages)
        model.save(output/'checkpoints'/f'seed-{seed}.json');write(output/'training'/f'seed-{seed}.json',training)
        models.append(model)
        for p in targets:
            for scheduler in ('hierarchy','untrained'):
                result=NativeSearch(p,limits).run(model,scheduler=scheduler)
                result['seed']=seed;results.append(result)
                write(output/'outcomes.json',results)
    # Independently constructed witnesses are evaluated only after discovery.
    # These are never injected into frontiers or counted as learned successes.
    controls=[reference_control(p) for p in named_benchmarks()]
    write(output/'reference_controls.json',controls)
    summary={'schema':SCHEMA,'target_count':len(targets),'outcomes':len(results),
             'native_representation_only':all(r['profile']['phase_obligation_transitions']==0 for r in results),
             'discovery':{s:{'attempts':sum(r['scheduler']==s for r in results),
                'certified':sum(r['scheduler']==s and r['status']=='feasible' for r in results)} for s in ('hierarchy','untrained')},
             'reference_controls_certified':sum(r['status']=='certified' for r in controls),
             'hard_target_results':[{k:r[k] for k in ('name','seed','scheduler','status','reason','edges')} for r in results if r['family']!='phase-oracle'],
             'claim':'native representation restored; no claim of learned superiority or unrestricted optimality'}
    write(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('optimize','discover'),default='optimize',
                        help='default: proof-carrying resource optimization; discover: historical first-feasible qualification')
    parser.add_argument('--objectives',nargs='+',default=['t_count','cnot','depth','gates'])
    parser.add_argument('--output-dir',default='outputs/native-hybrid-restoration')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    parser.add_argument('--edge-limit',type=int,default=None)
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--stages',type=int,nargs=3,default=[64,96,24])
    parser.add_argument('--named-only',action='store_true')
    args=parser.parse_args()
    if min(args.stages)<1:parser.error('each training stage must have at least one episode')
    edge_limit=args.edge_limit if args.edge_limit is not None else (20000 if args.mode=='optimize' else 512)
    if args.mode=='optimize':
        from .native_optimality_runner import run_study
        print(json.dumps(run_study(args.output_dir,seeds=args.seeds,stages=args.stages,
            targets='named' if args.named_only else 'all',objectives=args.objectives,
            limits=WorkLimits(edge_limit,20000,args.seconds,args.seconds)),indent=2))
        return
    print(json.dumps(qualify(args.output_dir,seeds=args.seeds,edge_limit=edge_limit,
                            seconds=args.seconds,stages=args.stages,named_only=args.named_only),indent=2))


if __name__=='__main__':main()
