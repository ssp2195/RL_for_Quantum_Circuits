"""Specification-only staged native training and validation-only model selection."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
from .native_study_corpus import dump,load_problem
from .native_study_search import StudyPolicy,StudySearch,run_discovery
from .resource_search import WorkLimits
from .native_policy import FAMILIES


def train_model(protocol,seed,alpha,*,variant='full',output=None,stages=None):
    stages=tuple(protocol['training_stages'] if stages is None else stages)
    model=StudyPolicy(seed,alpha,variant=variant)
    specs=protocol['splits']['training'];log=[];total_edges=0
    start,cpu=time.perf_counter(),time.process_time()
    rng=np.random.default_rng(seed+9000)
    for stage,count in zip(('outer','inner','outer'),stages,strict=True):
        order=rng.permutation(len(specs));cycle=0
        for i in range(count):
            if i and i%len(specs)==0:order=rng.permutation(len(specs));cycle+=1
            row=specs[int(order[i%len(specs)])]
            # Cross one in four episodes with an available clean qubit. This
            # changes only the physical contract, never the target unitary.
            anc=int((i+cycle)%4==0)
            p=load_problem(row,anc)
            edge_limit=protocol['training_edge_limit']
            limits=WorkLimits(edge_limit,5000,protocol['training_seconds_per_episode'],protocol['training_seconds_per_episode'])
            r=StudySearch(p,limits,axis='edges').run(model,train=stage,epsilon=.25,
                                                    inner_response=model.response_mode)
            total_edges+=r['edges']
            log.append({'name':row['name'],'target_digest':p.digest,'orbit':row['orbit'],'ancillas':anc,
                        'stage':stage,'status':r['status'],'edges':r['edges'],'reason':r['reason'],
                        'transitions':r['training_transitions']})
    model.freeze()
    if not all(model.updates):raise AssertionError('training did not update all native gate families')
    report={'seed':seed,'alpha':alpha,'variant':variant,'stages':list(stages),'episodes':log,
            'total_edges':total_edges,'cpu_seconds':time.process_time()-cpu,
            'wall_seconds':time.perf_counter()-start,'checkpoint_digest':model.digest,
            'family_updates':dict(zip(FAMILIES,map(int,model.updates))),
            'training_targets_only':True,'references_supplied':False,'audit_calls':0}
    if output:
        output=Path(output);model.save(output/'checkpoints'/variant/f'seed-{seed}.json')
        dump(output/'training'/variant/f'seed-{seed}.json',report)
    return model,report


def select_and_train(protocol,output):
    output=Path(output)
    if (output/'TRAINING_COMPLETE').exists():
        return json.loads((output/'selection.json').read_text())
    records=[];start_cpu=time.process_time();cap=protocol['training_cpu_cap_all_fits']
    for alpha in protocol['selection']['alpha_candidates']:
        seed=protocol['selection']['seeds'][0]
        m,tr=train_model(protocol,seed,alpha,output=output/'selection-fits'/str(alpha))
        rows=[]
        for row in protocol['splits']['validation']:
            p=load_problem(row)
            r=run_discovery(p,m,'hierarchy',axis='edges',budget=protocol['selection']['edges'])
            rows.append({'name':row['name'],'result':r})
        successes=sum(x['result']['timely_success'] for x in rows)
        # Assign a fixed cap+1 penalty to a failure; no success-only selection.
        gate_penalty=sum(x['result']['witness']['resources']['gates'] if x['result']['timely_success']
                         else load_problem(row).budget.max_gates+1 for row,x in zip(protocol['splits']['validation'],rows,strict=True))
        attempted=sum(x['result']['edges'] for x in rows)
        records.append({'alpha':alpha,'successes':successes,'gate_penalty':gate_penalty,'attempted_edges':attempted,
                        'rows':rows,'training_cpu_seconds':tr['cpu_seconds']})
        if time.process_time()-start_cpu>cap:raise RuntimeError('aggregate fitting CPU cap reached')
    winner=min(records,key=lambda r:(-r['successes'],r['gate_penalty'],r['attempted_edges'],r['alpha']))
    selection={'schema':'native-study-selection-v1','selected_alpha':winner['alpha'],'validation':records,
               'selection_uses_test_outcomes':False,'scope':'selection only, before all held-out evaluations'}
    dump(output/'selection.json',selection)
    for variant in ('full',*protocol['exploratory_ablations']):
        for seed in (protocol['primary_seeds'] if variant=='full' else protocol['ablation_seeds']):
            train_model(protocol,seed,winner['alpha'],variant=variant,output=output)
            if time.process_time()-start_cpu>cap:raise RuntimeError('aggregate fitting CPU cap reached')
    selection['all_fitting_cpu_seconds']=time.process_time()-start_cpu
    selection['training_cpu_cap']=cap
    dump(output/'selection.json',selection)
    (output/'TRAINING_COMPLETE').write_text('all native checkpoints frozen before held-out evaluation\n')
    return selection
