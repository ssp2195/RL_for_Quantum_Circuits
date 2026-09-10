"""On-policy staged training using specifications only, then immutable evaluation."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, replace
import time
import numpy as np
from .phase_policy import PhaseHierarchy, ARMS
from .phase_search import search_phase
from .resource_search import WorkLimits
from .publication_corpus import corpus, key


def train_phase_hierarchy(seed=0, *, episodes=(64,96,24), rate=.01,
                          limits=WorkLimits(512,10000,20.,20.), ablation=None):
    train,_,_=corpus()
    model=PhaseHierarchy(seed=seed,rate=rate)
    model.ablation=ablation
    schedule=np.random.default_rng(seed+1729)
    logs=[]
    start=time.perf_counter();cpu=time.process_time()
    counts=Counter();successes=Counter()
    for stage,number in zip(('outer','inner','adjust'),episodes,strict=True):
        model.stage=stage
        frozen_outer=model.outer.copy()
        frozen_inner=model.inner_updates
        order=schedule.permutation(len(train))
        for i in range(number):
            if i and i%len(train)==0:order=schedule.permutation(len(train))
            p=train[int(order[i%len(train)])]
            # Resource-conditioned episodes vary caps, not hidden target paths.
            # Some sampled caps can be infeasible. A failed episode is never
            # labelled an infeasibility proof or repaired by an audit witness.
            star = 2*sum(mask.bit_count()-1 for mask,_ in p.coefficients)
            p=replace(p,max_cnot=max(1,star+2*p.ancillas-(i%3)))
            result=search_phase(p,model,train='inner' if stage=='inner' else 'outer',
                                scheduler='outer' if stage=='outer' else 'hierarchy',
                                limits=limits,dag=False)
            for transition in result['training_transitions']:
                counts[transition['arm']]+=1
                counts['phase_emitting']+=transition['phase_emission']
                counts['workspace']+=transition['workspace']
            successes[stage]+=result['status']=='feasible'
            logs.append({'stage':stage,'episode':i,'target':p.name,'orbit':key(p),
                         'problem_digest':p.digest,'problem':p.manifest(),'status':result['status'],'reason':result['reason'],
                         'edges':result['edges'],'wall_seconds':result['wall_seconds'],
                         'profile':result['profile'],'outer_updates':model.outer_updates,
                         'inner_updates':model.inner_updates,
                         'witness':result['witness'],
                         'reward_sum':sum(t['reward'] for t in result['training_transitions'])})
        if stage=='inner' and not np.array_equal(frozen_outer,model.outer):
            raise AssertionError('outer weights changed during bandit training')
        if stage!='inner' and frozen_inner!=model.inner_updates:
            raise AssertionError('bandit weights changed during SARSA training')
    model.freeze()
    report={'seed':seed,'rate':rate,'episodes':list(episodes),'limits':asdict(limits),
            'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'transition_coverage':dict(counts),'stage_successes':dict(successes),
            'bandit_arm_updates':dict(model.arm_updates),
            'all_arms_trained':all(model.arm_updates[a]>0 for a in ARMS),
            'outer_updates':model.outer_updates,'inner_updates':model.inner_updates,
            'checkpoint_digest':model.digest,'logs':logs,
            'target_circuits_supplied':False,'audit_calls':0}
    return model,report
