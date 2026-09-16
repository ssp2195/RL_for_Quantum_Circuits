"""Fresh staged training, never target-witness imitation or test-time fitting."""
from __future__ import annotations
from dataclasses import asdict
import time
import numpy as np
from ..resource_search import WorkLimits
from .benchmarks import training_problems, candidate_problems
from .policy import CleanupHierarchy
from .search import CleanupSearch


def train_cleanup(seed: int, stages=(64,96,24), *, edge_budget=512, seconds=5.) -> tuple[CleanupHierarchy, dict]:
    if len(stages)!=3 or any(type(n) is not int or n<1 for n in stages):
        raise ValueError('outer/inner/outer schedules must all be nonempty')
    start=time.perf_counter(); model=CleanupHierarchy(seed); problems=training_problems()
    if {p.oracle_digest for p in problems}&{p.oracle_digest for p in candidate_problems()}:
        raise AssertionError('training/test logical specification overlap')
    rng=np.random.default_rng(seed+937); episodes=[];counter=0
    for stage,n in zip(('outer','inner','outer'),stages):
        for j in range(n):
            p=problems[int(rng.integers(len(problems)))];mode='coherent' if counter%5==4 else 'measured'
            before_outer=model.w.copy()
            before_inner=model.a.copy()
            limits=WorkLimits(edge_budget,10000,seconds,seconds)
            result=CleanupSearch(p,limits,mode=mode).run(model,train=stage)
            if stage=='inner' and not np.array_equal(before_outer,model.w):raise AssertionError('outer must stay frozen during inner fitting')
            if stage=='outer' and not np.array_equal(before_inner,model.a):raise AssertionError('inner must stay frozen during outer fitting')
            episodes.append({'index':counter,'stage':stage,'problem_digest':p.digest,'oracle_digest':p.oracle_digest,
                             'mode':mode,'status':result['status'],'reason':result['reason'],
                             'edges':result['edges'],'records':result['records'],
                             'transitions':result['training_transitions']})
            counter+=1
    if model.outer_updates<1 or np.any(model.updates<1):
        raise AssertionError('training did not exercise both learners and every continuation family')
    model.freeze()
    return model,{'schema':'cleanup-training-v1','seed':seed,'stages':list(stages),
                  'episodes':episodes,'episode_count':counter,'policy_digest':model.digest,
                  'outer_updates':model.outer_updates,'inner_updates':model.updates.tolist(),
                  'wall_seconds':time.perf_counter()-start,'edge_budget_per_episode':edge_budget,
                  'training_oracle_digests':sorted({p.oracle_digest for p in problems}),
                  'test_oracle_digests':sorted({p.oracle_digest for p in candidate_problems()}),
                  'test_inputs_used_for_training':False,'reference_circuits_used_for_training':False,
                  'schedule_completion_is_not_convergence':True}
