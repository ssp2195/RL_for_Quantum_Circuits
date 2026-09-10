"""Attributable, discovery-only anytime synthesis; auditing is a separate job.

This module deliberately has no auditor import. It never seeds a learned search
with a compiler witness. A public resource bound is not a target circuit.
"""
from __future__ import annotations
from dataclasses import replace
import time
from .phase_search import search_phase
from .phase_baselines import uniform_cost_phase
from .resource_search import WorkLimits


def public_parity_cnot_bound(problem):
    return 2*sum(mask.bit_count()-1 for mask,_ in problem.coefficients)


def anytime_discovery(problem, model, *, scheduler='hierarchy',
                       limits=WorkLimits(4096, 20000, .4, .4), panel_size=32,
                       dag=True):
    """Tighten CNOT caps without allowing an auditor to discover an incumbent."""
    started, cpu = time.perf_counter(), time.process_time()
    edges, rounds, incumbents = 0, [], []
    current = problem
    for attempt in range(problem.max_cnot+2):
        wall_left = limits.wall_seconds-(time.perf_counter()-started)
        cpu_left = limits.cpu_seconds-(time.process_time()-cpu)
        edge_left = limits.max_edges-edges
        if wall_left <= 0 or cpu_left <= 0 or edge_left <= 0:
            break
        allowance=WorkLimits(edge_left,limits.max_records,wall_left,cpu_left)
        if scheduler=='uniform_cost':
            result=uniform_cost_phase(current,limits=allowance,dag=dag)
        else:
            result=search_phase(current,model,scheduler=scheduler,limits=allowance,
                                panel_size=panel_size,dag=dag)
        edges += result['edges']
        result.pop('training_transitions',None)
        result.pop('expanded_tokens',None)
        rounds.append(result)
        if result['witness'] is None:
            break
        witness=result['witness']
        if incumbents and witness['resources']['cnot']>=incumbents[-1]['resources']['cnot']:
            raise AssertionError('bound tightening did not strictly improve')
        incumbents.append({'wall_seconds':time.perf_counter()-started,
                           'cpu_seconds':time.process_time()-cpu,'edges':edges,
                           'resources':witness['resources'],'witness':witness,
                           'source':result['witness_source'],
                           'contract_digest':current.digest})
        if not witness['resources']['cnot'] or scheduler=='uniform_cost':
            break
        current=replace(problem,max_cnot=witness['resources']['cnot']-1)
    best=incumbents[-1] if incumbents else None
    return {'schema':'attributable-anytime-discovery-v1','problem':problem.manifest(),
            'problem_digest':problem.digest,'scheduler':scheduler,
            'policy_digest':model.digest if model is not None else None,
            'status':'feasible' if best else 'unknown','best':best,
            'incumbents':incumbents,'rounds':rounds,'edges':edges,
            'lookahead_transitions':sum(r['profile']['lookahead_transitions'] for r in rounds),
            'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
            'first_correct_seconds':incumbents[0]['wall_seconds'] if incumbents else None,
            'improvements':max(0,len(incumbents)-1),'audit_calls':0,'audit_witness_used':False,
            'optimality':'not inferred from learned search or timeout'}
