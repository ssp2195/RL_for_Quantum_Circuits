"""Post-test bounded-continuation probes; never used to train or select a policy.

The diagnostic probes use withheld construction prefixes AFTER primary evaluation.
They are not a claim that evaluated policies visited these states. Exact replay
confirms every short completion found by independent numerical enumeration.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from .model import Gate,HybridState
from .native_domain import legal,next_t_depths,apply_gate_to_isometry,matrix_error
from .native_policy import NativeHierarchy,outer_features,inner_features
from .native_exact import ExactMatrix,verify_exact_word
from .native_study_corpus import dump,load_problem
from .native_study_search import StudyPolicy


def short_completion(p,prefix,state,td,iso,first,horizon,exact):
    """Return a verified completion length up to horizon, or unresolved."""
    if not legal(p,state,td,first):return None
    first_state=state.apply(first,partial_order_reduction=False)
    first_iso=apply_gate_to_isometry(iso,first)
    stack=[((first,),first_state,next_t_depths(td,first),first_iso)]
    for depth in range(1,horizon+1):
        nxt=[]
        for word,s,t,u in stack:
            if matrix_error(p,u)<=p.tolerance:
                native=[[g.name,list(g.qubits)] for g in (*prefix,*word)]
                if verify_exact_word(p.contract,exact,native)['valid']:return len(word)
            if depth==horizon:continue
            for g in p.actions:
                if legal(p,s,t,g):
                    # Hybrid representation for every diagnostic prefix too.
                    nxt.append((word+(g,),s.apply(g,partial_order_reduction=False),next_t_depths(t,g),apply_gate_to_isometry(u,g)))
        stack=nxt
    return None


def diagnose(protocol,output):
    output=Path(output)
    refs=json.loads((output/'construction_references.json').read_text())['words']
    models=[StudyPolicy.load(output/'checkpoints/full'/f'seed-{seed}.json') for seed in protocol['primary_seeds']]
    prior=NativeHierarchy().freeze();probes=[]
    # Fixed 2-step horizon keeps the diagnostic independent of a search budget
    # outcome and avoids relabelling lack of a completion as global infeasibility.
    for row in protocol['splits']['test']:
        p=load_problem(row);exact=ExactMatrix.from_payload(row['exact_target'])
        word=tuple(Gate(g,tuple(qs)) for g,qs in refs[row['name']])
        prefix=word[:-2]
        s=HybridState.identity(p.width,p.budget);td=(0,)*p.width;iso=np.array(p.contract.input_embedding)
        for g in prefix:
            s=s.apply(g,partial_order_reduction=False);td=next_t_depths(td,g);iso=apply_gate_to_isometry(iso,g)
        record=SimpleNamespace(state=s,t_depths=td,x=outer_features(p,s,iso))
        xs=[];costs=[];gates=[]
        for g in p.actions:
            if not legal(p,s,td,g):continue
            projected=apply_gate_to_isometry(iso,g)
            xs.append(inner_features(p,record,g,projected));gates.append(g)
            costs.append(short_completion(p,prefix,s,td,iso,g,2,exact))
        if not any(c is not None for c in costs):raise AssertionError('withheld suffix failed diagnostic verification')
        prior_scores=[prior.score_inner(x,g.name) for x,g in zip(xs,gates,strict=True)]
        prior_choice=int(np.argmax(prior_scores));items=[]
        for model in models:
            scores=[model.score_inner(x,g.name) for x,g in zip(xs,gates,strict=True)]
            selected=int(np.argmax(scores))
            items.append({'seed':model.seed,'selected':selected,'has_verified_two_gate_completion':costs[selected] is not None,
                          'disagrees_with_prior':selected!=prior_choice,
                          'positive_distance_step':bool(xs[selected][3]<-1e-12)})
        alias=[]
        for i in range(len(gates)):
            for j in range(i):
                if gates[i].name==gates[j].name and np.max(np.abs(xs[i]-xs[j]))<1e-12 and (costs[i] is None)!=(costs[j] is None):
                    alias.append([i,j])
        beneficial_worsen=any(c is not None and x[3]<-1e-12 for c,x in zip(costs,xs,strict=True))
        probes.append({'name':row['name'],'native_prefix':[[g.name,list(g.qubits)] for g in prefix],
                       'native_actions':[[g.name,list(g.qubits)] for g in gates],
                       'verified_completion_lengths':costs,'prior_choice':prior_choice,
                       'prior_has_verified_two_gate_completion':costs[prior_choice] is not None,
                       'trained':items,'near_feature_alias_pairs':alias,
                       'verified_completion_can_initially_increase_distance':beneficial_worsen})
    summary={'probes':len(probes),'trained_selections':sum(len(r['trained']) for r in probes),
             'trained_short_completion_hits':sum(x['has_verified_two_gate_completion'] for r in probes for x in r['trained']),
             'prior_short_completion_hits':sum(r['prior_has_verified_two_gate_completion'] for r in probes),
             'near_alias_probes':sum(bool(r['near_feature_alias_pairs']) for r in probes),
             'nonmonotone_completion_probes':sum(r['verified_completion_can_initially_increase_distance'] for r in probes),
             'scope':'off-policy withheld-prefix diagnostic after testing; unresolved is not an unbounded action-value label'}
    dump(output/'diagnostics.json',{'summary':summary,'probes':probes});return summary


def ranking_profile(protocol,output):
    """Separate raw scoring work from the policy change induced by panel restriction."""
    import time
    from .resource_search import WorkLimits
    from .native_study_search import StudySearch
    output=Path(output);model=StudyPolicy.load(output/'checkpoints/full/seed-11.json');rows=[]
    for spec in protocol['splits']['validation']:
        p=load_problem(spec);engine=StudySearch(p,WorkLimits(512,30000,10.,10.))
        engine.run(model,scheduler='untrained')
        if not engine.frontier:continue
        all_records=list(engine.frontier.values())
        panel=[engine.records[i] for i in engine.index.smallest(32)]
        xs=np.array([engine.context(r) for r in all_records]);xp=np.array([engine.context(r) for r in panel])
        scores=model.score_outer(xs);panel_scores=model.score_outer(xp)
        global_choice=all_records[int(np.argmax(scores))].record_id
        panel_choice=panel[int(np.argmax(panel_scores))].record_id
        def timed(a):
            start=time.process_time_ns()
            for _ in range(100):model.score_outer(a)
            return (time.process_time_ns()-start)/100/1e9
        rows.append({'name':spec['name'],'frontier_size':len(all_records),'panel_size':len(panel),
                     'full_score_cpu_seconds':timed(xs),'panel_score_cpu_seconds':timed(xp),
                     'different_selected_record':global_choice!=panel_choice})
    dump(output/'ranking_profile.json',{'snapshots':rows,
         'scope':'fixed validation frontiers; scoring arrays precomputed; no end-to-end speedup inferred',
         'selection_set_change_reported_separately':True})
    return rows
