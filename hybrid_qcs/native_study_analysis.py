"""Target-cluster paired analysis; no timing/seed pseudoreplication."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .native_study_corpus import dump


def read_rows(path):
    if not Path(path).exists():return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired_interval(rows,method,control,metric='success',seed=9201,resamples=5000):
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['method'] in (method,control):groups[row['name']][row['method']].append(row[metric])
    differences=[np.mean(group[method])-np.mean(group[control]) for group in groups.values()
                 if group[method] and group[control]]
    if not differences:return {'clusters':0,'mean':None,'lower_95':None,'upper_95':None}
    d=np.asarray(differences);rng=np.random.default_rng(seed)
    means=d[rng.integers(len(d),size=(resamples,len(d)))].mean(axis=1)
    return {'clusters':len(d),'mean':float(d.mean()),'lower_95':float(np.quantile(means,.025)),
            'upper_95':float(np.quantile(means,.975)),
            'unit':'logical target; seed/repetition means within target first',
            'scope':'descriptive target-population interval, no multiplicity adjustment', 'resamples':resamples}


def analyze(output):
    output=Path(output);protocol=json.loads((output/'protocol.json').read_text())
    specs={r['name']:r for r in protocol['splits']['test']}
    raw=read_rows(output/'primary.jsonl');flat=[]
    for row in raw:
        r=row['result'];success=bool(r['timely_success']);ref=specs[row['name']]['generator_length']
        gates=r['witness']['resources']['gates'] if success else None
        flat.append({**{k:row[k] for k in ('name','method','seed','axis','budget','repeat')},
                     'success':float(success),'savings':(ref-gates)/ref if success else 0.,
                     'resource_score':(specs[row['name']]['problem']['budget']['max_gates']+1-gates)/(specs[row['name']]['problem']['budget']['max_gates']+1) if success else 0.,
                     'gates':gates,
                     'wall':r['external_wall_seconds'],'edges':r['edges'],
                     'selection':r.get('profile',{}).get('selection_seconds',0.),
                     'strict_improvements':r['strict_improvements'],'late':r.get('late_witness',False),
                     'guard_limited':row['axis']=='edges' and any(q['reason'] in ('wall_limit','cpu_limit','record_limit') for q in r['rounds'])})
    summary=[]
    for axis,budgets in protocol['primary_budget_axes'].items():
        for budget in budgets:
            group=[r for r in flat if r['axis']==axis and r['budget']==budget]
            methods={}
            for method in protocol['methods']:
                rows=[r for r in group if r['method']==method]
                if not rows:continue
                means={k:float(np.mean([r[k] for r in rows])) for k in ('success','savings','resource_score','wall','edges','selection')}
                methods[method]={'runs':len(rows),'successes':int(sum(r['success'] for r in rows)),
                                 **means,'strict_improvement_runs':sum(r['strict_improvements']>0 for r in rows),
                                 'late_witness_rounds':sum(r['late'] for r in rows),
                                 'guard_limited_runs':sum(r['guard_limited'] for r in rows),
                                 'mean_gates_successful':float(np.mean([r['gates'] for r in rows if r['gates'] is not None])) if any(r['gates'] is not None for r in rows) else None}
            comparisons={control:{metric:paired_interval(group,'hierarchy',control,metric)
                                  for metric in ('success','resource_score','savings')} for control in ('untrained','mitm','uniform_cost')}
            summary.append({'axis':axis,'budget':budget,'methods':methods,'paired':comparisons})
    # Report between-training-seed sensitivity as well as target bootstrap.
    seed_summary={}
    for seed in protocol['primary_seeds']:
        rows=[r for r in flat if r['method']=='hierarchy' and r['seed']==seed]
        seed_summary[str(seed)]={'runs':len(rows),'success':float(np.mean([r['success'] for r in rows])) if rows else None}
    result={'schema':'native-study-analysis-v1','runs':len(flat),'logical_test_targets':len(specs),
            'budget_results':summary,'training_seed_summary':seed_summary,
            'interpretation':'controlled native study; successful certification does not imply learned superiority',
            'primary_objective':'minimize native gates under original caps; Q=success*(Gcap+1-G)/(Gcap+1), failure Q=0',
            'no_test_selection':True}
    dump(output/'analysis.json',result);return result
