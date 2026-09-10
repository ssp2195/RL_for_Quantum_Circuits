"""Repeat the original validation-only rate grid; never overwrite the lock."""
import argparse
from dataclasses import replace
from pathlib import Path
from hybrid_qcs.publication_corpus import corpus
from hybrid_qcs.publication_train import train_phase_hierarchy
from hybrid_qcs.publication_pipeline import anytime_discovery, public_parity_cnot_bound
from hybrid_qcs.publication_runner import write_json
from hybrid_qcs.resource_search import WorkLimits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/rate-selection.json'))
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose another path to preserve earlier selection')
    records=[]
    for rate in (.03,.001,.003,.01):
        model,training=train_phase_hierarchy(0,rate=rate)
        values=[]
        for problem in corpus()[1]:
            problem=replace(problem,max_cnot=max(1,public_parity_cnot_bound(problem)))
            result=anytime_discovery(problem,model,limits=WorkLimits(2048,12000,.3,.3))
            best=result['best']
            values.append({'name':problem.name,'success':best is not None,
                           'cnot':best['resources']['cnot'] if best else None,
                           'savings':1-best['resources']['cnot']/problem.max_cnot if best else 0.,
                           'result':result})
        records.append({'rate':rate,'training':training,'validation':values,
                        'mean_savings':sum(v['savings'] for v in values)/len(values),
                        'successes':sum(v['success'] for v in values)})
    winner=max(records,key=lambda r:(r['mean_savings'],r['successes']))
    write_json(args.output,{'runs':records,'selected_rate':winner['rate'],
                           'does_not_modify_existing_protocol':True,
                           'qualification':'timing-sensitive rerun; a different selection is a new experiment, not retroactive replacement'})


if __name__=='__main__':main()
