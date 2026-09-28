"""A separating shared-lifetime example, plus small independent checks.

This is a declared mechanism diagnostic, not another independent performance
sample. Its exact factor order illustrates what the streaming theorem adds.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

from .contract import CleanupProblem, Consumer, Limits
from .consumer_compile import compile_factors, verify_factor_receipt
from .phase_protocol import verify_phase
from .consumer_search import synthesize_oracle
from .consumer_runner import write
from .verify import verify_truth_table_generators


def overlapping_problem() -> CleanupProblem:
    terms = []
    for j in range(4):
        terms.append((f'f{j}_{j}',))
        pairs = ((0,1),(2,3)) if j%2==0 else ((0,2),(1,3))
        for i,k in pairs:
            terms.append((f'b{k}',f'f{i}_{j}'))
    return CleanupProblem('shared-quadratic-feature-lifetimes',4,4,Consumer(tuple(terms)))


def run(output: Path) -> dict:
    p=overlapping_problem()
    rows=[]
    for method,order,storage in (('store_all',None,'all'),('canonical_stream',(0,1,2,3),'stream'),
                                 ('minimum_live_stream',None,'stream'),('recompute',None,'recompute')):
        c,r=compile_factors(p,storage=storage,order=order)
        v=verify_phase(p,c)
        assert v['valid'] and verify_factor_receipt(p,c,r)
        truth=verify_truth_table_generators(p,c)
        assert truth['valid']
        c.persistent_tail()
        rows.append({'method':method,'problem':p.manifest(),'protocol':c.payload(),
                     'receipt':r,'verification':v,'truth_table':truth})
    # The exact schedule uses three auxiliary slots without duplicating feature
    # computations. The canonical order and all-at-once storage require five.
    assert [row['verification']['resources']['peak_aux'] for row in rows[:3]]==[5,5,3]
    assert [row['verification']['resources']['t_count'] for row in rows[:3]]==[32,32,32]
    tight=replace(p,limits=Limits(max_aux=3,max_t=32))
    result=synthesize_oracle(tight)
    assert result['status']=='certified_upper_bound'
    assert result['selected']['verification']['resources']['peak_aux']==3
    output=Path(output)
    write(output/'streaming_diagnostic.json',{'scope':'constructed mechanism example, not held-out performance evidence',
          'cases':rows,'tight_cap_selection':result['selected']['method']})
    for row in rows:
        from .phase_protocol import PhaseProtocol
        (output/f"streaming-{row['method']}.qasm").write_text(PhaseProtocol.from_payload(row['protocol']).qasm3())
    return {row['method']:{k:row['verification']['resources'][k]
               for k in ('peak_aux','t_count','native_depth','measurement_rounds')}for row in rows}


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output_dir),indent=2))
