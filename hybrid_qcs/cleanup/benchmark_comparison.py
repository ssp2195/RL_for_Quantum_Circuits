"""Matched, auditable comparisons. This module never changes the synthesizer.

The external mockturtle program optimizes XOR/AND networks. We independently
verify its Boolean function, then compile it to a *complete phase oracle* with
the same checked four-T AND and measured Clifford correction as our method.
Thus no output flag, hidden inverse, or uncharged parity workspace is imposed
on that baseline. These wrappers are not claimed to reproduce all of caterpillar.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time

from .contract import CleanupProblem, Consumer, Polynomial, plus, times, digest
from .benchmarks import candidate_problems
from .phase_protocol import PhaseProtocol, verify_phase
from .ir import Instruction
from .consumer_search import synthesize_oracle, DEFAULT_OBJECTIVE
from .consumer_runner import verify_truth_table_generators
from .verify import primitive_receipts

MOCKTURTLE_COMMIT = '47d1e70fdf775e1a295016c3c17a1ad206db24c0'
EXACT_T_COMMIT = 'bffe54c38b6bfd689a04e0d7d5afdbd949bdac3a'
BASE = '2ca775e70ec3b6da94a7246a57874f5d6cee6d3f'


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


@dataclass(frozen=True)
class Network:
    n: int
    gates: tuple[tuple[str, int, int], ...]
    output: int

    def __post_init__(self):
        if type(self.n) is not int or not 1 <= self.n <= 128:
            raise ValueError('invalid input width')
        for i, (op, a, b) in enumerate(self.gates, self.n+1):
            if op not in ('A','X') or min(a,b)<0 or max(a,b)>=2*i:
                raise ValueError('non-topological or unsupported network')
        if type(self.output) is not int or not 0 <= self.output < 2*(self.n+len(self.gates)+1):
            raise ValueError('invalid output')

    def text(self):
        return f'{self.n} {len(self.gates)} {self.output}\n'+''.join(f'{o} {a} {b}\n' for o,a,b in self.gates)

    @classmethod
    def parse(cls, text):
        lines=text.strip().splitlines()
        n,k,out=map(int,lines[0].split())
        if len(lines)!=k+1:
            raise ValueError('unexpected external output size')
        return cls(n,tuple((o,int(a),int(b)) for o,a,b in (s.split() for s in lines[1:])),out)

    def polynomial(self):
        values=[frozenset()]+[frozenset({1<<i}) for i in range(self.n)]
        def v(lit): return plus(values[lit//2],frozenset({0}) if lit&1 else frozenset())
        for o,a,b in self.gates: values.append(times(v(a),v(b)) if o=='A' else plus(v(a),v(b)))
        return v(self.output)

    def verilog(self):
        lines=['module oracle('+','.join([f'x{i}' for i in range(self.n)]+['y'])+');',
               'input '+','.join(f'x{i}' for i in range(self.n))+';', 'output y;']
        if self.gates: lines.append('wire '+','.join(f'g{i}' for i in range(len(self.gates)))+';')
        def lit(a):
            i=a//2
            name="1'b0" if i==0 else f'x{i-1}' if i<=self.n else f'g{i-self.n-1}'
            return ('~'+name) if a&1 else name
        for i,(o,a,b) in enumerate(self.gates): lines.append(f'assign g{i} = {lit(a)} '+('&' if o=='A' else '^')+f' {lit(b)};')
        lines.extend([f'assign y = {lit(self.output)};', 'endmodule'])
        return '\n'.join(lines)+'\n'


def input_network(p: CleanupProblem, form: str) -> Network:
    gates=[]
    def gate(o,a,b):
        gates.append((o,a,b));return 2*(p.n+len(gates))
    def product(xs):
        if not xs:return 1
        a=xs[0]
        for b in xs[1:]: a=gate('A',a,b)
        return a
    def summation(xs):
        a=0
        for b in xs:a=gate('X',a,b)
        return a
    if form=='anf':
        out=summation([product([2*(i+1) for i in range(p.n) if m>>i&1]) for m in sorted(p.target_polynomial)])
    elif form=='davio':
        memo={frozenset():0,frozenset({0}):1}
        def rec(poly):
            if poly in memo:return memo[poly]
            counts=[sum(bool(t>>i&1) for t in poly) for i in range(p.n)]
            # A fixed algebraic decomposition, not adjusted to benchmark outcomes.
            i=max(range(p.n),key=lambda i:(counts[i],-i))
            mask=1<<i
            q0=frozenset(t for t in poly if not t&mask)
            q1=frozenset(t^mask for t in poly if t&mask)
            f0,f1=rec(q0),rec(q1)
            term=2*(i+1) if f1==1 else 0 if f1==0 else gate('A',2*(i+1),f1)
            value=term if f0==0 else gate('X',f0,term)
            memo[poly]=value;return value
        out=rec(p.target_polynomial)
    elif form=='source':
        names={'a':2,**{f'b{i}':2*(i+2) for i in range(p.r)},
               **{f'x{j}':2*(p.r+j+2) for j in range(p.m)}}
        # Share the original common-enable masks, without imposing quantum storage.
        for i in range(p.r):
            h=gate('A',names['a'],names[f'b{i}'])
            for j in range(p.m):names[f'f{i}_{j}']=gate('A',h,names[f'x{j}'])
        out=summation([product([names[x] for x in term]) for term in p.consumer.terms])
    else:raise ValueError('unknown source form')
    n=Network(p.n,tuple(gates),out)
    if n.polynomial()!=p.target_polynomial:raise AssertionError('incorrect benchmark network')
    return n


def parity_pair(a: frozenset[int], b: frozenset[int]):
    """CNOT coordinates for two independent binary forms; restore by reversal."""
    if not a or not b or a==b:raise ValueError('independent nonzero forms required')
    i=min(a); ops=[(j,i) for j in sorted(a-{i})]
    second=set(b)
    if i in second:second.symmetric_difference_update(a-{i})
    j=min(second-{i})
    ops.extend((k,j) for k in sorted(second-{j}))
    return i,j,tuple(ops)


def lower_network(p: CleanupProblem, net: Network, *, parallel: bool=True) -> PhaseProtocol:
    """Compile affine fanins without extra parity ancillas; all branches checked.

    A complemented form c+L times d+R equals LR+cR+dL+cd. The constant and
    linear terms stay in the affine view; only LR needs a physical AND qubit.
    Reverse dependency layers ensure every measured node's parents survive.
    """
    if net.n!=p.n or net.polynomial()!=p.target_polynomial:raise ValueError('wrong external Boolean function')
    forms=[(frozenset(),0)]+[(frozenset({i}),0) for i in range(p.n)]
    ops=[]; definitions=[]; levels={i:0 for i in range(p.n)}
    def lit(v):
        f,c=forms[v//2];return f,c^(v&1)
    def cx(pairs,stage):
        ops.extend(Instruction('CX',(a,b),stage) for a,b in pairs)
    for o,a,b in net.gates:
        u,c=lit(a);v,d=lit(b)
        if o=='X':forms.append((u^v,c^d));continue
        linear=(v if c else frozenset()) ^ (u if d else frozenset())
        const=c&d
        if not u or not v:forms.append((linear,const));continue
        if u==v:forms.append((linear^u,const));continue
        target=p.n+len(definitions)
        i,j,coord=parity_pair(u,v)
        cx(coord,'prepare');ops.append(Instruction('AND',(i,j,target),'prepare'));cx(reversed(coord),'prepare')
        definitions.append((target,u,v,coord,i,j))
        levels[target]=1+max(levels[k] for k in u|v)
        forms.append((linear^frozenset({target}),const))
    support,const=lit(net.output)
    # Phase-oracle specialization: a terminal AND need not be computed into
    # an output flag. Apply its quadratic phase directly on its surviving fanins.
    parent_wires=set().union(*(set(u|v) for _,u,v,_,_,_ in definitions)) if definitions else set()
    phase_only={t for t,_,_,_,_,_ in definitions if t in support and t not in parent_wires}
    retained=[row for row in definitions if row[0] not in phase_only]
    mapping={i:i for i in range(p.n)}
    mapping.update({row[0]:p.n+j for j,row in enumerate(retained)})
    ops=[]; new_defs=[];levels={i:0 for i in range(p.n)}
    for old,u,v,_,_,_ in retained:
        target=mapping[old];u=frozenset(mapping[i] for i in u);v=frozenset(mapping[i] for i in v)
        i,j,coord=parity_pair(u,v);cx(coord,'prepare')
        ops.append(Instruction('AND',(i,j,target),'prepare'));cx(reversed(coord),'prepare')
        new_defs.append((target,u,v,coord,i,j));levels[target]=1+max(levels[k] for k in u|v)
    if const:ops.append(Instruction('MINUS',(0,),'consumer'))
    ops.extend(Instruction('Z',(mapping[i],),'consumer') for i in sorted(support-phase_only))
    for t,u,v,_,_,_ in definitions:
        if t not in phase_only:continue
        u=frozenset(mapping[i] for i in u);v=frozenset(mapping[i] for i in v)
        i,j,coord=parity_pair(u,v);cx(coord,'consumer')
        ops.append(Instruction('CZ',(i,j),'consumer'));cx(reversed(coord),'consumer')
    definitions=new_defs
    batches=defaultdict(list)
    for row in definitions:batches[levels[row[0]] if parallel else row[0]].append(row)
    outcome=0
    for level in sorted(batches,reverse=True):
        batch=batches[level];stage=f'xag_measure_{level}'
        bits={}
        for row in reversed(batch):
            t=row[0];bits[t]=outcome
            ops.append(Instruction('MX',(t,),stage,outcome=outcome));outcome+=1
        for t,u,v,coord,i,j in reversed(batch):
            stage=f'xag_correct_{level}';cx(coord,stage)
            ops.append(Instruction('CZ',(i,j),stage,guard=bits[t]));cx(reversed(coord),stage)
    return PhaseProtocol(p.digest,p.n+len(definitions),tuple(ops),
        'external_mockturtle_XAG_with_shared_4T_measured_lowering_'+('parallel' if parallel else 'serial'))


def extension_problems() -> list[CleanupProblem]:
    """36 labelled functions fixed without reference to solver outcomes.

    Not a new RL confirmation set: no policy is trained in this comparison.
    Families deliberately span bilinear, quartic and mixed degree-five phases.
    Reject only duplicate labelled functions and zero high-degree targets.
    """
    rng=random.Random(20261004)
    seen={p.oracle_digest for p in candidate_problems()};out=[]
    for family in ('bilinear','quadratic_feature','mixed'):
        i=0
        while i<12:
            r,m=((3,2),(4,2),(4,3),(5,3))[i%4]
            fs=[f'f{a}_{b}' for a in range(r) for b in range(m)]
            raw=['a']+[f'b{a}' for a in range(r)]+[f'x{b}' for b in range(m)]
            terms=[]
            for f in fs:
                if rng.random()<.45:terms.append((f,))
            if family!='bilinear':
                for _ in range(r+m):terms.append(tuple(rng.sample(fs,2)))
            if family=='mixed':
                for _ in range(r+m):terms.append((rng.choice(fs),rng.choice(raw)))
                for _ in range(3):terms.append(tuple(rng.sample(raw,2)))
            p=CleanupProblem(f'extension-{family}-{i:02}',r,m,Consumer(tuple(terms)),split='comparison')
            if p.oracle_digest in seen or max((t.bit_count() for t in p.target_polynomial),default=0)<3:continue
            seen.add(p.oracle_digest);out.append(p);i+=1
    return out


def run(output: Path, binary: str, repeats: int=3, *, resume: bool=False):
    if not resume and output.exists() and any(output.iterdir()):raise ValueError('refuse to overwrite an existing campaign')
    output.mkdir(parents=True,exist_ok=True)
    problems=list(candidate_problems())+extension_problems()
    manifest={'schema':'external-comparison-v1','base':BASE,'mockturtle_commit':MOCKTURTLE_COMMIT,
       'driver_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(), 'repetitions':repeats,
       'problems':[p.manifest() for p in problems], 'method_order_seed':60104,
       'methods':['current_menu','mockturtle_measured'], 'input_networks':['source','anf','davio'],
       'scope':'development comparison; strengthened controls after preserved pilots; 5 reused and 36 fixed labelled targets; not a held-out RL claim',
       'cost':'same 4T AND, phase-only terminal products use CZ instead of unnecessary flags; all corrections, parity CNOTs and reset charged',
       'timing':'warm target-independent primitive cache; all candidates and external subprocess cost included; compiler builds excluded',
       'baseline':'unmodified pinned mockturtle 5-cut min-MC rewriting plus XAG resubstitution; our explicit shared primitive lowering; not full caterpillar or published optimum'}
    if resume and json.loads((output/'PROTOCOL.json').read_text())!=manifest:
        raise ValueError('resume contract/binary mismatch')
    if not resume:save(output/'PROTOCOL.json',manifest)
    primitive_receipts('and4')
    rng=random.Random(60104);jobs=[(p,k,r) for p in problems for r in range(repeats) for k in manifest['methods']];rng.shuffle(jobs)
    rows=[json.loads(x) for x in (output/'runs.jsonl').read_text().splitlines()] if resume else []
    completed={(r['name'],r['method'],r['repeat']) for r in rows}
    for p,method,rep in jobs:
        if (p.name,method,rep) in completed:continue
        started=time.perf_counter();status={};c=None
        try:
            if method=='current_menu':
                result=synthesize_oracle(p,seconds=30.)
                if not result.get('selected'):raise RuntimeError(str(result))
                c=PhaseProtocol.from_payload(result['selected']['protocol'])
                status={'selected_method':result['selected']['method'],'menu_complete':result['menu_complete']}
                alternatives=[{'method':a['method'],'resources':a['verification']['resources']} for a in result['candidates']]
            else:
                choices=[];alternatives=[]
                for form in manifest['input_networks']:
                    net=input_network(p,form);raw=net.text()
                    proc=subprocess.run([binary],input=raw,text=True,capture_output=True,timeout=30,check=True)
                    opt=Network.parse(proc.stdout)
                    if opt.polynomial()!=p.target_polynomial:raise AssertionError('external network changed the truth table')
                    key=digest({'source':raw,'output':proc.stdout})
                    save(output/'networks'/f'{key}.json',{'source_form':form,'source':raw,'optimized':proc.stdout,'stderr':proc.stderr})
                    for parallel in (True,False):
                        circuit=lower_network(p,opt,parallel=parallel);checked=verify_phase(p,circuit)
                        if not checked['valid']:raise AssertionError(checked)
                        circuit.persistent_tail(); rr=checked['resources']
                        choices.append((tuple(rr[k] for k in DEFAULT_OBJECTIVE),circuit,form,key,parallel))
                        alternatives.append({'method':form+('_parallel' if parallel else '_serial'),'resources':rr})
                _,c,form,key,parallel=min(choices,key=lambda x:x[0])
                status={'source_form':form,'network_digest':key,'parallel_cleanup':parallel}
            checked=verify_phase(p,c)
            if not checked['valid']:raise AssertionError(checked)
            c.persistent_tail()
            elapsed=time.perf_counter()-started
            row={'name':p.name,'problem_digest':p.digest,'cohort':'original' if p.name in {a.name for a in candidate_problems()} else 'extension',
                 'method':method,'repeat':rep,'valid':True,'wall_seconds':elapsed,'resources':checked['resources'],
                 'protocol_digest':c.digest,'details':status,'alternatives':alternatives}
            save(output/'protocols'/f'{c.digest}.json',{'problem':p.manifest(),'protocol':c.payload()})
        except (ValueError,RuntimeError,AssertionError,subprocess.SubprocessError) as e:
            row={'name':p.name,'problem_digest':p.digest,'method':method,'repeat':rep,'valid':False,
                 'wall_seconds':time.perf_counter()-started,'reason':str(e)}
        rows.append(row)
        with (output/'runs.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    result=analyse(output);save(output/'summary.json',result)
    verify(output)
    (output/'COMPLETE').write_text('all declared jobs executed; failures remain explicit\n')
    save(output/'EVIDENCE_SHA256.json',{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json'})
    return result


def analyse(output):
    rows=[json.loads(x) for x in (output/'runs.jsonl').read_text().splitlines()]
    summary={'records':len(rows),'valid':sum(r['valid'] for r in rows),'by_method':{},'comparisons':{}}
    for method in ('current_menu','mockturtle_measured'):
        rs=[r for r in rows if r['method']==method]
        summary['by_method'][method]={'runs':len(rs),'valid':sum(r['valid'] for r in rs),
              'median_ms':1000*statistics.median(r['wall_seconds'] for r in rs)}
    per=[]
    for name in sorted({r['name'] for r in rows}):
        rs=[r for r in rows if r['name']==name and r['valid']]
        if {r['method'] for r in rs}!={'current_menu','mockturtle_measured'}:continue
        for method in ('current_menu','mockturtle_measured'):
            vectors={json.dumps(r['resources'],sort_keys=True) for r in rs if r['method']==method}
            if len(vectors)!=1:raise ValueError('resource outcomes vary across repetitions; analyse distributions explicitly')
        a=next(r for r in rs if r['method']=='current_menu');b=next(r for r in rs if r['method']=='mockturtle_measured')
        per.append({'name':name,'cohort':a['cohort'],'current':a['resources'],'baseline':b['resources'],
                    'current_method':a['details'],'baseline_method':b['details']})
    summary['problems']=per
    for cohort in ('original','extension'):
        group=[x for x in per if x['cohort']==cohort];comparison={}
        for key in ('t_count','peak_aux','cnot','native_gates','native_depth','worst_case_ticks','measurement_rounds'):
            ds=[p['current'][key]-p['baseline'][key] for p in group]
            comparison[key]={'current_better':sum(d<0 for d in ds),'equal':ds.count(0),'baseline_better':sum(d>0 for d in ds)}
        summary['comparisons'][cohort]=comparison
    return summary


def verify(output):
    manifest=json.loads((output/'PROTOCOL.json').read_text())
    expected={(p['name'],m,r) for p in manifest['problems'] for m in manifest['methods'] for r in range(manifest['repetitions'])}
    if (output/'EVIDENCE_SHA256.json').exists():
        for rel,sha in json.loads((output/'EVIDENCE_SHA256.json').read_text()).items():
            if hashlib.sha256((output/rel).read_bytes()).hexdigest()!=sha:raise AssertionError('evidence hash mismatch: '+rel)
    rows=[json.loads(x) for x in (output/'runs.jsonl').read_text().splitlines()]
    keys=[(r['name'],r['method'],r['repeat']) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected):raise AssertionError('missing/duplicate benchmark records')
    problems={p['name']:CleanupProblem.from_manifest(p) for p in manifest['problems']}
    seen=set();pairs=0
    for row in rows:
        if not row['valid']:continue
        data=json.loads((output/'protocols'/f"{row['protocol_digest']}.json").read_text())
        p=CleanupProblem.from_manifest(data['problem']);c=PhaseProtocol.from_payload(data['protocol'])
        if p.digest!=row['problem_digest'] or p.digest!=problems[row['name']].digest:
            raise AssertionError('record is attached to the wrong problem')
        checked=verify_phase(p,c)
        if not checked['valid'] or c.digest!=row['protocol_digest'] or checked['resources']!=row['resources']:
            raise AssertionError('invalid saved circuit/resource result')
        if c.digest not in seen:
            independent=verify_truth_table_generators(p,c)
            if not independent['valid']:raise AssertionError('independent truth-table generator check failed')
            pairs+=independent['input_generator_pairs'];seen.add(c.digest)
        if row['method']=='mockturtle_measured':
            net=json.loads((output/'networks'/f"{row['details']['network_digest']}.json").read_text())
            if Network.parse(net['optimized']).polynomial()!=p.target_polynomial:raise AssertionError('wrong external function')
    result={'valid':True,'records':len(rows),'distinct_protocols':len(seen),'input_outcome_generator_pairs':pairs,
            'scope':'exact symbolic all-branch replay plus independent truth-table phase checks; no generator or policy called'}
    save(output/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mockturtle');parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--verify',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify(args.output_dir),indent=2))
    else:
        if not args.mockturtle:parser.error('--mockturtle is required for fresh execution')
        print(json.dumps(run(args.output_dir,args.mockturtle,args.repeats,resume=args.resume),indent=2))
