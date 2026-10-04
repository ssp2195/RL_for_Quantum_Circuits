"""Run the pinned Exact-T Library mapper, without changing its implementation.

Its output is a function evaluator, not automatically a clean phase oracle.
The measured compiler is compared separately. Here we explicitly form U^dag Z U
and certify the evaluator's output on all computational inputs over Z[omega,1/2].
This is a coherent-wrapper comparison, not a reproduction of the paper's whole
optimization flow and not a matched measurement-enabled optimality comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from ..native_exact import ZERO, ONE, omega_times, add, sub, conjugate
from .contract import CleanupProblem, evaluate
from .benchmarks import candidate_problems
from .benchmark_comparison import input_network, save, EXACT_T_COMMIT


def parse_qasm(text):
    register=re.search(r'qreg q\[(\d+)\]',text)
    if register is None:raise ValueError('missing quantum register')
    n=int(register.group(1))
    def indices(name):
        match=re.search(r'// '+name+r':\s*([\d,]+)',text)
        if match is None:raise ValueError('missing input/output mapping')
        return tuple(map(int,match.group(1).split(',')))
    ins,outs=indices('input_qubits'),indices('output_qubits')
    gates=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith(('//','OPENQASM','include','qreg','creg')):continue
        m=re.fullmatch(r'(h|s|sdg|t|tdg|z|x|cx)\s+q\[(\d+)\](?:,\s*q\[(\d+)\])?;',line)
        if not m:raise ValueError('unsupported upstream instruction: '+line)
        name,a,b=m.groups();qs=(int(a),) if b is None else (int(a),int(b))
        if len(qs)!=(2 if name=='cx' else 1) or len(set(qs))!=len(qs) or max(qs)>=n:raise ValueError('bad gate')
        gates.append((name,qs))
    if len(set(ins))!=len(ins) or len(outs)!=1 or any(q<0 or q>=n for q in ins+outs):raise ValueError('bad I/O')
    return n,ins,outs[0],tuple(gates)


def multiply(a,b):
    result=ZERO
    for j,x in enumerate(b):result=add(result,tuple(x*y for y in omega_times(a,j)))
    return result


def run_column(gates, basis: int, max_support: int=65536):
    state={basis:ONE};power=0;peak=1
    for name,qs in gates:
        mask=1<<qs[-1]
        if name=='cx':state={i^((bool(i&(1<<qs[0])))*mask):v for i,v in state.items()}
        elif name=='x':state={i^mask:v for i,v in state.items()}
        elif name=='h':
            out={}
            for i,v in state.items():
                w=sub(omega_times(v,1),omega_times(v,3))
                out[i&~mask]=add(out.get(i&~mask,ZERO),w)
                out[i|mask]=add(out.get(i|mask,ZERO),tuple(-x for x in w) if i&mask else w)
            state={i:v for i,v in out.items() if v!=ZERO};power+=1
            while power and all(all(x%2==0 for x in v) for v in state.values()):
                state={i:tuple(x//2 for x in v) for i,v in state.items()};power-=1
        else:
            k={'t':1,'tdg':-1,'s':2,'sdg':-2,'z':4}[name]
            state={i:omega_times(v,k) if i&mask else v for i,v in state.items()}
        peak=max(peak,len(state))
        if len(state)>max_support:raise RuntimeError('exact sparse-column limit exceeded')
    norm=ZERO
    for v in state.values():norm=add(norm,multiply(conjugate(v),v))
    if norm!=(1<<(2*power),0,0,0):raise AssertionError('exact norm mismatch')
    return state,peak


def certify_evaluator(p,text):
    width,ins,out,gates=parse_qasm(text)
    if len(ins)!=p.n:raise ValueError('input arity mismatch')
    peak=0
    for x in range(1<<p.n):
        basis=sum(((x>>j)&1)<<q for j,q in enumerate(ins))
        state,size=run_column(gates,basis);peak=max(peak,size)
        wanted=evaluate(p.target_polynomial,x)
        if any(((i>>out)&1)!=wanted for i in state):
            return {'valid':False,'input':x,'wanted':wanted,'scope':'exact output-bit test; cannot accept wrapper'}
    return {'valid':True,'basis_inputs':1<<p.n,'max_sparse_support':peak,
            'scope':'exact evaluator output eigenspace on all inputs; U^dag Z_output U gives exact phase and clean return by linearity'}


def wrapper_qasm(text):
    w,ins,out,gates=parse_qasm(text)
    inv={'h':'h','s':'sdg','sdg':'s','t':'tdg','tdg':'t','x':'x','z':'z','cx':'cx'}
    word=gates+(('z',(out,)),)+tuple((inv[g],qs) for g,qs in reversed(gates))
    prefix=f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{w}];\n'
    return prefix+''.join(g+' '+','.join(f'q[{q}]' for q in qs)+';\n' for g,qs in word), word


def counts(width,n,word):
    d=[0]*width;td=[0]*width;gates=tc=cx=0
    for name,qs in word:
        # Same native gate library: Z=S;S and X=H;S;S;H.
        local=('s','s') if name=='z' else ('h','s','s','h') if name=='x' else (name,)
        for name in local:
            a=1+max(d[q] for q in qs);b=int(name in ('t','tdg'))+max(td[q] for q in qs)
            for q in qs:d[q]=a;td[q]=b
            gates+=1;tc+=name in ('t','tdg');cx+=name=='cx'
    return {'t_count':tc,'cnot':cx,'native_gates':gates,'native_depth':max(d),
            't_depth':max(td),'peak_aux':width-n,'physical_qubits':width,
            'measurement_rounds':0,'worst_case_ticks':max(d),
            'workspace_scope':'allocated auxiliary register of complete reversible wrapper'}


def run(output: Path, binary: str):
    if output.exists() and any(output.iterdir()):raise ValueError('refuse overwrite')
    output.mkdir(parents=True,exist_ok=True)
    save(output/'PROTOCOL.json',{'upstream_commit':EXACT_T_COMMIT,'methods':['source','anf','davio'],
         'solver':['-n','6','--cut-size','5','--clean-ancilla'],'repetitions':1,
         'scope':'five development targets; complete coherent evaluator-adjoint wrapper; no hardware or measurement-matched superiority claim'})
    rows=[]
    for p in candidate_problems():
        for form in ('source','anf','davio'):
            tag=p.name+'-'+form;src=output/(tag+'.v');dest=output/(tag+'.qasm');stats=output/(tag+'-stats.json')
            src.write_text(input_network(p,form).verilog());start=time.perf_counter()
            args=[binary,'-n','6','--cut-size','5','--clean-ancilla','--stats-out',str(stats),'-o',str(dest),str(src)]
            try:
                proc=subprocess.run(args,text=True,capture_output=True,timeout=30,check=True)
                mapped=time.perf_counter()-start;text=dest.read_text();w,ins,out,gates=parse_qasm(text)
                check_start=time.perf_counter();check=certify_evaluator(p,text);verify_secs=time.perf_counter()-check_start
                wrapper,word=wrapper_qasm(text);(output/(tag+'-oracle.qasm')).write_text(wrapper)
                row={'name':p.name,'form':form,'problem':p.manifest(),'valid':check['valid'],'verification':check,
                     'mapping_seconds':mapped,'verification_seconds':verify_secs,'wrapper_resources':counts(w,p.n,word),
                     'reported_upstream_stats':json.loads(stats.read_text()),'stdout':proc.stdout,'stderr':proc.stderr}
            except (ValueError,RuntimeError,AssertionError,subprocess.SubprocessError) as e:
                row={'name':p.name,'form':form,'valid':False,'reason':str(e),'elapsed_seconds':time.perf_counter()-start}
            rows.append(row);save(output/'results.json',rows)
    save(output/'EVIDENCE_SHA256.json',{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(output.rglob('*')) if p.is_file() and p.name!='EVIDENCE_SHA256.json'})
    return {'jobs':len(rows),'accepted':sum(r['valid'] for r in rows)}

def verify(output: Path):
    """Recheck saved author-generated evaluators without rerunning the mapper."""
    manifest=json.loads((output/'PROTOCOL.json').read_text())
    if manifest['upstream_commit']!=EXACT_T_COMMIT:raise ValueError('unrecognized upstream source')
    for rel,sha in json.loads((output/'EVIDENCE_SHA256.json').read_text()).items():
        if hashlib.sha256((output/rel).read_bytes()).hexdigest()!=sha:
            raise AssertionError('evidence hash mismatch: '+rel)
    problems={p.name:p for p in candidate_problems()}
    rows=json.loads((output/'results.json').read_text())
    keys=[(r['name'],r['form']) for r in rows]
    expected={(name,form) for name in problems for form in manifest['methods']}
    if set(keys)!=expected or len(keys)!=len(expected):raise AssertionError('missing or duplicate outcomes')
    inputs=accepted=0
    for row in rows:
        if not row['valid']:continue
        p=CleanupProblem.from_manifest(row['problem'])
        if p.digest!=problems[row['name']].digest:raise AssertionError('wrong target contract')
        tag=row['name']+'-'+row['form']
        text=(output/(tag+'.qasm')).read_text()
        receipt=certify_evaluator(p,text)
        if not receipt['valid'] or receipt!=row['verification']:raise AssertionError('invalid evaluator')
        wrapper,word=wrapper_qasm(text);width,_,_,_=parse_qasm(text)
        if wrapper!=(output/(tag+'-oracle.qasm')).read_text():raise AssertionError('invalid complete wrapper')
        if counts(width,p.n,word)!=row['wrapper_resources']:raise AssertionError('wrong wrapper resources')
        accepted+=1;inputs+=receipt['basis_inputs']
    return {'valid':True,'records':len(rows),'accepted':accepted,'exact_basis_inputs':inputs,
            'scope':'exact all-input evaluator eigenspace plus literal inverse-wrapper verification; no external mapper called'}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--binary');a.add_argument('--verify',action='store_true')
    args=a.parse_args()
    if args.verify:result=verify(args.output_dir)
    else:
        if not args.binary:a.error('--binary is required for fresh execution')
        result=run(args.output_dir,args.binary)
    print(json.dumps(result,indent=2))
