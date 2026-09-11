"""Locked native mixed-axis corpus; construction words are separate references.

Splits are generated BEFORE evaluation, without calling any search scheduler.
Exact orbit de-duplication excludes eighth-root phase, adjoint and qubit
permutation copies across training/validation/test/regression. No witness is
stored on the NativeProblem presented to a policy.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
from .model import Gate, Budget, generate_gates
from .native_benchmarks import contract, named_benchmarks, prior_phase_benchmarks
from .native_domain import NativeProblem
from .native_exact import ExactMatrix, exact_word, orbit_digest, exact_qft, exact_mcx, omega_times, ONE, ZERO
from .native_optimality_runner import problem_from_manifest

STUDY_SCHEMA = 'native-publication-study-v1'
SEEDS = (11, 19, 23, 31, 47)
# Gate lengths refer to specification constructors, NOT minimum lengths.
SPLIT_DESIGN = {
    'training': ((1,12,2,5),(2,18,3,5),(3,6,4,5)),
    'validation': ((1,4,3,6),(2,6,3,6),(3,2,4,6)),
    'test': ((1,8,4,7),(2,16,3,6),(3,6,4,7)),
    'regression': ((1,12,3,6),(2,12,3,4)),
}


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n')
    tmp.replace(path)


def connected(word,n):
    if n==1:return True
    seen={0}
    while True:
        old=set(seen)
        for g in word:
            if g.name=='CNOT' and set(g.qubits)&seen:seen.update(g.qubits)
        if seen==old:return len(seen)==n


def operator_entangles(u,n):
    """Structural filter only: non-product operator on every single-wire cut."""
    if n==1:return True
    for q in range(n):
        order=(q,)+tuple(j for j in range(n) if j!=q)
        inds=[sum(((x>>j)&1)<<r for j,r in enumerate(order)) for x in range(1<<n)]
        v=u[np.ix_(inds,inds)]
        realigned=v.reshape(1<<(n-1),2,1<<(n-1),2).transpose(1,3,0,2).reshape(4,-1)
        if np.linalg.matrix_rank(realigned,tol=1e-10)<2:return False
    return True


def make_corpus(seed=20260911):
    rng=np.random.default_rng(seed);seen=set();splits={};references={}
    # Exclude every <=2-gate one-qubit target and the old <=2-gate entanglers.
    # Not a performance filter: minimum length is not inferred beyond this set.
    for n in (1,2,3):
        seen.add(orbit_digest(ExactMatrix.identity(n)))
        if n<=2:
            gs=generate_gates(n)
            for g in gs:
                seen.add(orbit_digest(exact_word(n,[g])))
                for h in gs:
                    seen.add(orbit_digest(exact_word(n,[g,h])))
    for split,design in SPLIT_DESIGN.items():
        out=[]
        for n,count,lo,hi in design:
            attempts=0;accepted=0;gs=generate_gates(n)
            while accepted<count:
                attempts+=1
                if attempts>30000:raise RuntimeError('could not form disjoint native corpus')
                length=int(rng.integers(lo,hi+1))
                word=[]
                while len(word)<length:
                    g=gs[int(rng.integers(len(gs)))]
                    if word and g.qubits==word[-1].qubits and (
                        (g.name,word[-1].name) in (('H','H'),('CNOT','CNOT'),('T','TDG'),('TDG','T'),('S','SDG'),('SDG','S'))):
                        continue
                    word.append(g)
                if not any(g.name=='H' for g in word) or not any(g.is_non_clifford for g in word):continue
                if not connected(word,n):continue
                exact=exact_word(n,word);u=exact.numerical()
                if not operator_entangles(u,n):continue
                orbit=orbit_digest(exact)
                if orbit in seen:continue
                seen.add(orbit);accepted+=1
                name=f'{split}-n{n}-{accepted:02d}'
                # Upper bounds are determined by the generating specification,
                # with one gate of slack. No solver or witness seeds discovery.
                nt=sum(g.is_non_clifford for g in word);nc=sum(g.is_two_qubit for g in word)
                p=NativeProblem(name,contract(n),Budget(nt+1,nc+1 if n>1 else 0,length+1,length+1),u,
                                'mixed-native',split,max_t_depth=nt+1)
                row={'name':name,'orbit':orbit,'n':n,'generator_length':length,'problem':p.manifest(),
                     'exact_target':exact.payload()}
                out.append(row);references[name]=[[g.name,list(g.qubits)] for g in word]
        splits[split]=out
    return splits,references



def exact_benchmark_target(p):
    n=p.contract.num_logical_qubits
    if p.family=='QFT':return exact_qft(n)
    if p.family=='Toffoli':return exact_mcx(n)
    if p.family=='SWAP':
        rows=[]
        for i in range(1<<n):
            j=i^((1|(1<<(n-1))) if ((i&1)^((i>>(n-1))&1)) else 0)
            rows.append(tuple(ONE if k==j else ZERO for k in range(1<<n)))
        return ExactMatrix(tuple(rows))
    if p.family=='ancilla-mixed-Pauli':
        return exact_word(2,[Gate('H',(0,)),Gate('CNOT',(0,1)),Gate('T',(1,)),Gate('CNOT',(0,1)),Gate('H',(0,))])
    source=Path(__file__).resolve().parents[1]/'experiments/publication_v1/protocol.json'
    row=next(r for r in json.loads(source.read_text())['split']['test'] if r['name']==p.name)['problem']
    exponents=[(row['constant']+sum(c*((x&m).bit_count()%2) for m,c in row['coefficients']))%8 for x in range(1<<n)]
    return ExactMatrix(tuple(tuple(omega_times(ONE,exponents[i]) if i==j else ZERO for j in range(1<<n)) for i in range(1<<n)))


def lock(output):
    output=Path(output)
    if (output/'protocol.json').exists():
        protocol=json.loads((output/'protocol.json').read_text())
        if protocol.get('schema')!=STUDY_SCHEMA:raise ValueError('incompatible existing protocol')
        return protocol
    splits,refs=make_corpus()
    # Every named/historical target remains a separate, unchanged challenge case.
    retained=[{'name':p.name,'family':p.family,'problem':p.manifest(),'exact_target':exact_benchmark_target(p).payload()}
              for p in named_benchmarks()+prior_phase_benchmarks()]
    protocol={'schema':STUDY_SCHEMA,'generator_seed':20260911,'splits':splits,'retained':retained,
              'primary_seeds':list(SEEDS),'training_stages':[64,96,24],
              'training_edge_limit':128,'training_seconds_per_episode':1.5,
              'training_cpu_cap_all_fits':1800.,
              'selection':{'seeds':[101],'alpha_candidates':[0.003,0.01,0.03],
                           'criterion':'validation success, then lower gates, then fewer attempted edges; alpha smallest on ties',
                           'edges':512,'seconds_guard':15.},
              'primary_budget_axes':{'edges':[128,1024],'wall_seconds':[0.2,1.0]},
              'methods':['hierarchy','untrained','outer','inner','greedy','uniform_cost','mitm'],
              'timing_repetitions':2,'edge_repetitions':1,
              'deterministic_controls_seeds':[11],
              'primary_endpoint':{'axis':'wall_seconds','budget':1.0,'metric':'failure-aware native-gate resource score'},
              'objective':'native gate count under all fixed native T/CNOT/depth/T-depth bounds',
              'discovery_audit':False,'proofs':'separate post-discovery sample, charged separately',
              'no_witness_inputs':True,'corpus_filter':'syntactic mixed-axis/connected and exact orbit exclusion, never observed search performance',
              'clustering':'logical target; seeds, timing repetitions and ancilla widths are not independent targets',
              'exploratory_ablations':['no_inner_bootstrap','frontier_inner_bootstrap','no_structure'],
              'ablation_seeds':[11,23,47],
              'secondary_seconds':3.,'ancilla_budgets':[0,1,2],
              'reachability':'determinant classifications are reported separately, never inferred from timeout'}
    dump(output/'protocol.json',protocol)
    dump(output/'construction_references.json',{'scope':'not accessible to discovery or training','words':refs})
    checksum=hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest()
    (output/'PROTOCOL_SHA256').write_text(checksum+'\n')
    return protocol


def load_problem(row, ancillas=0):
    p=problem_from_manifest(row['problem'])
    if ancillas:
        n=len(p.contract.logical_qubits)
        p=replace(p,contract=contract(n,ancillas))
    return replace(p,name=row['name'],family=row.get('family','mixed-native'))
