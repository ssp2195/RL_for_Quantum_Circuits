"""Unchanged logical benchmarks; only the native search representation changes.

Reference decompositions are a separate control and are never passed to the
learned search. A four-qubit Toffoli means C^3 X on FOUR logical qubits;
workspace, when present, is additional and explicitly counted.
"""
from __future__ import annotations
from dataclasses import replace
from itertools import product
from pathlib import Path
import json
import numpy as np
from .model import Gate, Budget, HybridState
from .ancilla_contract import AncillaContract, PhaseMode
from .certify import unitary_from_gates
from .native_domain import NativeProblem, phase_target, certify_native
from .ancilla_benchmarks import toffoli_decomposition, parity_phase_witness
from .qft_guided import qft_matrix, exact_qft_macros, swap_native


def contract(n,a=0,mode=PhaseMode.EXACT):
    return AncillaContract(n+a,tuple(range(n)),tuple(range(n,n+a)),phase_mode=mode)


def mcx_matrix(n):
    u=np.eye(1<<n,dtype=complex);mask=(1<<(n-1))-1
    a=mask;b=mask|(1<<(n-1));u[:,[a,b]]=u[:,[b,a]]
    return u


def swap_matrix(n,left=0,right=1):
    u=np.zeros((1<<n,1<<n),complex)
    for x in range(1<<n):
        bit=((x>>left)^(x>>right))&1
        u[x^(bit<<left)^(bit<<right),x]=1
    return u


def named_benchmarks():
    out=[]
    for n in (2,3):
        for a in (0,1,2):
            out.append(NativeProblem(f'qft-{n}-clean-{a}',contract(n,a),
                        Budget(18,20,64,64),qft_matrix(n),'QFT'))
    for n in (3,4):
        for a in (0,1,2):
            out.append(NativeProblem(f'toffoli-{n}-clean-{a}',contract(n,a),
                        Budget(24,24,64,64),mcx_matrix(n),'Toffoli'))
    for n in (2,3,4):
        out.append(NativeProblem(f'swap-{n}-q0-q{n-1}',contract(n),Budget(0,3,3,3),
                                 swap_matrix(n,0,n-1),'SWAP'))
    for a in (0,1,2):
        # Non-Clifford parity phase surrounded by H: not a computational-basis
        # phase-obligation problem in the active synthesis environment.
        u=unitary_from_gates(2,(Gate('H',(0,)),Gate('CNOT',(0,1)),Gate('T',(1,)),
                                  Gate('CNOT',(0,1)),Gate('H',(0,))))
        out.append(NativeProblem(f'mixed-axis-parity-clean-{a}',contract(2,a),
                                  Budget(3,6,12,12),u,'ancilla-mixed-Pauli'))
    return tuple(out)


def prior_phase_benchmarks(split='test'):
    """Read the already frozen phase specification; do not reroll the corpus.

    Coefficients are used only to form a diagonal target matrix. All native
    continuations, including H, remain enabled in each NativeProblem.
    """
    path=Path(__file__).resolve().parents[1]/'experiments/publication_v1/protocol.json'
    rows=json.loads(path.read_text())['split'][split]
    out=[]
    for row in rows:
        p=row['problem'];cs=p['coefficients']
        budget=Budget(sum(c%2 for _,c in cs),p['max_cnot'],p['max_gates'],p['max_depth'])
        target=phase_target(row['name'],p['n'],cs,p['ancillas'],constant=p['constant'],
                            budget=budget,max_t_depth=p['max_t_depth'])
        out.append(replace(target,split=split))
    return tuple(out)


def training_problems():
    # The target matrices are derived from short native words, but those words
    # are neither stored on NativeProblem nor given to either policy.
    out=[]
    words=[('H',),('S',),('SDG',),('T',),('TDG',),('H','T'),('T','H'),
           ('H','TDG'),('TDG','H'),('S','H','T'),('H','T','H','TDG'),
           ('H','S','T','H')]
    for i,word in enumerate(words):
        u=unitary_from_gates(1,[Gate(g,(0,)) for g in word])
        out.append(NativeProblem(f'native-train-1q-{i}',contract(1,mode=PhaseMode.PROJECTIVE),
                                  Budget(4,0,6,6),u,'mixed-Clifford-T','train'))
    for reverse in (False,True):
        c,t=(1,0) if reverse else (0,1)
        for phase in ('T','TDG','S'):
            word=(Gate('H',(c,)),Gate('CNOT',(c,t)),Gate(phase,(t,)))
            u=unitary_from_gates(2,word)
            for a in (0,1):
                out.append(NativeProblem(f'native-train-2q-{c}-{phase}-a{a}',contract(2,a),
                                         Budget(3,3,6,6),u,'entangling-native','train'))
    # Preserve original phase-only training targets in addition to true
    # basis-changing and noncommuting tasks. Their output path is not supplied.
    return tuple(out)+prior_phase_benchmarks('training')


def regression_problems(count=32):
    training=training_problems();seen=[np.eye(2,dtype=complex)]
    def equivalent(u,v):
        if u.shape!=v.shape:return False
        z=np.vdot(v,u);phase=z/abs(z) if abs(z)>1e-12 else 1
        return np.max(np.abs(u-phase*v))<1e-9
    for p in training:seen.append(p.unitary)
    out=[]
    for length in (2,3,4,5):
        for names in product(('H','T','TDG','S','SDG'),repeat=length):
            u=unitary_from_gates(1,[Gate(g,(0,)) for g in names])
            if any(equivalent(u,v) for v in seen):continue
            seen.append(u)
            out.append(NativeProblem(f'native-regression-{len(out):02d}',contract(1,mode=PhaseMode.PROJECTIVE),
                         Budget(5,0,7,7),u,'mixed-native-regression','regression'))
            if len(out)>=count:return tuple(out)
    raise RuntimeError('insufficient distinct regression unitaries')


def reference_gates(p):
    """Known constructive controls; intentionally separate from discovery."""
    n=len(p.contract.logical_qubits);anc=p.contract.clean_ancillas
    if p.family=='QFT':
        if n==3 and not anc:return None
        return tuple(g for m in exact_qft_macros(p.contract) for g in m.gates)
    if p.family=='Toffoli':
        if n==3:return toffoli_decomposition(0,1,2)
        if not anc:return None
        compute=toffoli_decomposition(0,1,anc[0])
        return (*compute,*toffoli_decomposition(anc[0],2,3),*compute)
    if p.family=='SWAP':return swap_native(0,n-1)
    if p.family=='ancilla-mixed-Pauli':
        middle=parity_phase_witness((0,1),anc[0]) if anc else (Gate('CNOT',(0,1)),Gate('T',(1,)),Gate('CNOT',(0,1)))
        return (Gate('H',(0,)),*middle,Gate('H',(0,)))
    return None


def reference_control(p):
    gates=reference_gates(p)
    if gates is None:
        return {'name':p.name,'source':'constructive_reference','status':'no_reference_for_this_contract'}
    state=HybridState.identity(p.width,p.budget)
    for gate in gates:
        state=state.apply(gate,partial_order_reduction=False)
        if state is None:raise AssertionError('reference violates declared caps')
    cert=certify_native(p,state,provenance='constructive_reference_not_RL')
    if not cert['success']:raise AssertionError('reference fails native isometry certification')
    return {'name':p.name,'status':'certified','certificate':cert}
