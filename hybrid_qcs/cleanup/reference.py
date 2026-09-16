"""Bank-free ANF compiler control, deliberately outside the materialization theorem.

Each logical phase monomial is realized using a chain of the SAME exact four-T
AND primitives, a Z/CZ consumer and reverse measured cleanup. Scratch is reused
between monomials. This is a straightforward reference, not a state-of-the-art
Boolean factorization solver. Its purpose is to expose cases where avoiding the
entire bank is better; no theorem workspace bound is assigned to this protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from .contract import CleanupProblem, plus, times, digest
from .ir import Instruction, Protocol, ResourceState
from .verify import primitive_receipts


@dataclass(frozen=True)
class ReferenceLayout:
    width:int


@dataclass(frozen=True)
class PhaseReference:
    problem_digest:str
    width:int
    ops:tuple[Instruction,...]
    primitive:str='and4'

    @property
    def layout(self):return ReferenceLayout(self.width)
    @property
    def digest(self):return digest(self.payload())
    def payload(self):
        return {'schema':'bank-free-phase-reference-v1','problem_digest':self.problem_digest,
                'width':self.width,'ops':[op.payload() for op in self.ops],'primitive':self.primitive}
    @classmethod
    def from_payload(cls,d):
        if d.get('schema')!='bank-free-phase-reference-v1':raise ValueError('wrong reference schema')
        obj=cls(d['problem_digest'],d['width'],tuple(Instruction.from_payload(o) for o in d['ops']),d['primitive'])
        if obj.payload()!=d:raise ValueError('extra reference fields')
        return obj
    def qasm3(self):return Protocol.qasm3(self)


def compile_bank_free(p:CleanupProblem)->PhaseReference:
    ops=[];degree=max((m.bit_count() for m in p.target_polynomial),default=0)
    width=p.n+max(0,degree-2);bit=0
    for mask in sorted(p.target_polynomial):
        qs=[q for q in range(p.n) if mask>>q&1]
        if not qs:ops.append(Instruction('MINUS',(0,),'consumer'));continue
        if len(qs)<=2:
            ops.append(Instruction('Z' if len(qs)==1 else 'CZ',tuple(qs),'consumer'));continue
        chain=[];left=qs[0]
        for j in range(len(qs)-2):
            target=p.n+j;operands=(left,qs[j+1],target)
            ops.append(Instruction('AND',operands,'prepare'));chain.append(operands);left=target
        ops.append(Instruction('CZ',(left,qs[-1]),'consumer'))
        for c,d,t in reversed(chain):
            ops.append(Instruction('MX',(t,),f'measure_reference_{bit}',outcome=bit))
            ops.append(Instruction('CZ',(c,d),f'correct_reference_{bit}',guard=bit));bit+=1
    return PhaseReference(p.digest,width,tuple(ops))


def verify_reference(p:CleanupProblem,prot:PhaseReference|dict)->dict:
    try:
        pr=PhaseReference.from_payload(prot) if isinstance(prot,dict) else prot
        if pr.problem_digest!=p.digest or type(pr.width)is not int or not p.n<=pr.width<=p.n+64:
            raise ValueError('reference contract mismatch')
        primitive_receipts(pr.primitive)
        values=[frozenset({1<<i}) for i in range(p.n)]+[frozenset()]*(pr.width-p.n)
        initial=values[:p.n];phase=frozenset();coeff={};rs=ResourceState.zero(pr.width)
        for op in pr.ops:
            if max(op.wires)>=pr.width:raise ValueError('reference wire outside layout')
            if op.kind=='AND':
                c,d,t=op.wires
                if values[t]:raise ValueError('unclean reference AND target')
                values[t]=times(values[c],values[d])
            elif op.kind=='MX':
                if op.outcome in coeff:raise ValueError('overwritten transcript')
                coeff[op.outcome]=values[op.wires[0]];values[op.wires[0]]=frozenset()
            elif op.kind in ('Z','CZ','MINUS'):
                term=frozenset({0}) if op.kind=='MINUS' else values[op.wires[0]] if op.kind=='Z' else times(values[op.wires[0]],values[op.wires[1]])
                if op.guard is None:phase=plus(phase,term)
                else:
                    if op.guard not in coeff:raise ValueError('missing reference outcome')
                    coeff[op.guard]=plus(coeff[op.guard],term)
            else:raise ValueError('not a supported reference instruction')
            rs=rs.append(op,pr.primitive,p)
        if phase!=p.target_polynomial or values[:p.n]!=initial or any(values[p.n:]) or any(coeff.values()):
            raise ValueError('reference branch identity or clean return failed')
        if not p.limits.accepts(rs.report()):raise ValueError('reference resource cap exceeded')
        return {'valid':True,'protocol_digest':pr.digest,'problem_digest':p.digest,'resources':rs.report(),
                'branch_count':1<<len(coeff),'all_outcomes_checked_symbolically':True,
                'source':'direct ANF monomial compiler with measured AND-chain cleanup',
                'scope':'bank-free reference, NOT covered by the materialization lower bound',
                'global_oracle_optimality':False}
    except (ValueError,TypeError,KeyError,IndexError)as e:return {'valid':False,'reason':str(e)}
