"""Independent exact replay over Z[omega, 1/2], omega=exp(i*pi/4).

This is a verifier/target-specification backend, NOT a frontier representation.
Native search still stores HybridState=(G,Theta,R,phi,rho). No numerical matrix
hash authorizes search pruning. Coefficients use the basis (1,omega,omega^2,
omega^3), with a common 2^denominator_power denominator and omega^4=-1.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import hashlib
import json
from typing import Iterable
import numpy as np

ZERO = (0, 0, 0, 0)
ONE = (1, 0, 0, 0)


def omega_times(a: tuple[int, ...], k: int) -> tuple[int, ...]:
    out = [0]*4
    for j, x in enumerate(a):
        power = (j+k) % 8
        out[power % 4] += x if power < 4 else -x
    return tuple(out)


def add(a, b):
    return tuple(x+y for x, y in zip(a, b, strict=True))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b, strict=True))


def conjugate(a):
    out = ZERO
    for j, x in enumerate(a):
        out = add(out, tuple(x*y for y in omega_times(ONE, -j)))
    return out


@dataclass(frozen=True)
class ExactMatrix:
    rows: tuple[tuple[tuple[int, ...], ...], ...]
    denominator_power: int = 0

    def __post_init__(self):
        rows = tuple(tuple(tuple(a) for a in row) for row in self.rows)
        if type(self.denominator_power) is not int or self.denominator_power < 0:
            raise ValueError('invalid exact denominator')
        if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError('invalid exact matrix shape')
        if any(len(a) != 4 or any(type(x) is not int for x in a) for r in rows for a in r):
            raise ValueError('invalid cyclotomic coefficients')
        power = self.denominator_power
        while power and all(x % 2 == 0 for r in rows for a in r for x in a):
            rows = tuple(tuple(tuple(x//2 for x in a) for a in r) for r in rows)
            power -= 1
        object.__setattr__(self, 'rows', rows)
        object.__setattr__(self, 'denominator_power', power)

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0])

    @classmethod
    def identity(cls, n):
        d = 1 << n
        return cls(tuple(tuple(ONE if i == j else ZERO for j in range(d)) for i in range(d)))

    @classmethod
    def from_payload(cls, data):
        if set(data) != {'schema', 'rows', 'denominator_power'} or data['schema'] != 'cyclotomic-matrix-v1':
            raise ValueError('invalid exact-target schema')
        return cls(data['rows'], data['denominator_power'])

    def payload(self):
        return {'schema': 'cyclotomic-matrix-v1', 'rows': self.rows,
                'denominator_power': self.denominator_power}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def numerical(self):
        basis = np.exp(1j*np.pi*np.arange(4)/4)
        return np.asarray([[sum(x*y for x,y in zip(a,basis)) for a in r] for r in self.rows]) / 2.**self.denominator_power

    def phase(self, k):
        return ExactMatrix(tuple(tuple(omega_times(a, k) for a in r) for r in self.rows), self.denominator_power)

    def dagger(self):
        return ExactMatrix(tuple(tuple(conjugate(self.rows[j][i]) for j in range(self.shape[0]))
                                 for i in range(self.shape[1])), self.denominator_power)

    def apply(self, name, qubits):
        """Left-apply one native gate using integer row operations only."""
        name = str(name).upper(); qs = tuple(qubits)
        n = self.shape[0].bit_length()-1
        if self.shape[0] != 1 << n or any(type(q) is not int or not 0 <= q < n for q in qs):
            raise ValueError('gate outside exact register')
        if name == 'CNOT':
            if len(qs) != 2 or qs[0] == qs[1]:
                raise ValueError('invalid CNOT operands')
            c,t = qs
            return ExactMatrix(tuple(self.rows[i ^ (((i>>c)&1)<<t)] for i in range(1<<n)), self.denominator_power)
        if name not in ('H','S','SDG','T','TDG') or len(qs) != 1:
            raise ValueError('not a native Clifford+T gate')
        q = qs[0]; out = list(self.rows)
        if name == 'H':
            for i in range(1<<n):
                if not i & (1<<q):
                    hi = i | (1<<q)
                    for index, sign in ((i, 1),(hi,-1)):
                        sums = [add(a,b) if sign == 1 else sub(a,b)
                                for a,b in zip(self.rows[i],self.rows[hi],strict=True)]
                        # 1/sqrt(2) = (omega - omega^3)/2.
                        out[index] = tuple(sub(omega_times(a,1),omega_times(a,3)) for a in sums)
            return ExactMatrix(tuple(out), self.denominator_power+1)
        k = {'T':1,'TDG':-1,'S':2,'SDG':-2}[name]
        for i in range(1<<n):
            if i & (1<<q):
                out[i] = tuple(omega_times(a,k) for a in self.rows[i])
        return ExactMatrix(tuple(out),self.denominator_power)

    def embed_output(self, contract):
        """Return J (U tensor I_borrowed), with clean wires restored."""
        if self.shape != (contract.logical_dimension,)*2:
            raise ValueError('logical exact matrix shape mismatch')
        out = [[ZERO]*contract.domain_dimension for _ in range(contract.physical_dimension)]
        n = contract.num_logical_qubits
        for column in range(contract.domain_dimension):
            logical_in = column & ((1<<n)-1)
            borrowed = column >> n
            for logical_out in range(1<<n):
                bits = logical_out | (borrowed << n)
                row = sum(((bits>>i)&1)<<q for i,q in enumerate(contract.domain_qubits))
                out[row][column] = self.rows[logical_out][logical_in]
        return ExactMatrix(tuple(tuple(r) for r in out),self.denominator_power)

    def permuted(self, order):
        n = len(order)
        if self.shape != (1<<n,1<<n) or sorted(order) != list(range(n)):
            raise ValueError('invalid qubit permutation')
        inds = [sum(((x>>j)&1)<<q for j,q in enumerate(order)) for x in range(1<<n)]
        return ExactMatrix(tuple(tuple(self.rows[i][j] for j in inds) for i in inds),self.denominator_power)


def exact_word(n: int, word: Iterable) -> ExactMatrix:
    u = ExactMatrix.identity(n)
    for gate in word:
        name, qs = (gate.name,gate.qubits) if hasattr(gate,'name') else gate
        u = u.apply(name,qs)
    return u


def orbit_digest(u: ExactMatrix) -> str:
    """Exact disjointness under wire permutation, dagger and eighth-root phase.

    This is corpus de-duplication only, not a complete Clifford conjugacy orbit.
    """
    n = u.shape[0].bit_length()-1
    if u.shape != (1<<n,)*2 or n>4:
        raise ValueError('orbit audit limited to four logical qubits')
    keys=[]
    for order in permutations(range(n)):
        p = u.permuted(order)
        for v in (p,p.dagger()):
            for k in range(8):
                keys.append(v.phase(k).digest)
    return min(keys)


def verify_exact_word(contract, exact_target, native):
    """Exact promised-input equality; no tolerance, policy or HybridState replay."""
    expected = exact_target.embed_output(contract)
    actual = ExactMatrix.identity(contract.num_logical_qubits).embed_output(contract)
    for name,qs in native:
        actual = actual.apply(name,qs)
    if contract.phase_mode.value == 'exact':
        valid = actual == expected; phases = [0] if valid else []
    else:
        # Scope intentionally only the scalar eighth roots, not an arbitrary
        # floating phase. All certificates in the new study are exact-phase.
        phases = [k for k in range(8) if actual == expected.phase(k)];valid=bool(phases)
    return {'valid':valid,'target_digest':exact_target.digest,'actual_digest':actual.digest,
            'scope':'exact cyclotomic promised-input equality','global_phase_eighth_roots':phases,
            'uses_floating_point_acceptance':False}


def exact_qft(n):
    if n not in (1,2,3):
        raise ValueError('exact Fourier target supported on one to three qubits')
    d=1<<n; rows=[]
    norm = ONE if n%2==0 else sub(omega_times(ONE,1),omega_times(ONE,3))
    for i in range(d):
        rows.append(tuple(omega_times(norm,(8//d)*i*j) for j in range(d)))
    return ExactMatrix(tuple(rows),(n+1)//2)


def exact_mcx(n):
    if n<2:raise ValueError('MCX requires a control and target')
    rows=list(ExactMatrix.identity(n).rows);a=(1<<(n-1))-1;b=a|(1<<(n-1))
    rows[a],rows[b]=rows[b],rows[a]
    return ExactMatrix(tuple(rows))


def determinant_obstruction(kind, n):
    """Known exact-phase determinant obstruction, not a novel lower bound.

    No statement about a register with clean/borrowed extra wires, measurement,
    or arbitrary projective phase is implied. Native determinant exponents are
    in Z_8. Fourier determinant: (-1)^(d(d-1)/2) i^((d-1)(d-2)/2).
    """
    if type(n) is not int or n<2:raise ValueError('at least two logical qubits')
    exponents = {'H':(4*(1<<(n-1)))%8,'S':(2*(1<<(n-1)))%8,
                 'SDG':(-2*(1<<(n-1)))%8,'T':(1<<(n-1))%8,
                 'TDG':(-(1<<(n-1)))%8,'CNOT':(4*(1<<(n-2)))%8}
    if kind=='QFT' and n in (2,3):
        d=1<<n; target=(4*d*(d-1)//2+2*(d-1)*(d-2)//2)%8
        exact=exact_qft(n)
    elif kind=='Toffoli' and n in (3,4):
        target=4;exact=exact_mcx(n)
    else:raise ValueError('unsupported analytic target')
    reachable={0}
    while True:
        new=reachable|{(x+k)%8 for x in reachable for k in exponents.values()}
        if new==reachable:break
        reachable=new
    excluded=target not in reachable
    # |det U-det V| <= d ||U-V||_2 <= d^2 ||U-V||_max for unitaries.
    # This is an explanatory separation, not a certified interval for a
    # floating target representation; the certificate below is exact-phase.
    return {'schema':'native-determinant-obstruction-v1','kind':kind,'logical_qubits':n,
            'clean_ancillas':0,'phase_mode':'exact','target_digest':exact.digest,
            'native_determinant_exponents':exponents,'reachable_exponents':sorted(reachable),
            'target_exponent':target,'excluded':excluded,
            'scope':'all finite words in native gate set, exact logical unitary, no ancillas',
            'reference':'Giles and Selinger, PRA 87, 032332 (2013)'}


def verify_determinant_certificate(certificate):
    try:
        expected=determinant_obstruction(certificate['kind'],certificate['logical_qubits'])
        return expected==certificate and expected['excluded']
    except (KeyError,ValueError,TypeError):return False


def verify_exact_cover(problem, exact_target, certificate, *, limits=None):
    """Check a native closed cover with an exact, not numerical, target predicate.

    The same native symbolic-key algebra remains in the trusted base. Target
    exclusion itself uses independent cyclotomic gate replay. No learned policy,
    dense target tolerance, or auditor queue is used to decide exclusion.
    """
    from .model import HybridState, Gate
    from .native_domain import archive_key,legal,next_t_depths,resources,digest
    from .native_audit import COVER_SCHEMA
    from .resource_search import WorkLimits,WorkMeter
    limits=WorkLimits(1000000,200000,60.,60.) if limits is None else limits
    meter=WorkMeter(limits);checked=0
    def fail(reason):return {'valid':False,'reason':reason,'checked_edges':meter.edges,'checked_records':checked}
    try:
        if problem.contract.phase_mode.value!='exact':return fail('exact-phase verifier only')
        if exact_target.shape!=(problem.contract.logical_dimension,)*2:return fail('exact target shape')
        if np.max(np.abs(exact_target.numerical()-problem.unitary))>1e-12:
            return fail('exact target is not the declared numerical embedding')
        if certificate['schema']!=COVER_SCHEMA or certificate['problem_digest']!=problem.digest:
            return fail('native cover domain mismatch')
        labels=certificate['labels']
        if len(labels)>limits.max_records:return fail('record limit')
        if certificate['digest']!=digest({'problem_digest':problem.digest,'labels':labels}):
            return fail('altered cover digest')
        expected=exact_target.embed_output(problem.contract);group={};states=[]
        for word in labels:
            s=HybridState.identity(problem.width,problem.budget);td=(0,)*problem.width
            actual=ExactMatrix.identity(problem.contract.num_logical_qubits).embed_output(problem.contract)
            for name,qs in word:
                if meter.reason():return fail(meter.reason())
                g=Gate(name,tuple(qs))
                if g not in problem.actions or not legal(problem,s,td,g):return fail('infeasible label')
                s=s.apply(g,partial_order_reduction=False);td=next_t_depths(td,g)
                actual=actual.apply(g.name,g.qubits);meter.edges+=1
            if actual==expected:return fail('target label in exclusion')
            checked+=1;states.append((s,td));group.setdefault(archive_key(problem,s),[]).append(resources(s,td))
        def covered(s,td):
            r=resources(s,td)
            return any(all(a<=b for a,b in zip(old,r,strict=True)) for old in group.get(archive_key(problem,s),[]))
        if not covered(HybridState.identity(problem.width,problem.budget),(0,)*problem.width):return fail('root uncovered')
        for s,td in states:
            for g in problem.actions:
                if meter.reason():return fail(meter.reason())
                meter.edges+=1
                if legal(problem,s,td,g) and not covered(s.apply(g,partial_order_reduction=False),next_t_depths(td,g)):
                    return fail('feasible child uncovered')
        if meter.reason():return fail(meter.reason())
        return {'valid':True,'schema':'exact-native-closed-cover-check-v1','checked_edges':meter.edges,
                'checked_records':checked,'exact_target_digest':exact_target.digest,
                'floating_point_target_exclusion':False,'shared_trusted_base':'HybridState algebra and exact archive key'}
    except (KeyError,ValueError,TypeError,IndexError,AssertionError):return fail('malformed exact cover')


def verify_exact_optimization(problem, exact_target, result):
    """Upgrade an eligible receipt by rechecking its witness and exclusions exactly.

    Numerical certificates are not automatically exact proofs. This separate
    routine must actually finish its integer-arithmetic target checks.
    """
    from .native_optimize import _replay,OPT_SCHEMA,OBJECTIVES
    if result.get('schema')!=OPT_SCHEMA or result.get('problem_digest')!=problem.digest:
        return {'valid':False,'reason':'optimization domain mismatch'}
    try:
        if problem.contract.phase_mode.value!='exact' or np.max(np.abs(exact_target.numerical()-problem.unitary))>1e-12:
            return {'valid':False,'reason':'exact target binding failed'}
        order=result['objectives']
        if not order or len(order)!=len(set(order)) or any(o not in OBJECTIVES for o in order):
            return {'valid':False,'reason':'invalid objective sequence'}
        replay=_replay(problem,result['witness'])
        if replay['resources']!=result['witness']['resources']:return {'valid':False,'reason':'forged witness costs'}
        witness_check=verify_exact_word(problem.contract,exact_target,result['witness']['native'])
        if not witness_check['valid']:return {'valid':False,'reason':'not an exact implementation'}
        current=problem;checks=[]
        for objective,stage in zip(order,result['stages'],strict=True):
            if stage['objective']!=objective or not stage['proved']:return {'valid':False,'reason':'incomplete objective'}
            k=replay['resources'][objective]
            if stage['lower_bound']!=k or stage['upper_bound']!=k:return {'valid':False,'reason':'forged stage cost'}
            proof=stage['proof']
            if proof['kind']=='nonnegative_integer_resource':
                if k!=0:return {'valid':False,'reason':'nonzero nonnegativity claim'}
                checks.append({'valid':True,'kind':'nonnegative_integer_resource'})
            elif proof['kind']=='checked_native_exclusion':
                if k<1:return {'valid':False,'reason':'invalid positive exclusion'}
                trial=current.cap(objective,k-1)
                if proof['trial_digest']!=trial.digest:return {'valid':False,'reason':'wrong k-1 subproblem'}
                check=verify_exact_cover(trial,exact_target,proof['certificate']);checks.append(check)
                if not check['valid']:return {'valid':False,'reason':check['reason']}
            else:return {'valid':False,'reason':'unknown proof rule'}
            current=current.cap(objective,k)
        return {'valid':True,'exact_optimality_verified':True,'witness':witness_check,'stages':checks,
                'exact_target_digest':exact_target.digest,
                'scope':'exact ideal target, same finite native grammar and all other resource caps'}
    except (KeyError,ValueError,TypeError,IndexError,AssertionError):return {'valid':False,'reason':'malformed exact receipt'}
