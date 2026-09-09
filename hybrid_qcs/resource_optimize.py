"""Deterministic bound tightening around learned discovery and exact audits.

Physical objectives are lexicographic and macro count is only a finiteness cap.
Every infeasibility claim is checked against the exact bound that was searched.
Incomplete discovery, audit or proof checking produces an upper bound/unknown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Callable

from .resource_audit import audit_bound, verify_closed_cover
from .resource_domain import BoundedProblem, CAP_FIELDS, OBJECTIVES, certify_witness, nonnegative_integer
from .resource_policy import BudgetedLinUCB, BudgetedSarsa
from .resource_search import WorkLimits, run_budgeted_search


@dataclass(frozen=True)
class OptimizationLimits:
    total_edges: int = 50_000
    discovery_edges: int = 1_000
    audit_edges: int = 20_000
    max_records: int = 20_000
    wall_seconds: float = 60.
    cpu_seconds: float = 60.
    max_rounds: int = 24

    def __post_init__(self):
        for field in ("total_edges", "discovery_edges", "audit_edges", "max_records", "max_rounds"):
            nonnegative_integer(getattr(self, field), field)
        WorkLimits(self.total_edges, self.max_records, self.wall_seconds, self.cpu_seconds)


class _Ledger:
    def __init__(self, limits, cancel):
        self.limits, self.cancel = limits, cancel
        self.start_wall, self.start_cpu = time.perf_counter(), time.process_time()
        self.edges = 0

    @property
    def wall(self):
        return time.perf_counter() - self.start_wall

    @property
    def cpu(self):
        return time.process_time() - self.start_cpu

    def remaining(self, edge_cap=None):
        edges = max(0, self.limits.total_edges - self.edges)
        if edge_cap is not None:
            edges = min(edges, edge_cap)
        return WorkLimits(edges, self.limits.max_records,
                          max(0., self.limits.wall_seconds - self.wall),
                          max(0., self.limits.cpu_seconds - self.cpu))

    def exhausted(self):
        if self.cancel and self.cancel():
            return "cancelled"
        if self.wall >= self.limits.wall_seconds:
            return "total_wall_limit"
        if self.cpu >= self.limits.cpu_seconds:
            return "total_cpu_limit"
        if self.edges >= self.limits.total_edges:
            return "total_edge_limit"
        return None


def optimize_resources(
    problem: BoundedProblem, *, objectives: tuple[str, ...] = OBJECTIVES,
    outer: BudgetedSarsa | None = None, inner: BudgetedLinUCB | None = None,
    scheduler: str = "hierarchy", limits: OptimizationLimits = OptimizationLimits(),
    initial_tokens: tuple[int, ...] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict:
    if not objectives or len(set(objectives)) != len(objectives) or any(o not in OBJECTIVES for o in objectives):
        raise ValueError("objectives must be a nonempty ordered subset without duplicates")
    if scheduler not in {"hierarchy", "outer", "distance", "cost"}:
        raise ValueError("unsupported scheduler")
    ledger = _Ledger(limits, cancel)
    journal, proofs, stages, trace = [], [], [], []
    current = problem
    incumbent = None
    discovery_seconds = audit_seconds = verification_seconds = 0.
    first_correct = time_to_best = None
    rounds = 0

    def remember(certificate, source):
        nonlocal incumbent, first_correct, time_to_best
        if not certificate or not certificate["success"]:
            raise AssertionError("uncertified candidate cannot become an incumbent")
        if incumbent is not None:
            old = tuple(incumbent["resources"][o] for o in objectives)
            new = tuple(certificate["resources"][o] for o in objectives)
            if new > old:
                raise AssertionError("bound tightening worsened the lexicographic incumbent")
        incumbent = certificate
        time_to_best = ledger.wall
        if first_correct is None:
            first_correct = time_to_best
        trace.append({"source": source, "at_edges": ledger.edges, "wall_seconds": ledger.wall,
                      "resources": certificate["resources"], "tokens": certificate["tokens"]})

    if initial_tokens is not None:
        seed = certify_witness(problem, initial_tokens)
        if not seed["success"]:
            raise ValueError("supplied initial witness fails the synthesis contract")
        remember(seed, "explicit_initial_witness")

    def finish(status, reason):
        return {
            "schema": "qcs-resource-optimization-v1", "status": status, "reason": reason,
            "problem": problem.manifest(), "problem_digest": problem.digest,
            "objective_order": list(objectives), "scheduler": scheduler,
            "limits": asdict(limits), "incumbent": incumbent,
            "stages": stages, "journal": journal, "infeasibility_certificates": proofs,
            "incumbent_trace": trace, "work_edges_including_verification": ledger.edges,
            "timing": {"wall_seconds": ledger.wall, "cpu_seconds": ledger.cpu,
                       "discovery_seconds": discovery_seconds, "audit_seconds": audit_seconds,
                       "proof_verification_seconds": verification_seconds,
                       "time_to_first_correct": first_correct, "time_to_best": time_to_best,
                       "time_to_first_optimal_circuit": time_to_best if status == "optimal" else None,
                       "time_to_complete_proof": ledger.wall if status in {"optimal", "infeasible"} else None},
            "claim": ("lexicographic optimum within this finite domain" if status == "optimal" else
                      "no feasible circuit within this finite domain" if status == "infeasible" else
                      "certified incumbent only; optimality unproved" if incumbent else
                      "no correctness or infeasibility conclusion"),
            "scope_warning": "not unrestricted Clifford+T optimality; no borrowed ancillas",
        }

    def unknown(reason):
        return finish("upper_bound" if incumbent is not None else "unknown", reason)

    for objective in objectives:
        stage = {"objective": objective, "lower_bound": 0,
                 "upper_bound": incumbent["resources"][objective] if incumbent else None,
                 "proved": False}
        stages.append(stage)
        while True:
            if incumbent is not None:
                value = incumbent["resources"][objective]
                stage["upper_bound"] = value
                if value == 0:
                    stage.update(proved=True, lower_bound=0, proof="nonnegative_integer_resource")
                    current = current.capped(objective, 0)
                    break
                cap = value - 1
            else:
                cap = getattr(current.budget, CAP_FIELDS[objective])
            trial = current.capped(objective, cap)
            reason = ledger.exhausted()
            if reason:
                return unknown(reason)
            if rounds >= limits.max_rounds:
                return unknown("round_limit")
            rounds += 1
            attempt = {"round": rounds, "objective": objective, "cap": cap,
                       "problem_digest": trial.digest}
            journal.append(attempt)
            discovery = run_budgeted_search(
                trial, outer=outer, inner=inner, objective=objective, scheduler=scheduler,
                incumbent=incumbent["resources"] if incumbent else None,
                limits=ledger.remaining(limits.discovery_edges), cancel=cancel)
            ledger.edges += discovery.edges
            discovery_seconds += discovery.wall_seconds
            attempt["discovery"] = {k: v for k, v in discovery.to_dict().items()
                                    if k not in {"witness", "proof", "rewards"}}
            if discovery.status == "feasible":
                remember(discovery.witness, "linear_policy_scheduler" if scheduler in {"hierarchy", "outer"} else scheduler)
                continue
            reason = ledger.exhausted()
            if reason:
                return unknown(reason)
            audited = audit_bound(trial, limits=ledger.remaining(limits.audit_edges), cancel=cancel)
            ledger.edges += audited.edges
            audit_seconds += audited.wall_seconds
            attempt["audit"] = {k: v for k, v in audited.to_dict().items()
                                if k not in {"witness", "proof", "rewards"}}
            if audited.status == "feasible":
                remember(audited.witness, "independent_audit")
                continue
            if audited.status != "infeasible":
                return unknown(audited.reason)
            checked = verify_closed_cover(audited.proof, trial, limits=ledger.remaining(), cancel=cancel)
            ledger.edges += checked["edges_checked"]
            verification_seconds += checked["wall_seconds"]
            attempt["proof_verification"] = checked
            if not checked["valid"]:
                return unknown("proof_unverified: " + checked["reason"])
            proof_index = len(proofs)
            proofs.append(audited.proof)
            attempt["certificate_index"] = proof_index
            stage["lower_bound"] = cap + 1
            if incumbent is None:
                return finish("infeasible", "independently_verified_empty_domain")
            if incumbent["resources"][objective] != cap + 1:
                raise AssertionError("proof bound does not meet incumbent upper bound")
            stage.update(proved=True, proof={"certificate_index": proof_index,
                                            "problem_digest": trial.digest})
            current = current.capped(objective, cap + 1)
            break
    return finish("optimal", "all_lexicographic_stages_proved")


def sweep_clean_ancillas(problem: BoundedProblem, *, budgets: tuple[int, ...] = (0, 1, 2), **kwargs) -> dict:
    """Independent contracts/archives per available-width budget; no width merges.

    Work limits apply separately and equally to each width, not to the whole sweep.
    The returned minimum concerns feasibility under the original finite bounds,
    not a claim that the target requires ancillas with unrestricted gate counts.
    """
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("ancilla budgets must be nonempty and unique")
    for value in budgets:
        nonnegative_integer(value, "ancilla budget")
    if "initial_tokens" in kwargs:
        raise ValueError("a fixed-width initial witness cannot be reused across ancilla contracts")
    rows = []
    for budget in sorted(budgets):
        per_width = replace(problem, clean_workspace=budget)
        result = optimize_resources(per_width, **kwargs)
        rows.append({"available_clean_workspace": budget, "result": result})
    feasible = [row for row in rows if row["result"]["incumbent"] is not None]
    minimum = None
    if feasible:
        candidate = min(row["available_clean_workspace"] for row in feasible)
        statuses = {row["available_clean_workspace"]: row["result"]["status"] for row in rows}
        if all(statuses.get(a) == "infeasible" for a in range(candidate)):
            minimum = candidate
    pareto = []
    for row in feasible:
        cert = row["result"]["incumbent"]
        vector = (row["available_clean_workspace"], *(cert["resources"][o] for o in OBJECTIVES))
        other_vectors = [(other["available_clean_workspace"],
                          *(other["result"]["incumbent"]["resources"][o] for o in OBJECTIVES))
                         for other in feasible if other is not row]
        if not any(all(x <= y for x, y in zip(v, vector)) and v != vector for v in other_vectors):
            pareto.append({"available_clean_workspace": row["available_clean_workspace"],
                           "used_clean_workspace": cert["used_clean_workspace"],
                           "resources": cert["resources"],
                           "auxiliary_if_phase_wrapped": cert["auxiliary_qubits_if_wrapped_as_phase_oracle"]})
    return {"schema": "qcs-clean-ancilla-sweep-v1", "rows": rows,
            "minimum_available_clean_workspace_proved": minimum,
            "minimum_scope": "feasibility within declared grammar and other finite resource bounds",
            "pareto_among_found_incumbents": pareto,
            "archive_policy": "separate exact archive for each complete contract",
            "work_budget_policy": "same declared total work limits per ancilla budget"}
