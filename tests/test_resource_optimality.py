from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hybrid_qcs.model import Budget
from hybrid_qcs.oracle_synthesis import BooleanOracleSpec, OracleLayout, oracle_macro_library
from hybrid_qcs.optimality import (
    _phase_gates_for_coefficient, native_unitary, oracle_macro_grammar,
    prove_marked_evaluator_lexicographic_optimum,
)
from hybrid_qcs.resource_domain import (
    BoundedProblem, ExactImage, Resources, canonical_digest, certify_witness, clean_mask,
    dominates, tokens_from_names, transition,
)
from hybrid_qcs.resource_policy import (
    BudgetedLinUCB, BudgetedSarsa, PolicyContext, OUTER_FEATURE_NAMES,
    outer_features, inner_features, load_hierarchy, save_hierarchy,
)
from hybrid_qcs.resource_search import (
    DeferredResourceSearch, WorkLimits, run_budgeted_search, train_budgeted_hierarchy,
)
from hybrid_qcs.resource_audit import audit_bound, reference_successor, verify_closed_cover, problem_from_manifest
from hybrid_qcs.resource_optimize import OptimizationLimits, optimize_resources, sweep_clean_ancillas


def problem(truth=(0, 1), *, work=0, mode="evaluator", operations=4, budget=None):
    n = (len(truth) - 1).bit_length()
    return BoundedProblem(BooleanOracleSpec("test", n, tuple(truth)),
                          budget or Budget(7, 6, 16, 16), operations, work, mode)


def certificate(p, names):
    return certify_witness(p, tokens_from_names(p, names))


def rehash(proof):
    proof["payload_sha256"] = canonical_digest({k: v for k, v in proof.items() if k != "payload_sha256"})
    return proof


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_operation_and_ancilla_caps(value):
    with pytest.raises(ValueError):
        problem(operations=value)
    with pytest.raises(ValueError):
        problem(work=value)


def test_reject_unsupported_domain_and_width():
    with pytest.raises(ValueError):
        problem(work=3)
    with pytest.raises(ValueError):
        problem(mode="borrowed")
    with pytest.raises(ValueError):
        problem().capped("estimated_cost", 3)


@pytest.mark.parametrize("mode", ["evaluator", "direct_phase"])
@pytest.mark.parametrize("work", [0, 1, 2])
def test_manifest_binds_width_grammar_target_phase_and_bounds(mode, work):
    p = problem(mode=mode, work=work)
    assert problem_from_manifest(p.manifest()).digest == p.digest
    assert p.width == 1 + work + (mode == "evaluator")
    assert p.capped("depth", 2).digest != p.digest
    assert replace(p, spec=BooleanOracleSpec("other", 1, (1, 0))).digest != p.digest


@pytest.mark.parametrize("mode", ["evaluator", "direct_phase"])
def test_reference_transition_and_wire_costs_agree(mode):
    p = problem(mode=mode, work=1, truth=(0, 1, 1, 0))
    image, resource = p.root, Resources.zero(p.width)
    for token, op in enumerate(p.operations):
        ref_image, ref_resource = reference_successor(p, image, resource, token)
        assert ref_image == transition(image, op)
        assert ref_resource == resource.append(op)
        image, resource = ref_image, ref_resource


def test_clean_workspace_must_be_zero_for_every_input_and_is_reusable():
    p = problem(work=1, budget=Budget(0, 4, 10, 10))
    compute, use, uncompute = tokens_from_names(p, ("CNOT(0,2)", "CNOT(2,1)", "CNOT(0,2)"))
    computed = transition(p.root, p.operations[compute])
    assert computed.mapping[0] == 0  # One zero sample is not sufficient.
    assert clean_mask(p, computed) == 0
    restored = transition(transition(computed, p.operations[use]), p.operations[uncompute])
    assert clean_mask(p, restored) == 1 << 2
    good = certify_witness(p, (compute, use, uncompute))
    assert good["success"] and good["dag_validated"]
    assert good["native_isometry_error"] < 1e-12
    assert good["used_clean_workspace"] == 1
    assert good["auxiliary_qubits_if_wrapped_as_phase_oracle"] == 2
    bad = certify_witness(p, (compute, use))
    assert not bad["success"] and bad["workspace_leakage"] > 0


def test_direct_phase_has_no_hidden_flag():
    p = problem(mode="direct_phase", budget=Budget(0, 0, 2, 2))
    cert = certificate(p, ("S(0)", "S(0)"))
    assert cert["success"]
    assert cert["required_logical_output_bits"] == 0
    assert cert["auxiliary_qubits_if_wrapped_as_phase_oracle"] == 0
    assert cert["additional_decomposition_scratch"] == 0


def test_literal_emitted_t_cost_does_not_shrink_after_cancellation():
    p = problem((0, 0), mode="direct_phase")
    cert = certificate(p, ("T(0)", "TDG(0)"))
    assert cert["success"] and cert["resources"]["t_count"] == 2
    assert not certificate(p.capped("t_count", 1), ("T(0)", "TDG(0)"))["success"]


def test_phase_mode_does_not_merge_global_minus_identity():
    p = problem((1, 1), mode="direct_phase", operations=6)
    image = p.root
    for token in tokens_from_names(p, ("S(0)", "S(0)", "X(0)", "S(0)", "S(0)", "X(0)")):
        image = transition(image, p.operations[token])
    assert image == p.goal and image != p.root


def test_native_certifier_checks_both_flag_inputs():
    p = problem((0, 0, 0, 1), operations=1, budget=Budget(7, 6, 15, 15))
    cert = certificate(p, ("TOFFOLI(0,1,2)",))
    assert cert["success"] and cert["native_isometry_error"] < 1e-12
    assert cert["resources"]["t_count"] == 7


@pytest.mark.parametrize("token", [True, -1, 9999, 0.5])
def test_invalid_witness_tokens_rejected(token):
    with pytest.raises(ValueError):
        certify_witness(problem(), (token,))


def test_per_wire_depth_is_required_for_sound_dominance():
    left = Resources(0, 0, 1, (1, 0), 1)
    right = Resources(0, 0, 1, (0, 1), 1)
    assert left.depth == right.depth
    assert not dominates(left, right) and not dominates(right, left)


def test_archive_retains_incomparable_resource_labels():
    p = problem()
    env = DeferredResourceSearch(p, WorkLimits())
    image = transition(p.root, p.operations[0])
    a = env.insert(image, Resources(0, 0, 1, (1, 0), 1), 0, 0)
    b = env.insert(image, Resources(0, 0, 1, (0, 1), 1), 0, 0)
    assert a and b and len(env.archive[image]) == 2


@pytest.mark.parametrize("scheduler", ["hierarchy", "outer", "distance", "cost"])
def test_all_schedulers_discover_but_do_not_declare_optimality(scheduler):
    result = run_budgeted_search(problem(budget=Budget(0, 1, 1, 1)), scheduler=scheduler)
    assert result.status == "feasible" and result.witness["success"]
    assert result.proof is None
    assert result.profile["full_dag_policy_reconstructions"] == 0


@pytest.mark.parametrize("reason,kwargs", [
    ("edge_limit", {"limits": WorkLimits(0)}),
    ("wall_limit", {"limits": WorkLimits(wall_seconds=0)}),
    ("cpu_limit", {"limits": WorkLimits(cpu_seconds=0)}),
    ("cancelled", {"cancel": lambda: True}),
    ("record_limit", {"limits": WorkLimits(max_records=1)}),
])
def test_limits_never_imply_infeasibility(reason, kwargs):
    p = problem()
    discovery = run_budgeted_search(p, **kwargs)
    audit = audit_bound(p, **kwargs)
    assert discovery.status == audit.status == "unknown"
    assert discovery.reason == audit.reason == reason
    assert discovery.proof is audit.proof is None


def test_exhausted_discovery_requires_independent_audit():
    p = problem(budget=Budget(0, 0, 4, 4))
    result = run_budgeted_search(p)
    assert result.status == "unknown"
    assert result.reason == "frontier_exhausted_requires_audit"
    audit = audit_bound(p)
    assert audit.status == "infeasible"
    assert verify_closed_cover(audit.proof, p)["valid"]


@pytest.mark.parametrize("mode", ["evaluator", "direct_phase"])
@pytest.mark.parametrize("truth", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_audit_agrees_with_unpruned_enumeration(mode, truth):
    p = problem(truth, mode=mode, operations=2, budget=Budget(1, 1, 8, 8))
    reachable = [(p.root, Resources.zero(p.width))]
    hit = p.root == p.goal
    for _ in range(p.max_operations):
        children = []
        for image, resource in reachable:
            for op in p.operations:
                new_cost = resource.append(op)
                if new_cost.within(p):
                    successor = transition(image, op)
                    hit |= successor == p.goal
                    children.append((successor, new_cost))
        reachable = children
    result = audit_bound(p)
    assert (result.status == "feasible") == hit
    if not hit:
        assert verify_closed_cover(result.proof, p)["valid"]


def test_certificate_rejects_changed_scope_even_with_new_hash():
    p = problem(budget=Budget(0, 0, 4, 4))
    proof = audit_bound(p).proof
    assert not verify_closed_cover(proof, problem())["valid"]
    tampered = deepcopy(proof)
    tampered["problem"]["budget"]["max_cnot_count"] = 1
    tampered["problem_digest"] = canonical_digest(tampered["problem"])
    assert not verify_closed_cover(rehash(tampered))["valid"]


def test_certificate_rejects_missing_successor_and_root():
    p = problem(budget=Budget(0, 0, 4, 4))
    proof = audit_bound(p).proof
    assert len(proof["cover"]) > 1
    missing = deepcopy(proof)
    missing["cover"] = [row for row in missing["cover"] if row["mapping"] == list(p.root.mapping)]
    assert verify_closed_cover(rehash(missing))["reason"] == "successor_not_covered"
    missing["cover"] = [row for row in proof["cover"] if row["mapping"] != list(p.root.mapping)]
    assert verify_closed_cover(rehash(missing))["reason"] == "root_not_covered"


def test_certificate_checksum_timeout_and_grammar_tampering():
    p = problem(budget=Budget(0, 0, 4, 4))
    proof = audit_bound(p).proof
    bad = deepcopy(proof)
    bad["claim"] = "timeout"
    assert not verify_closed_cover(bad)["valid"]
    bad = deepcopy(proof)
    bad["cover"][0]["resources"]["gate_count"] += 1
    assert verify_closed_cover(bad)["reason"] == "payload_digest_mismatch"
    bad = deepcopy(proof)
    bad["problem"]["grammar"].pop()
    bad["problem_digest"] = canonical_digest(bad["problem"])
    assert not verify_closed_cover(rehash(bad))["valid"]
    assert not verify_closed_cover(proof, limits=WorkLimits(0))["valid"]


def test_budget_interactions_change_candidate_relative_features():
    p = problem(budget=Budget(7, 6, 20, 20), work=1)
    env = DeferredResourceSearch(p, WorkLimits())
    token = tokens_from_names(p, ("CNOT(0,2)",))[0]
    child = env.step(0, token)
    root = env.records[0]
    context = PolicyContext("cnot_count", .8)
    delta = outer_features(p, child, context) - outer_features(p, root, context)
    tight = p.capped("cnot_count", 2)
    new_delta = outer_features(tight, child, context) - outer_features(tight, root, context)
    assert not np.allclose(delta, new_delta)
    assert np.isfinite(inner_features(p, root, token, context)).all()


def test_cached_features_refresh_incumbent_work_and_pending_context():
    p = problem()
    env = DeferredResourceSearch(p, WorkLimits())
    root = env.records[0]
    context = PolicyContext("t_count", 1., {"t_count": 7})
    first = outer_features(p, root, context)
    root.pending &= root.pending - 1
    changed = outer_features(p, root, PolicyContext("t_count", .1, {"t_count": 3}))
    assert not np.array_equal(first, changed)
    assert np.isfinite(changed).all()


@pytest.mark.parametrize("max_edges", [1, 2, 20])
def test_terminal_potential_zero_and_shaping_telescopes(max_edges):
    p = problem((1, 0), budget=Budget(0, 1, 5, 5))
    result = run_budgeted_search(p, limits=WorkLimits(max_edges), scheduler="cost", learn="outer")
    assert result.rewards
    assert result.rewards[-1]["terminal"]
    assert result.rewards[-1]["potential_after"] == 0.
    initial = result.rewards[0]["potential_before"]
    assert sum(x["shaping"] for x in result.rewards) == pytest.approx(-4 * initial)
    # Classical work is attempted edges, NOT native T-count over explored paths.
    hit_reward = 5 if result.status == "feasible" else 0
    assert sum(x["base_reward"] for x in result.rewards) == pytest.approx(hit_reward - .01 * result.edges)


def test_staged_training_checkpoint_and_frozen_evaluation(tmp_path):
    curriculum = (problem(), problem(mode="direct_phase"), problem(work=1))
    outer, inner, logs = train_budgeted_hierarchy(curriculum, episodes=(2, 2, 1), limits=WorkLimits(12, 100, 3., 3.))
    assert len(logs) == 5 and outer.updates > 0 and inner.updates > 0
    saved_weights = outer.weights.copy()
    saved_responses = {k: v.copy() for k, v in inner.responses.items()}
    save_hierarchy(tmp_path / "policy.json", outer, inner)
    restored, bandit = load_hierarchy(tmp_path / "policy.json")
    assert np.array_equal(restored.weights, saved_weights)
    run_budgeted_search(problem(), outer=outer, inner=inner, limits=WorkLimits(20))
    assert np.array_equal(outer.weights, saved_weights)
    assert all(np.array_equal(inner.responses[k], saved_responses[k]) for k in saved_responses)
    outer.weights[:] = 0
    save_hierarchy(tmp_path / "zero.json", outer, inner)
    assert not np.any(load_hierarchy(tmp_path / "zero.json")[0].weights)
    data = json.loads((tmp_path / "policy.json").read_text())
    data["schema"] = "legacy"
    (tmp_path / "bad.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_hierarchy(tmp_path / "bad.json")


def test_iterative_bound_tightening_improves_explicit_seed():
    p = problem(work=1, budget=Budget(0, 4, 8, 8))
    seed = tokens_from_names(p, ("CNOT(0,2)", "CNOT(2,1)", "CNOT(0,2)"))
    result = optimize_resources(p, initial_tokens=seed, scheduler="distance")
    assert result["status"] == "optimal"
    assert result["incumbent"]["resources"]["cnot_count"] == 1
    assert result["incumbent_trace"][0]["source"] == "explicit_initial_witness"
    assert all(stage["proved"] for stage in result["stages"])
    assert all(verify_closed_cover(proof)["valid"] for proof in result["infeasibility_certificates"])
    assert result["timing"]["time_to_complete_proof"] >= result["timing"]["time_to_first_correct"]


def test_timeout_keeps_seed_as_upper_bound_not_optimum():
    p = problem(work=1)
    seed = tokens_from_names(p, ("CNOT(0,2)", "CNOT(2,1)", "CNOT(0,2)"))
    result = optimize_resources(p, initial_tokens=seed, limits=OptimizationLimits(total_edges=0))
    assert result["status"] == "upper_bound"
    assert result["timing"]["time_to_first_optimal_circuit"] is None
    assert not result["infeasibility_certificates"]


def test_optimization_proves_infeasibility_without_incumbent():
    p = problem(budget=Budget(0, 0, 4, 4))
    result = optimize_resources(p)
    assert result["status"] == "infeasible" and result["incumbent"] is None
    assert result["journal"][-1]["proof_verification"]["valid"]


def test_ancilla_sweep_separates_archives_and_does_not_infer_from_timeout():
    p = problem(budget=Budget(0, 1, 1, 1), operations=1)
    sweep = sweep_clean_ancillas(p, budgets=(0, 1, 2), scheduler="distance")
    assert sweep["minimum_available_clean_workspace_proved"] == 0
    assert len({row["result"]["problem_digest"] for row in sweep["rows"]}) == 3
    assert all(row["result"]["incumbent"]["used_clean_workspace"] == 0 for row in sweep["rows"])
    unknown = sweep_clean_ancillas(p, budgets=(0, 1), limits=OptimizationLimits(total_edges=0))
    assert unknown["minimum_available_clean_workspace_proved"] is None


@pytest.mark.parametrize("coefficient", range(8))
def test_all_phase_coefficients_lower_exactly(coefficient):
    gates = _phase_gates_for_coefficient(coefficient, 0)
    actual = native_unitary(1, gates)
    expected = np.diag([1., np.exp(1j * np.pi * coefficient / 4)])
    assert np.allclose(actual, expected, atol=1e-12)
    assert sum(name in {"T", "TDG"} for name, _ in gates) == coefficient % 2


def test_legacy_auditor_and_learned_search_use_same_grammar_and_costs():
    exact = oracle_macro_grammar()
    learned = oracle_macro_library(OracleLayout.standard(3, 1))
    assert [(x.kind, x.qubits) for x in exact] == [(x.family, x.qubits) for x in learned]
    for macro, action in zip(exact, learned, strict=True):
        assert macro.resources.native_gate_count == len(action.native_gates)
        assert macro.resources.t_count == sum(g.is_non_clifford for g in action.native_gates)
        assert macro.resources.cnot_count == sum(g.is_two_qubit for g in action.native_gates)


def test_corrected_evaluator_optimum_has_independent_full_flag_certificate():
    proof = prove_marked_evaluator_lexicographic_optimum("100")
    p = problem((0, 1, 0, 0, 0, 0, 0, 0), work=1, operations=6, budget=Budget(21, 21, 60, 60))
    cert = certificate(p, proof.witness)
    assert cert["success"]
    assert cert["resources"]["cnot_count"] == 19
    assert cert["resources"]["gate_count"] == 54


def test_cli_optimize_export_and_independent_verify(tmp_path, capsys):
    from hybrid_qcs.resource_runner import main
    assert main(["optimize", "--truth-table", "01", "--ancillas", "0", "--t-cap", "0",
                 "--cnot-cap", "1", "--gate-cap", "1", "--depth-cap", "1",
                 "--operation-cap", "1", "--output-dir", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["rows"][0]["result"]["status"] == "optimal"
    files = list((tmp_path / "certificates").glob("*.json"))
    assert files
    assert main(["verify", *(str(p) for p in files)]) == 0
    assert main(["verify", str(files[0]), "--max-edges", "0"]) == 2


def test_cli_rejects_invalid_truth_table():
    from hybrid_qcs.resource_runner import main
    with pytest.raises(SystemExit) as exc:
        main(["optimize", "--truth-table", "001"])
    assert exc.value.code == 2


def test_bounded_three_input_conjunction_proves_one_workspace_minimum():
    p = problem((0, 0, 0, 0, 0, 0, 0, 1), operations=3, budget=Budget(21, 18, 45, 45))
    result = sweep_clean_ancillas(p, budgets=(0, 1), scheduler="distance",
                                 limits=OptimizationLimits(80_000, 512, 40_000, 10_000, 8., 8., 16))
    assert result["rows"][0]["result"]["status"] == "infeasible"
    assert result["rows"][1]["result"]["incumbent"]["success"]
    assert result["minimum_available_clean_workspace_proved"] == 1
    assert result["rows"][1]["result"]["incumbent"]["used_clean_workspace"] == 1
