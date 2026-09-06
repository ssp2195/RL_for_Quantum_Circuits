"""Exact hybrid Clifford+T synthesis with hierarchical and ancilla-aware search."""
from .ancilla_benchmarks import (
    ancilla_evaluation_targets,
    ancilla_training_targets,
    qft3_clean_ancilla_target,
)
from .ancilla_certify import AncillaCertificationResult, certify_ancilla_state
from .ancilla_contract import (
    AncillaContract,
    AncillaSynthesisTarget,
    PhaseMode,
    target_from_hidden_ancilla_gates,
)
from .ancilla_search import (
    AncillaDeferredSearch,
    DisjointAncillaLinUCB,
    LinearAncillaOuterSarsa,
    evaluate_ancilla_hierarchy,
    train_ancilla_inner_bandit,
    train_ancilla_outer_sarsa,
)
from .benchmarks import (
    held_out_targets,
    qft2_target,
    structured_toffoli_target,
    training_targets,
    validation_targets,
)
from .canonicalize import (
    ProjectiveCanonicalization,
    canonicalize_projective,
    legacy_projective_key,
    projective_key,
)
from .certify import certify_state
from .cnot_crossover import (
    DeferredCnotSearch,
    EagerCnotSearch,
    LinearCnotLinUCB,
    LinearOuterSarsa,
    crossover_evaluation_targets,
    crossover_training_targets,
)
from .mixed_crossover import (
    DeferredMixedSearch,
    DisjointMixedLinUCB,
    EagerMixedSearch,
    LinearMixedOuterSarsa,
    mixed_evaluation_targets,
    mixed_training_targets,
)
from .model import Budget, Gate, HybridState
from .rl import LinearSarsaRanker, train_online_sarsa
from .oracle_benchmarks import (
    bnn_verification_oracle_spec,
    bnn_verification_oracle_target,
    oracle_training_targets,
)
from .oracle_synthesis import (
    BooleanOracleSpec,
    DisjointOracleLinUCB,
    LinearOracleOuterSarsa,
    OracleEvaluatorTarget,
    OracleLayout,
    OracleMacroDeferredSearch,
    OracleSynthesisResult,
    assemble_phase_oracle,
    evaluate_oracle_hierarchy,
    train_oracle_inner_bandit,
    train_oracle_outer_sarsa,
)
from .qft_guided import (
    NativeMacro,
    QFTGuidedResult,
    controlled_t_relative_phase_compute,
    exact_qft_macros,
    qft_matrix,
    relative_phase_and_compute,
    synthesize_qft_decomposition_guided,
)
from .search import HybridSearch
from .structured_toffoli import StructuredToffoliSearch

__all__ = [
    "BooleanOracleSpec",
    "DisjointOracleLinUCB",
    "LinearOracleOuterSarsa",
    "OracleEvaluatorTarget",
    "OracleLayout",
    "OracleMacroDeferredSearch",
    "OracleSynthesisResult",
    "assemble_phase_oracle",
    "bnn_verification_oracle_spec",
    "bnn_verification_oracle_target",
    "evaluate_oracle_hierarchy",
    "oracle_training_targets",
    "train_oracle_inner_bandit",
    "train_oracle_outer_sarsa",
    "AncillaCertificationResult",
    "AncillaContract",
    "AncillaDeferredSearch",
    "AncillaSynthesisTarget",
    "DisjointAncillaLinUCB",
    "LinearAncillaOuterSarsa",
    "PhaseMode",
    "Budget",
    "DeferredCnotSearch",
    "DeferredMixedSearch",
    "DisjointMixedLinUCB",
    "EagerCnotSearch",
    "EagerMixedSearch",
    "Gate",
    "HybridSearch",
    "HybridState",
    "LinearCnotLinUCB",
    "LinearMixedOuterSarsa",
    "LinearOuterSarsa",
    "LinearSarsaRanker",
    "NativeMacro",
    "QFTGuidedResult",
    "ProjectiveCanonicalization",
    "StructuredToffoliSearch",
    "canonicalize_projective",
    "certify_state",
    "crossover_evaluation_targets",
    "crossover_training_targets",
    "held_out_targets",
    "mixed_evaluation_targets",
    "legacy_projective_key",
    "mixed_training_targets",
    "projective_key",
    "qft2_target",
    "qft_matrix",
    "exact_qft_macros",
    "relative_phase_and_compute",
    "controlled_t_relative_phase_compute",
    "synthesize_qft_decomposition_guided",
    "structured_toffoli_target",
    "train_online_sarsa",
    "training_targets",
    "validation_targets",
    "ancilla_evaluation_targets",
    "ancilla_training_targets",
    "certify_ancilla_state",
    "evaluate_ancilla_hierarchy",
    "qft3_clean_ancilla_target",
    "target_from_hidden_ancilla_gates",
    "train_ancilla_inner_bandit",
    "train_ancilla_outer_sarsa",
]
