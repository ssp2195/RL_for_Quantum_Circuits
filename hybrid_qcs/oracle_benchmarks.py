"""Exact Boolean-oracle curricula and the BNN verification benchmark."""
from __future__ import annotations

from .model import Budget
from .oracle_synthesis import BooleanOracleSpec, OracleEvaluatorTarget, OracleLayout


def _truth_table(num_inputs: int, function) -> tuple[int, ...]:
    return tuple(int(bool(function(*bits))) for bits in (
        tuple((basis >> index) & 1 for index in range(num_inputs))
        for basis in range(1 << num_inputs)
    ))


def oracle_target(
    spec: BooleanOracleSpec,
    *,
    num_work: int,
    max_macros: int,
) -> OracleEvaluatorTarget:
    layout = OracleLayout.standard(spec.num_inputs, num_work)
    # Bounds are deliberately fixed by width rather than fitted to a hidden
    # witness.  They cover the small NCT curriculum and the held-out BNN target.
    if spec.num_inputs == 1:
        budget = Budget(max_t_count=14, max_cnot_count=16, max_gates=36, max_depth=32)
    elif spec.num_inputs == 2:
        budget = Budget(max_t_count=28, max_cnot_count=32, max_gates=72, max_depth=64)
    else:
        budget = Budget(max_t_count=35, max_cnot_count=48, max_gates=96, max_depth=88)
    return OracleEvaluatorTarget(
        spec=spec,
        layout=layout,
        budget=budget,
        max_macros=max_macros,
    )


def oracle_training_targets() -> tuple[OracleEvaluatorTarget, ...]:
    """Small exact Boolean evaluators used for staged linear-policy training."""

    specs: list[tuple[BooleanOracleSpec, int, int]] = []
    specs.extend(
        [
            (
                BooleanOracleSpec(
                    "train-literal-x1",
                    1,
                    _truth_table(1, lambda x1: x1),
                    application="literal predicate",
                ),
                0,
                2,
            ),
            (
                BooleanOracleSpec(
                    "train-negated-literal-x1",
                    1,
                    _truth_table(1, lambda x1: not x1),
                    application="negated literal predicate",
                ),
                0,
                3,
            ),
        ]
    )
    specs.extend(
        [
            (
                BooleanOracleSpec(
                    "train-xor-2",
                    2,
                    _truth_table(2, lambda x1, x2: x1 ^ x2),
                    application="parity predicate",
                ),
                0,
                4,
            ),
            (
                BooleanOracleSpec(
                    "train-and-2",
                    2,
                    _truth_table(2, lambda x1, x2: x1 and x2),
                    application="conjunction predicate",
                ),
                0,
                3,
            ),
            (
                BooleanOracleSpec(
                    "train-or-2",
                    2,
                    _truth_table(2, lambda x1, x2: x1 or x2),
                    application="disjunction predicate",
                ),
                0,
                5,
            ),
        ]
    )
    for marked in ("00", "01", "10", "11"):
        specs.append(
            (
                BooleanOracleSpec.from_marked_bitstrings(
                    f"train-marked-{marked}",
                    2,
                    (marked,),
                    application="two-input marked-state predicate",
                ),
                0,
                5,
            )
        )
    specs.extend(
        [
            (
                BooleanOracleSpec(
                    "train-parity-3",
                    3,
                    _truth_table(3, lambda x1, x2, x3: x1 ^ x2 ^ x3),
                    application="three-input parity predicate",
                ),
                1,
                5,
            ),
            (
                BooleanOracleSpec(
                    "train-majority-3",
                    3,
                    _truth_table(3, lambda x1, x2, x3: x1 + x2 + x3 >= 2),
                    application="three-input threshold predicate",
                ),
                1,
                5,
            ),
            (
                BooleanOracleSpec(
                    "train-and-3",
                    3,
                    _truth_table(3, lambda x1, x2, x3: x1 and x2 and x3),
                    application="three-input conjunction predicate",
                ),
                1,
                5,
            ),
            (
                BooleanOracleSpec.from_marked_bitstrings(
                    "train-marked-111",
                    3,
                    ("111",),
                    application="three-input marked-state predicate",
                ),
                1,
                6,
            ),
            (
                BooleanOracleSpec.from_marked_bitstrings(
                    "train-marked-011",
                    3,
                    ("011",),
                    application="three-input marked-state predicate",
                ),
                1,
                7,
            ),
        ]
    )
    return tuple(
        oracle_target(spec, num_work=num_work, max_macros=max_macros)
        for spec, num_work, max_macros in specs
    )


def bnn_verification_oracle_spec() -> BooleanOracleSpec:
    """Concrete three-input verification predicate from the attached BNN study.

    The source experiment identifies ``x1 x2 x3 = 100`` as the unique violating
    input and writes the phase oracle as ``I - 2 |100><100|``.  We therefore use
    that complete marked-state extension as the exact synthesis target.
    """

    return BooleanOracleSpec.from_marked_bitstrings(
        "bnn-hamming-ball-violation-100",
        3,
        ("100",),
        application=(
            "verification phase oracle for the unique robustness violation in "
            "the three-input BNN Hamming-ball example"
        ),
    )


def bnn_verification_oracle_target() -> OracleEvaluatorTarget:
    return oracle_target(
        bnn_verification_oracle_spec(),
        num_work=1,
        max_macros=7,
    )


__all__ = [
    "bnn_verification_oracle_spec",
    "bnn_verification_oracle_target",
    "oracle_target",
    "oracle_training_targets",
]
