"""Exhaustive arithmetic checks for the declared CCZ parity polynomial."""

from search.problems.toffoli_parity import (
    PHASE_TERM_ORDER,
    REQUIRED_PHASE_TERMS,
    phase_identity_holds,
    phase_identity_rows,
)


def test_seven_term_ccz_phase_identity_holds_for_all_boolean_inputs():
    rows = phase_identity_rows()

    assert len(rows) == 8
    assert {row["assignment"] for row in rows} == set(range(8))
    assert all(row["lhs_mod_8"] == row["rhs_mod_8"] for row in rows)
    assert phase_identity_holds()


def test_phase_terms_have_the_exact_required_signed_masks():
    assert PHASE_TERM_ORDER == (1, 2, 3, 4, 5, 6, 7)
    assert dict(REQUIRED_PHASE_TERMS) == {
        0b001: +1,
        0b010: +1,
        0b100: +1,
        0b011: -1,
        0b101: -1,
        0b110: -1,
        0b111: +1,
    }
