from benchmarks.toffoli import build_known_toffoli_state, toffoli_resource_summary


def test_toffoli_resource_claims_name_external_baseline_without_search_proof():
    summary = toffoli_resource_summary(build_known_toffoli_state())

    assert summary["matches_known_resource_baseline"] is True
    assert summary["matches_published_lower_bound"] is True
    assert summary["matches_published_t_lower_bound"] is True
    assert summary["matches_published_cnot_lower_bound"] is True
    assert summary["resource_regression_passed"] is True
    assert summary["published_resource_baseline"] == {
        "t_count": 7,
        "cnot_count": 6,
        "scope": "external published Toffoli result",
        "proved_by_this_search_run": False,
    }
    assert set(summary["deprecated_resource_claim_fields"]) == {
        "matches_known_optimal_T_count",
        "matches_known_optimal_CNOT_count",
    }
