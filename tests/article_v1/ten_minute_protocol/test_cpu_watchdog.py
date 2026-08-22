from ckt_types import ResourceBudget
from circuit.gate import Gate
from enums import GateType
from evaluate import evaluate
from experiments.article_v1_ten_minute_protocol import audit_ten_minute_runs
from experiments.article_v1_ten_minute_protocol import load_ten_minute_corpus_config
from benchmarks.article_native_corpus import build_article_v1_corpus
from experiments.article_v1_runner import evaluate_article_v1_run
from pathlib import Path


def test_cpu_watchdog_stops_only_between_complete_expansions():
    ticks = iter((0, 0, 2_000_000_000, 2_000_000_000))
    report = evaluate(
        num_qubits=2,
        target_gates=(Gate(GateType.H, (0,)), Gate(GateType.H, (1,))),
        budget=ResourceBudget(
            max_t_count=4,
            max_two_qubit_count=4,
            max_gates=4,
            max_depth=4,
        ),
        max_steps=10,
        scheduler="fifo",
        observation_features=False,
        process_cpu_limit_seconds=1.0,
        process_cpu_clock_ns=lambda: next(ticks),
    )

    assert report["expansions"] == 1
    assert report["complete"] is False
    assert report["terminal_reason"] == "OPERABILITY_TIMEOUT"
    assert report["terminated"] is False
    assert report["truncated"] is False
    assert report["certified"] is False
    assert report["process_cpu_time_ns"] == 2_000_000_000
    assert report["process_cpu_seconds"] == 2.0


def test_audit_rejects_timeout_as_incomplete_not_ordinary_failure():
    timeout = {
        "schema_version": "article-v1-10min-raw-run-v1",
        "complete": False,
        "certified": False,
        "terminal_reason": "OPERABILITY_TIMEOUT",
    }
    result = audit_ten_minute_runs((timeout,))
    assert result["passed"] is False
    assert result["operability_timeout_indices"] == [0]
    assert result["timeouts_are_not_failures"] is True

    completed = {
        **timeout,
        "complete": True,
        "terminal_reason": "UNSOLVED_WITHIN_EXPANSION_BUDGET",
    }
    assert audit_ten_minute_runs((completed,))["passed"] is True


def test_v3_raw_run_records_cpu_timeout_without_starting_expansion():
    root = Path(__file__).resolve().parents[3]
    corpus_config = load_ten_minute_corpus_config(
        root / "configs/article_v1_10min_pilot.json"
    )
    case = build_article_v1_corpus(corpus_config).cases(
        split="train", difficulty="easy"
    )[0]
    ticks = iter((0, 1_000_000_000, 1_000_000_000))
    row = evaluate_article_v1_run(
        case,
        scheduler="fifo",
        expansion_budget=64,
        evaluation_seed=1,
        budget_mode="fixed-max-horizon-anytime-v1",
        budget_thresholds=(64,),
        process_cpu_limit_seconds=1.0,
        process_cpu_clock_ns=lambda: next(ticks),
    )
    assert row["schema_version"] == "article-v1-10min-raw-run-v1"
    assert row["expansions"] == 0
    assert row["terminal_reason"] == "OPERABILITY_TIMEOUT"
    assert row["complete"] is False
    assert row["process_cpu_seconds"] == 1.0


def test_equal_cpu_boundary_is_completed_secondary_evidence():
    ticks = iter((0, 1_000_000_000, 1_000_000_000))
    report = evaluate(
        num_qubits=2,
        target_gates=(Gate(GateType.H, (0,)), Gate(GateType.H, (1,))),
        budget=ResourceBudget(4, 4, 4, 4),
        max_steps=10,
        scheduler="fifo",
        observation_features=False,
        process_cpu_limit_seconds=1.0,
        process_cpu_clock_ns=lambda: next(ticks),
        process_cpu_limit_status="CPU_BUDGET_EXHAUSTED",
    )
    assert report["expansions"] == 0
    assert report["complete"] is True
    assert report["terminal_reason"] == "CPU_BUDGET_EXHAUSTED"
