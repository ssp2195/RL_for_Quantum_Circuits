from pathlib import Path
import pytest
from experiments.article_v1_ten_minute_protocol import CurriculumAccounting, ProcessCPUWatchdog, anytime_success_rows, load_ten_minute_config, load_ten_minute_corpus_config, seeded_round_robin_schedule, train_fixed_interaction_curriculum
ROOT = Path(__file__).resolve().parents[3]
def test_pilot_config():
    c = load_ten_minute_config(ROOT / "configs/article_v1_10min_pilot.json")
    assert c.digest.startswith("sha256:") and c.runtime.target_episode_cpu_seconds == 540
    assert c.training.eligible_difficulties == ("easy", "medium")
    corpus = load_ten_minute_corpus_config(ROOT / "configs/article_v1_10min_pilot.json")
    assert corpus.profile == "pilot"
def test_template_grid_and_freeze_guard():
    p = ROOT / "configs/article_v1_10min_publication_template.json"; c = load_ten_minute_config(p)
    assert c.training.candidate_total_expansions_per_seed == (20000,40000,60000)
    with pytest.raises(ValueError, match="not frozen"): load_ten_minute_config(p, require_frozen=True)
def test_anytime_curve():
    rows = anytime_success_rows(first_certified_hit_expansion=730, executed_max_horizon=1792, thresholds=[256,512,1024,1792])
    assert [r["success_by_threshold"] for r in rows] == [False,False,True,True]
    with pytest.raises(ValueError, match="exceeds"): anytime_success_rows(first_certified_hit_expansion=None, executed_max_horizon=512, thresholds=[1024])
def test_watchdog_boundary():
    ticks = iter([100,200,700]); w = ProcessCPUWatchdog(0.0000006, clock_ns=lambda: next(ticks)); w.start(); assert w.at_safe_boundary().allowed is True; d = w.at_safe_boundary(); assert d.status == "OPERABILITY_TIMEOUT"
def test_schedule():
    a = seeded_round_robin_schedule(["b","a","c"], total_expansions=8, seed=7); assert a == seeded_round_robin_schedule(["b","a","c"], total_expansions=8, seed=7); assert set(a) == {"a","b","c"}
def test_curriculum_excludes_hard_and_accounts_partial_final_episode():
    c = CurriculumAccounting(("e", "m"), {"e":"easy", "m":"medium"}, 5, {"easy":2, "medium":4}, 7)
    first = c.next_episode(); assert first in (("e",2), ("m",4)); c.record_expansions(*first)
    second = c.next_episode(); assert second is not None and second[1] <= c.remaining; c.record_expansions(*second)
    assert c.completed == 5 and c.remaining == 0 and c.metadata()["hard_targets_used_for_training"] is False
    with pytest.raises(ValueError, match="hard targets"):
        CurriculumAccounting(("h",), {"h":"hard"}, 2, {"easy":2}, 1)
def test_fixed_curriculum_calls_episode_adapter_with_effective_caps():
    calls = []
    result = train_fixed_interaction_curriculum(
        [{"target_id":"e", "difficulty":"easy"}, {"target_id":"m", "difficulty":"medium"}],
        total_expansions=5, episode_caps_by_difficulty={"easy":2, "medium":4}, seed=3,
        train_episode=lambda target, cap, index: (calls.append((target["target_id"], cap, index)) or {"expansions": cap}),
    )
    assert result["total_training_expansions_completed"] == 5
    assert sum(row["effective_episode_cap"] for row in result["history"]) >= 5
    assert result["convergence_stopping"] if "convergence_stopping" in result else True
    assert calls[-1][1] <= 4
