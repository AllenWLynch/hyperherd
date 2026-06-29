"""Tests for the stateless successive-halving planner (`successive_halving.py`).

These exercise the pure ranking core: rung schedules, value-at-rung dedup,
the early-decision inequalities (incl. ties), promotion across rungs, the
paused→resume/prune transitions, stickiness, idempotence, and the
status×verdict→action table.
"""

import math
import unittest

from hyperherd.successive_halving import (
    Action,
    SweepConfig,
    TrialState,
    Verdict,
    decision_label,
    dedup_stream,
    explain,
    plan_successive_halving,
    reached_rung_index,
    rung_schedule,
    value_at_rung,
    verdict_to_action,
)


def _stream(*pairs, ts_offset=0):
    """Build a metric stream from (step, value) pairs. ts == step by default."""
    return [{"step": s, "value": v, "ts": s + ts_offset} for s, v in pairs]


def _cfg(direction="min", min_steps=10, budget=80, eta=2, mode="sync"):
    return SweepConfig(metric="m", direction=direction,
                       min_steps=min_steps, budget=budget, eta=eta, mode=mode)


def _by_index(plan):
    return {p.index: p for p in plan}


class TestRungSchedule(unittest.TestCase):
    def test_geometric_ladder(self):
        self.assertEqual(rung_schedule(10, 80, 2), [10, 20, 40, 80])

    def test_top_rung_below_budget(self):
        # 96 > 100 stops the ladder at 80.
        self.assertEqual(rung_schedule(10, 100, 2), [10, 20, 40, 80])

    def test_single_rung_when_min_equals_budget(self):
        self.assertEqual(rung_schedule(10, 10, 2), [10])

    def test_empty_when_min_exceeds_budget(self):
        self.assertEqual(rung_schedule(100, 80, 2), [])

    def test_eta_three(self):
        self.assertEqual(rung_schedule(10, 100, 3), [10, 30, 90])

    def test_bad_eta_returns_empty(self):
        self.assertEqual(rung_schedule(10, 80, 1), [])


class TestValueAtRung(unittest.TestCase):
    def test_dedup_keeps_max_ts(self):
        stream = [
            {"step": 10, "value": 0.9, "ts": 1},
            {"step": 10, "value": 0.5, "ts": 5},  # newer → wins
            {"step": 20, "value": 0.3, "ts": 2},
        ]
        d = dedup_stream(stream)
        self.assertEqual(d[10], 0.5)
        self.assertEqual(d[20], 0.3)

    def test_out_of_order_append(self):
        d = dedup_stream(_stream((20, 0.3), (10, 0.9)))
        self.assertEqual(value_at_rung(d, 10), 0.9)
        self.assertEqual(value_at_rung(d, 20), 0.3)

    def test_value_at_largest_step_at_or_below(self):
        d = dedup_stream(_stream((5, 1.0), (12, 0.8), (18, 0.7)))
        # rung 10: latest step <= 10 is step 5
        self.assertEqual(value_at_rung(d, 10), 1.0)
        # rung 20: latest step <= 20 is step 18
        self.assertEqual(value_at_rung(d, 20), 0.7)

    def test_empty_stream_is_none(self):
        self.assertIsNone(value_at_rung({}, 10))

    def test_sparse_logging_past_rung_falls_back_to_earliest(self):
        # First record is at step 15, past rung 10 → use the earliest value.
        d = dedup_stream(_stream((15, 0.7), (25, 0.6)))
        self.assertEqual(value_at_rung(d, 10), 0.7)

    def test_reached_rung_index(self):
        rungs = [10, 20, 40, 80]
        self.assertEqual(reached_rung_index(dedup_stream(_stream((25, 1))), rungs), 1)
        self.assertEqual(reached_rung_index(dedup_stream(_stream((10, 1))), rungs), 0)
        self.assertEqual(reached_rung_index(dedup_stream(_stream((9, 1))), rungs), -1)
        self.assertEqual(reached_rung_index({}, rungs), -1)


class TestEarlyDecisionInequalities(unittest.TestCase):
    """Top K=ceil(m/2) kept, bottom floor(m/2) pruned, for m=2..5, all reached."""

    def _verdicts_all_reached(self, values, direction="min"):
        # All trials reached rung 0 with the given values; decide each.
        cfg = _cfg(direction=direction, min_steps=10, budget=10)  # single rung
        trials = [TrialState(i, "running", _stream((10, v)))
                  for i, v in enumerate(values)]
        return {p.index: p.verdict for p in plan_successive_halving(trials, cfg)}

    def test_m2(self):
        v = self._verdicts_all_reached([0.1, 0.2])
        self.assertEqual(v[0], Verdict.RUN_FREE)  # single rung → cleared = run_free
        self.assertEqual(v[1], Verdict.PRUNE)

    def test_m3_keeps_two(self):
        v = self._verdicts_all_reached([0.1, 0.2, 0.3])
        kept = [i for i in v if v[i] in (Verdict.PROMOTE, Verdict.RUN_FREE)]
        pruned = [i for i in v if v[i] == Verdict.PRUNE]
        self.assertEqual(sorted(kept), [0, 1])     # ceil(3/2)=2 kept
        self.assertEqual(pruned, [2])              # floor(3/2)=1 pruned

    def test_m4_keeps_two(self):
        v = self._verdicts_all_reached([0.1, 0.2, 0.3, 0.4])
        kept = [i for i in v if v[i] in (Verdict.PROMOTE, Verdict.RUN_FREE)]
        self.assertEqual(sorted(kept), [0, 1])
        self.assertEqual(sorted(i for i in v if v[i] == Verdict.PRUNE), [2, 3])

    def test_m5_keeps_three(self):
        v = self._verdicts_all_reached([0.5, 0.4, 0.3, 0.2, 0.1])
        kept = [i for i in v if v[i] in (Verdict.PROMOTE, Verdict.RUN_FREE)]
        # best three values are indices 4,3,2
        self.assertEqual(sorted(kept), [2, 3, 4])
        self.assertEqual(sorted(i for i in v if v[i] == Verdict.PRUNE), [0, 1])

    def test_direction_max_mirrors(self):
        v = self._verdicts_all_reached([0.1, 0.2, 0.3, 0.4], direction="max")
        kept = [i for i in v if v[i] in (Verdict.PROMOTE, Verdict.RUN_FREE)]
        # higher is better → 3,2 kept
        self.assertEqual(sorted(kept), [2, 3])


class TestTieBreak(unittest.TestCase):
    def test_all_equal_keeps_smallest_indices(self):
        cfg = _cfg(min_steps=10, budget=10)
        trials = [TrialState(i, "running", _stream((10, 0.5))) for i in range(3)]
        v = {p.index: p.verdict for p in plan_successive_halving(trials, cfg)}
        # ceil(3/2)=2 kept → smallest two indices survive, largest pruned
        self.assertIn(v[0], (Verdict.PROMOTE, Verdict.RUN_FREE))
        self.assertIn(v[1], (Verdict.PROMOTE, Verdict.RUN_FREE))
        self.assertEqual(v[2], Verdict.PRUNE)

    def test_tie_straddling_boundary(self):
        cfg = _cfg(min_steps=10, budget=10)
        # values: 0.1, 0.5, 0.5, 0.9 → m=4 K=2. index 0 clearly top.
        # The two 0.5 ties: index1 ranks ahead of index2 → index1 kept, index2 pruned.
        trials = [TrialState(i, "running", _stream((10, val)))
                  for i, val in enumerate([0.1, 0.5, 0.5, 0.9])]
        v = {p.index: p.verdict for p in plan_successive_halving(trials, cfg)}
        self.assertIn(v[0], (Verdict.PROMOTE, Verdict.RUN_FREE))
        self.assertIn(v[1], (Verdict.PROMOTE, Verdict.RUN_FREE))
        self.assertEqual(v[2], Verdict.PRUNE)
        self.assertEqual(v[3], Verdict.PRUNE)


class TestLastSurvivorNeverPruned(unittest.TestCase):
    """The field can never be pruned to zero.

    For a cohort of size m, K = ceil(m/2) >= 1 and PRUNE requires
    ahead_definite >= K. The cohort leader (best objective, ties broken by
    smallest index) always has ahead_definite == 0, so 0 >= K is never true:
    every non-empty cohort promotes at least one trial, and a singleton
    always promotes. These pin that guarantee against regressions.
    """

    def test_single_running_trial_with_terrible_metric_not_pruned(self):
        # A sole trial blowing up (huge loss) must still survive — there is
        # nobody to lose the comparison to.
        cfg = _cfg(min_steps=1, budget=8)
        trials = [TrialState(0, "running", _stream(*[(i, 999.0) for i in range(9)]))]
        p = _by_index(plan_successive_halving(trials, cfg))[0]
        self.assertNotEqual(p.action, Action.PRUNE)
        self.assertIn(p.verdict, (Verdict.PROMOTE, Verdict.RUN_FREE))

    def test_single_trial_with_nan_not_pruned(self):
        cfg = _cfg(min_steps=1, budget=8)
        trials = [TrialState(0, "running", _stream((0, math.nan), (1, math.nan)))]
        p = _by_index(plan_successive_halving(trials, cfg))[0]
        self.assertNotEqual(p.action, Action.PRUNE)

    def test_sole_survivor_among_excluded_not_pruned(self):
        # Everyone else is already pruned/failed/cancelled (excluded from the
        # cohort). The lone competitor must not be pruned even with a bad metric.
        cfg = _cfg(min_steps=1, budget=8)
        trials = [
            TrialState(0, "pruned",    _stream((0, 0.1))),
            TrialState(1, "failed",    _stream((0, 0.1))),
            TrialState(2, "cancelled", _stream((0, 0.1))),
            TrialState(4, "running",   _stream(*[(i, 50.0) for i in range(9)])),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))[4]
        self.assertNotEqual(p.action, Action.PRUNE)
        self.assertIn(p.verdict, (Verdict.PROMOTE, Verdict.RUN_FREE))

    def test_never_prunes_whole_cohort(self):
        # Exhaustive small-field check: across cohort sizes and value
        # orderings, at least one trial always survives each tick.
        import itertools
        cfg = _cfg(min_steps=1, budget=8)
        for m in range(1, 7):
            for vals in itertools.permutations(range(m)):
                trials = [
                    TrialState(i, "running", _stream(*[(s, float(v)) for s in range(9)]))
                    for i, v in enumerate(vals)
                ]
                plan = plan_successive_halving(trials, cfg)
                pruned = sum(1 for p in plan if p.action == Action.PRUNE)
                self.assertLess(pruned, m, f"all {m} pruned for values {vals}")


class TestAmbiguityAndPause(unittest.TestCase):
    def test_lone_reached_with_unreached_pauses(self):
        cfg = _cfg(min_steps=10, budget=80)
        # idx0 reached rung0; others still climbing toward it → ambiguous.
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),
            TrialState(1, "running", _stream((5, 0.2))),
            TrialState(2, "running", _stream((5, 0.3))),
            TrialState(3, "running", _stream((5, 0.4))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PAUSE)
        self.assertEqual(p[0].action, Action.PAUSE)
        for i in (1, 2, 3):
            self.assertEqual(p[i].verdict, Verdict.NOT_AT_RUNG)
            self.assertEqual(p[i].action, Action.NONE)

    def test_enough_reached_resolves_without_full_field(self):
        cfg = _cfg(min_steps=10, budget=80)
        # m=4, K=2. idx3 beaten by 2 reached (idx0,1) → PRUNE even with idx2 unreached.
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((5, 0.0))),   # unreached
            TrialState(3, "running", _stream((10, 0.9))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[3].verdict, Verdict.PRUNE)   # ahead_definite=2 >= K=2
        self.assertEqual(p[0].verdict, Verdict.PROMOTE)  # ahead_definite=0


class TestPausedTransitions(unittest.TestCase):
    def test_paused_promoted_resumes(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [
            TrialState(0, "paused", _stream((10, 0.05))),  # best
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.3))),
            TrialState(3, "running", _stream((10, 0.4))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PROMOTE)
        self.assertEqual(p[0].action, Action.SUBMIT)  # resume

    def test_paused_now_bottom_half_prunes(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [
            TrialState(0, "paused", _stream((10, 0.9))),  # worst
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.3))),
            TrialState(3, "running", _stream((10, 0.4))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PRUNE)
        self.assertEqual(p[0].action, Action.PRUNE)

    def test_paused_still_ambiguous_stays(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [
            TrialState(0, "paused", _stream((10, 0.25))),
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((5, 0.0))),   # unreached
            TrialState(3, "running", _stream((5, 0.0))),   # unreached
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PAUSE)
        self.assertEqual(p[0].action, Action.NONE)  # stay paused


class TestMultiRung(unittest.TestCase):
    def test_promotion_then_prune_at_higher_rung(self):
        cfg = _cfg(min_steps=10, budget=80)  # rungs 10,20,40,80
        # rung0: all 4 reach; top2 (0,1) promote. rung1: only 0,1 in cohort,
        # 0 better → 0 promote, 1 prune at rung1.
        trials = [
            TrialState(0, "running", _stream((10, 0.1), (20, 0.05))),
            TrialState(1, "running", _stream((10, 0.2), (20, 0.15))),
            TrialState(2, "running", _stream((10, 0.3))),
            TrialState(3, "running", _stream((10, 0.4))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PROMOTE)
        self.assertEqual(p[0].rung, 1)
        self.assertEqual(p[1].verdict, Verdict.PRUNE)
        self.assertEqual(p[1].rung, 1)
        self.assertEqual(p[2].verdict, Verdict.PRUNE)
        self.assertEqual(p[2].rung, 0)
        self.assertEqual(p[3].verdict, Verdict.PRUNE)

    def test_cleared_final_rung_runs_free(self):
        cfg = _cfg(min_steps=10, budget=20)  # rungs 10,20
        # 2 trials, both reach rung1; better one clears the final rung.
        trials = [
            TrialState(0, "running", _stream((10, 0.1), (20, 0.05))),
            TrialState(1, "running", _stream((10, 0.2), (20, 0.15))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.RUN_FREE)
        self.assertEqual(p[0].action, Action.NONE)  # running → keep running
        self.assertEqual(p[1].verdict, Verdict.PRUNE)


class TestStickyAndExcluded(unittest.TestCase):
    def test_pruned_stays_pruned_even_with_great_metric(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [
            TrialState(0, "pruned", _stream((10, 0.001), (20, 0.001))),
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.3))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].action, Action.NONE)
        self.assertEqual(p[0].verdict, Verdict.NONE)

    def test_failed_excluded_does_not_strand_cohort(self):
        cfg = _cfg(min_steps=10, budget=80)
        # idx3 failed (excluded). Remaining 3 form the cohort: K=2.
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.9))),
            TrialState(3, "failed", _stream((10, 0.0))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[3].action, Action.NONE)
        # m=3 over {0,1,2}: keep 2, prune 1 (idx2 worst).
        self.assertEqual(p[2].verdict, Verdict.PRUNE)
        self.assertIn(p[0].verdict, (Verdict.PROMOTE, Verdict.RUN_FREE))


class TestThrashGuards(unittest.TestCase):
    def test_submitted_trial_not_judged(self):
        cfg = _cfg(min_steps=10, budget=80)
        # idx0 just (re)submitted but already has stale data making it look bad —
        # must NOT be pruned/paused while 'submitted'.
        trials = [
            TrialState(0, "submitted", _stream((10, 0.9))),
            TrialState(1, "running", _stream((10, 0.1))),
            TrialState(2, "running", _stream((10, 0.2))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].action, Action.NONE)

    def test_ready_trial_not_submitted(self):
        # SH never launches never-submitted (ready) trials — that's the user's
        # job (`herd run`). SH only ever resumes trials it itself paused.
        cfg = _cfg(min_steps=10, budget=80)
        trials = [TrialState(i, "ready", []) for i in range(3)]
        p = _by_index(plan_successive_halving(trials, cfg))
        for i in range(3):
            self.assertEqual(p[i].action, Action.NONE)
            self.assertEqual(p[i].verdict, Verdict.NOT_AT_RUNG)

    def test_ready_trials_excluded_from_cohort_denominator(self):
        # Policy A: ready (never-launched) trials are out of the cohort, so they
        # neither raise K nor count as `unreached`. Two arrived running trials
        # therefore form a 2-cohort (K=1) and decide immediately, instead of
        # both pausing forever while waiting on ready trials SH will never start.
        cfg = _cfg(min_steps=10, budget=80)
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),  # better
            TrialState(1, "running", _stream((10, 0.2))),  # worse
            TrialState(2, "ready", []),
            TrialState(3, "ready", []),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertIn(p[0].verdict, (Verdict.PROMOTE, Verdict.RUN_FREE))
        self.assertEqual(p[1].verdict, Verdict.PRUNE)
        self.assertEqual(p[1].action, Action.PRUNE)
        # The ready trials are untouched.
        self.assertEqual(p[2].action, Action.NONE)
        self.assertEqual(p[3].action, Action.NONE)


class TestNanHandling(unittest.TestCase):
    def test_nan_ranks_last_and_is_pruned(self):
        cfg = _cfg(min_steps=10, budget=10)
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, float("nan")))),
            TrialState(3, "running", _stream((10, float("inf")))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        # NaN/inf trials are worst → pruned; finite top-2 kept.
        self.assertEqual(p[2].verdict, Verdict.PRUNE)
        self.assertEqual(p[3].verdict, Verdict.PRUNE)
        self.assertIn(p[0].verdict, (Verdict.PROMOTE, Verdict.RUN_FREE))


class TestIdempotence(unittest.TestCase):
    def test_same_input_same_output(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [
            TrialState(0, "running", _stream((10, 0.1), (20, 0.05))),
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.3))),
            TrialState(3, "paused", _stream((10, 0.4))),
        ]
        r1 = [(p.index, p.action, p.verdict) for p in plan_successive_halving(trials, cfg)]
        r2 = [(p.index, p.action, p.verdict) for p in plan_successive_halving(trials, cfg)]
        self.assertEqual(r1, r2)


class TestVerdictToAction(unittest.TestCase):
    def test_table(self):
        # (status, verdict) -> expected action
        cases = [
            ("ready", Verdict.NOT_AT_RUNG, Action.NONE),
            ("running", Verdict.PRUNE, Action.PRUNE),
            ("running", Verdict.PAUSE, Action.PAUSE),
            ("running", Verdict.PROMOTE, Action.NONE),
            ("running", Verdict.RUN_FREE, Action.NONE),
            ("running", Verdict.NOT_AT_RUNG, Action.NONE),
            ("submitted", Verdict.PRUNE, Action.NONE),
            ("queued", Verdict.PAUSE, Action.NONE),
            ("paused", Verdict.PROMOTE, Action.SUBMIT),
            ("paused", Verdict.RUN_FREE, Action.SUBMIT),
            ("paused", Verdict.PRUNE, Action.PRUNE),
            ("paused", Verdict.PAUSE, Action.NONE),
            ("completed", Verdict.PRUNE, Action.NONE),
            ("failed", Verdict.PROMOTE, Action.NONE),
            ("cancelled", Verdict.PROMOTE, Action.NONE),
            ("pruned", Verdict.PROMOTE, Action.NONE),
        ]
        for status, verdict, expected in cases:
            self.assertEqual(
                verdict_to_action(status, verdict), expected,
                f"{status} + {verdict} should be {expected}",
            )


class TestStandingAndExplain(unittest.TestCase):
    """The cohort arithmetic surfaced by `herd sh --reason`."""

    def test_standing_records_cohort_arithmetic(self):
        cfg = _cfg(min_steps=10, budget=80)
        # 0,1,2 reached rung 0; 3,4 below it (unreached). Cohort=5, keep top 3.
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),  # best
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.3))),
            TrialState(3, "running", _stream((5, 0.05))),
            TrialState(4, "running", _stream((5, 0.06))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        s0 = p[0].standing
        self.assertIsNotNone(s0)
        self.assertEqual((s0.rung, s0.step), (0, 10))
        self.assertEqual((s0.cohort_size, s0.keep), (5, 3))
        self.assertEqual((s0.ahead_definite, s0.unreached), (0, 2))
        self.assertEqual(s0.value, 0.1)
        # idx2: 2 trials ahead, 2 unreached → undecidable → PAUSE.
        self.assertEqual(p[2].verdict, Verdict.PAUSE)
        self.assertEqual((p[2].standing.ahead_definite, p[2].standing.unreached),
                         (2, 2))

    def test_off_rung_trials_have_no_standing(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [TrialState(0, "running", _stream((5, 0.1))),  # below rung 0
                  TrialState(1, "ready", [])]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertIsNone(p[0].standing)
        self.assertIsNone(p[1].standing)

    def test_explain_falls_back_to_reason_off_rung(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [TrialState(0, "ready", [])]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(explain(p[0]), p[0].reason)

    def test_explain_spells_out_prune(self):
        cfg = _cfg(min_steps=10, budget=10)  # single rung
        trials = [TrialState(i, "running", _stream((10, v)))
                  for i, v in enumerate([0.1, 0.2, 0.3, 0.4])]
        p = _by_index(plan_successive_halving(trials, cfg))
        # idx3 is worst of 4 → 3 ahead, keep top 2 → prune. All four have
        # arrived (unreached=0), so explain uses the rank-of-arrived phrasing.
        self.assertEqual(p[3].verdict, Verdict.PRUNE)
        text = explain(p[3])
        self.assertIn("rank 4 of 4 arrived", text)
        self.assertIn("keep top 2", text)
        self.assertIn("below the cut", text)

    def test_decision_label_maps_effect(self):
        cfg = _cfg(min_steps=10, budget=10)
        trials = [TrialState(i, "running", _stream((10, v)))
                  for i, v in enumerate([0.1, 0.2, 0.3, 0.4])]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(decision_label(p[3]), "prune")        # PRUNE action
        self.assertEqual(decision_label(p[0]), "run-to-budget")  # RUN_FREE
        # A promoted (paused→resume) trial reads "resume".
        resume = next(p2 for p2 in plan_successive_halving(
            [TrialState(0, "paused", _stream((10, 0.1))),
             TrialState(1, "running", _stream((10, 0.9)))], cfg) if p2.index == 0)
        self.assertEqual(decision_label(resume), "resume")

    def test_status_and_max_step_recorded(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [TrialState(0, "running", _stream((10, 0.1), (20, 0.05)))]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].status, "running")
        self.assertEqual(p[0].max_step, 20)

    def test_completed_trial_labelled_complete_not_paused(self):
        # Regression: a completed trial that is undecidable at its rung (the
        # running field hasn't caught up) gets verdict PAUSE but action NONE —
        # it must read "complete", never "stay-paused". It's terminal, not held.
        cfg = _cfg(min_steps=10, budget=10)  # single rung at step 10
        trials = [
            TrialState(0, "completed", _stream((10, 0.1))),  # at rung 0
            TrialState(1, "running", _stream((5, 0.2))),     # below rung 0
            TrialState(2, "running", _stream((5, 0.3))),     # below rung 0
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PAUSE)   # ranking undecidable
        self.assertEqual(p[0].action, Action.NONE)      # terminal — no action
        self.assertEqual(decision_label(p[0]), "complete")

    def test_running_below_rung_labelled_pre_rung(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [TrialState(0, "running", _stream((5, 0.2)))]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.NOT_AT_RUNG)
        self.assertEqual(p[0].max_step, 5)
        self.assertEqual(decision_label(p[0]), "pre-rung")

    def test_just_launched_labelled(self):
        cfg = _cfg(min_steps=10, budget=80)
        trials = [TrialState(0, "submitted", []),
                  TrialState(1, "queued", [])]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(decision_label(p[0]), "just-launched")
        self.assertEqual(decision_label(p[1]), "just-launched")


class TestAshaMode(unittest.TestCase):
    """`mode="asha"`: rank only the arrived; keep top floor(n/eta); never pause."""

    def test_never_pauses_decides_among_arrived(self):
        # idx0,1,2 reached rung 0; idx3 still below. Under SYNC this pauses the
        # undecidable ones; under ASHA it ranks the 3 arrived and cuts now.
        cfg = _cfg(min_steps=10, budget=80, mode="asha")
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),  # best of arrived
            TrialState(1, "running", _stream((10, 0.2))),
            TrialState(2, "running", _stream((10, 0.3))),
            TrialState(3, "running", _stream((5, 0.05))),  # not yet at rung 0
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        verdicts = {i: p[i].verdict for i in range(4)}
        self.assertNotIn(Verdict.PAUSE, verdicts.values())
        # n=3 arrived, eta=2 → keep floor(3/2)=1. Only idx0 survives.
        self.assertEqual(p[0].verdict, Verdict.PROMOTE)
        self.assertEqual(p[1].verdict, Verdict.PRUNE)
        self.assertEqual(p[2].verdict, Verdict.PRUNE)
        self.assertEqual(p[3].verdict, Verdict.NOT_AT_RUNG)  # still warming up
        # Standing reflects the arrived denominator, no stragglers.
        self.assertEqual(p[1].standing.cohort_size, 3)
        self.assertEqual(p[1].standing.keep, 1)
        self.assertEqual(p[1].standing.unreached, 0)

    def test_gate_keeps_all_below_eta(self):
        # Fewer than eta arrived → no cut (keep == n), so nobody is pruned.
        cfg = _cfg(min_steps=10, budget=80, mode="asha")
        trials = [
            TrialState(0, "running", _stream((10, 0.1))),
            TrialState(1, "running", _stream((5, 0.2))),   # below rung 0
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].verdict, Verdict.PROMOTE)  # lone arrival, not cut
        self.assertEqual(p[0].standing.keep, 1)
        self.assertEqual(p[0].standing.cohort_size, 1)

    def test_resumes_promoted_paused_trial(self):
        # A paused trial that's top of the arrived field resumes (SUBMIT).
        cfg = _cfg(min_steps=10, budget=80, mode="asha")
        trials = [
            TrialState(0, "paused", _stream((10, 0.05))),  # best
            TrialState(1, "running", _stream((10, 0.2))),
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        self.assertEqual(p[0].action, Action.SUBMIT)

    def test_cut_at_lowest_failed_rung(self):
        # A trial that clears rung 0 but tanks at rung 1 is pruned *at rung 1*.
        cfg = _cfg(min_steps=10, budget=40, mode="asha")  # rungs [10,20,40]
        trials = [
            TrialState(0, "running", _stream((10, 0.1), (20, 0.1))),  # survives
            TrialState(1, "running", _stream((10, 0.2), (20, 0.5))),  # tanks at r1
            TrialState(2, "running", _stream((10, 0.3))),  # cut at rung 0
            TrialState(3, "running", _stream((10, 0.4))),  # cut at rung 0
        ]
        p = _by_index(plan_successive_halving(trials, cfg))
        # rung 0: 4 arrived, keep 2 → idx0,idx1 survive; idx2,idx3 cut at rung 0.
        self.assertEqual(p[2].rung, 0)
        self.assertEqual(p[3].rung, 0)
        # rung 1: 2 arrived (idx0,idx1), keep 1 → idx0 survives, idx1 cut at rung 1.
        self.assertEqual(p[1].verdict, Verdict.PRUNE)
        self.assertEqual(p[1].rung, 1)
        self.assertEqual(p[0].verdict, Verdict.PROMOTE)


if __name__ == "__main__":
    unittest.main()
