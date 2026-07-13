"""Tests for bracketed successive halving.

Brackets partition the field so trials only compete with comparable trials. Two
flavors, mutually exclusive: `bracket_by` (explicit, keyed on param values) and
`hyperband` (random, keyed on a seeded hash of the trial index).

The load-bearing invariants, each pinned below:

* trials in different brackets NEVER affect each other's verdict;
* the hyperband hedge (s=0) is structurally unprunable;
* hyperband assignment is stable when the sweep grows (append-stability);
* every non-empty bracket keeps at least one trial (the last-survivor rule,
  which bracketing exercises far harder than a flat sweep did);
* planning twice gives the same answer (idempotence).
"""

import itertools
import unittest

from hyperherd.successive_halving import (
    Action,
    BracketSpec,
    SweepConfig,
    TrialState,
    Verdict,
    bracket_can_prune,
    bracket_partition,
    bracket_warnings,
    hyperband_bracket,
    hyperband_sizes,
    plan_successive_halving,
    rung_schedule,
)


def _stream(*pairs):
    return [{"step": s, "value": v, "ts": s} for s, v in pairs]


def _full(value, upto=80):
    """A trial that logs `value` at every step up to `upto`."""
    return _stream(*[(s, value) for s in range(upto + 1)])


def _cfg(bracket=None, direction="min", min_steps=10, budget=80, eta=2, mode="sync"):
    return SweepConfig(
        metric="m", direction=direction, min_steps=min_steps, budget=budget,
        eta=eta, mode=mode, bracket=bracket or BracketSpec(),
    )


def _by_index(plan):
    return {p.index: p for p in plan}


class TestNoBracketing(unittest.TestCase):
    """The unbracketed case must stay exactly what it was — one bracket."""

    def test_single_bracket_over_whole_field(self):
        cfg = _cfg()
        trials = [TrialState(i, "running", _full(float(i))) for i in range(4)]
        brackets = bracket_partition(trials, cfg)
        self.assertEqual(list(brackets), ["all"])
        self.assertEqual(brackets["all"].members, [0, 1, 2, 3])
        self.assertEqual(brackets["all"].rungs, rung_schedule(10, 80, 2))

    def test_bracket_field_is_none_on_actions(self):
        cfg = _cfg()
        trials = [TrialState(i, "running", _full(float(i))) for i in range(4)]
        for p in plan_successive_halving(trials, cfg):
            self.assertIsNone(p.bracket)

    def test_no_warnings(self):
        cfg = _cfg()
        trials = [TrialState(0, "running", _full(1.0))]
        self.assertEqual(bracket_warnings(bracket_partition(trials, cfg), cfg), [])


class TestBracketByParams(unittest.TestCase):
    def _trials(self):
        # Two optimizers x 3 trials. `sgd` is uniformly worse than `adam` — the
        # exact situation that biases a flat sweep: every sgd trial would be
        # pruned by the adam trials regardless of how sgd trials rank among
        # THEMSELVES.
        return [
            TrialState(0, "running", _full(0.10), params={"opt": "adam"}),
            TrialState(1, "running", _full(0.20), params={"opt": "adam"}),
            TrialState(2, "running", _full(0.30), params={"opt": "adam"}),
            TrialState(3, "running", _full(5.00), params={"opt": "sgd"}),
            TrialState(4, "running", _full(6.00), params={"opt": "sgd"}),
            TrialState(5, "running", _full(7.00), params={"opt": "sgd"}),
        ]

    def test_partition_keys_on_param_values(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        brackets = bracket_partition(self._trials(), cfg)
        self.assertEqual(len(brackets), 2)
        members = {b.key: b.members for b in brackets.values()}
        self.assertEqual(sorted(members.values()), [[0, 1, 2], [3, 4, 5]])

    def test_all_param_brackets_share_the_global_ladder(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        brackets = bracket_partition(self._trials(), cfg)
        for b in brackets.values():
            self.assertEqual(b.rungs, rung_schedule(10, 80, 2))

    def test_worse_bracket_is_judged_on_its_own(self):
        # THE headline behavior. Flat SH prunes all three sgd trials (they're
        # the bottom half of the field). Bracketed, sgd's best survives and only
        # sgd's own worst is cut.
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        plan = _by_index(plan_successive_halving(self._trials(), cfg))
        # Best of each bracket survives.
        self.assertNotEqual(plan[0].action, Action.PRUNE)   # best adam
        self.assertNotEqual(plan[3].action, Action.PRUNE)   # best sgd, worst overall
        # Worst of each bracket is cut.
        self.assertEqual(plan[2].action, Action.PRUNE)      # worst adam
        self.assertEqual(plan[5].action, Action.PRUNE)      # worst sgd

    def test_flat_sh_would_have_pruned_the_whole_sgd_bracket(self):
        # The control: without bracketing, every sgd trial dies. This is the bias
        # bracketing exists to remove — assert it, so the previous test is
        # demonstrably doing something.
        plan = _by_index(plan_successive_halving(self._trials(), _cfg()))
        for i in (3, 4, 5):
            self.assertEqual(plan[i].action, Action.PRUNE)

    def test_brackets_are_independent(self):
        # Changing a trial's value in one bracket cannot change a verdict in
        # another. This is the property that makes bracketing *mean* anything.
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        base = _by_index(plan_successive_halving(self._trials(), cfg))
        perturbed = self._trials()
        perturbed[0] = TrialState(0, "running", _full(99.0), params={"opt": "adam"})
        after = _by_index(plan_successive_halving(perturbed, cfg))
        for i in (3, 4, 5):  # the sgd bracket
            self.assertEqual(base[i].action, after[i].action)
            self.assertEqual(base[i].verdict, after[i].verdict)

    def test_multi_key_bracket(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("opt", "dim")))
        trials = [
            TrialState(0, "running", _full(0.1), params={"opt": "adam", "dim": 64}),
            TrialState(1, "running", _full(0.2), params={"opt": "adam", "dim": 128}),
            TrialState(2, "running", _full(0.3), params={"opt": "sgd", "dim": 64}),
        ]
        self.assertEqual(len(bracket_partition(trials, cfg)), 3)

    def test_bracket_key_appears_on_actions(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        plan = _by_index(plan_successive_halving(self._trials(), cfg))
        self.assertIn("adam", plan[0].bracket)
        self.assertIn("sgd", plan[3].bracket)


class TestHyperbandSizes(unittest.TestCase):
    def test_matches_the_paper(self):
        # n_s = ceil((s_max+1)/(s+1) * eta^s), indexed by s.
        # s_max=3, eta=2 -> s=0: ceil(4/1 * 1)=4, s=1: ceil(4/2 * 2)=4,
        #                   s=2: ceil(4/3 * 4)=6, s=3: ceil(4/4 * 8)=8
        self.assertEqual(hyperband_sizes(3, 2), [4, 4, 6, 8])
        # s_max=2, eta=3 -> s=0: ceil(3/1*1)=3, s=1: ceil(3/2*3)=5, s=2: ceil(3/3*9)=9
        self.assertEqual(hyperband_sizes(2, 3), [3, 5, 9])

    def test_hedge_is_the_smallest_bracket(self):
        # The unpruned hedge should be a minority of the field — it's insurance,
        # not the plan.
        sizes = hyperband_sizes(4, 2)
        self.assertEqual(sizes[0], min(sizes))


class TestHyperbandBrackets(unittest.TestCase):
    def _trials(self, n, status="running", value=1.0):
        return [TrialState(i, status, _full(value + i)) for i in range(n)]

    def test_ladders_are_the_top_s_plus_1_rungs(self):
        # Bracket s starts at budget*eta^-s — Hyperband's r_s. With the ladder
        # [10,20,40,80]: s=3 -> the whole thing, s=1 -> [40,80], s=0 -> nothing.
        cfg = _cfg(BracketSpec(kind="hyperband", seed=0))
        brackets = bracket_partition(self._trials(24), cfg)
        rungs = rung_schedule(10, 80, 2)          # [10, 20, 40, 80]
        s_max = len(rungs) - 1
        self.assertEqual(brackets["s=0"].rungs, [], "the hedge must have NO rungs")
        self.assertEqual(brackets["s=1"].rungs, [40, 80])
        self.assertEqual(brackets["s=2"].rungs, [20, 40, 80])
        self.assertEqual(brackets["s=3"].rungs, [10, 20, 40, 80])
        # The most aggressive bracket gets the *full* ladder — if no bracket
        # does, the sweep never prunes at its earliest rung and the whole
        # aggressive end of Hyperband is missing.
        self.assertEqual(brackets[f"s={s_max}"].rungs, rungs)

    def test_higher_bracket_starts_earlier(self):
        # That gradient IS the algorithm: high s prunes aggressively from an
        # early rung, low s barely at all, s=0 not at all.
        cfg = _cfg(BracketSpec(kind="hyperband", seed=0))
        brackets = bracket_partition(self._trials(40), cfg)
        first = {k: (b.rungs[0] if b.rungs else None) for k, b in brackets.items()}
        self.assertEqual(first["s=3"], 10)
        self.assertEqual(first["s=2"], 20)
        self.assertEqual(first["s=1"], 40)
        self.assertIsNone(first["s=0"])

    def test_hedge_is_never_pruned_even_with_the_worst_metric(self):
        # The whole point of the hedge. Give every s=0 trial a catastrophic loss
        # and confirm none of them is ever pruned — not at any rung, not at the
        # budget. (A naive "single rung at budget" hedge WOULD prune these, after
        # burning their full compute. See the module docstring.)
        cfg = _cfg(BracketSpec(kind="hyperband", seed=0))
        s_max = len(rung_schedule(10, 80, 2)) - 1
        trials = []
        for i in range(40):
            s = hyperband_bracket(i, 0, s_max, 2)
            # s=0 trials get the worst possible metric in the sweep.
            trials.append(TrialState(i, "running", _full(999.0 if s == 0 else 0.1 + i)))
        hedge = [i for i in range(40) if hyperband_bracket(i, 0, s_max, 2) == 0]
        self.assertTrue(hedge, "seed produced no hedge trials; pick another")

        plan = _by_index(plan_successive_halving(trials, cfg))
        for i in hedge:
            self.assertNotEqual(
                plan[i].action, Action.PRUNE,
                f"hedge trial {i} was pruned despite being unprunable by design")
            self.assertEqual(plan[i].verdict, Verdict.NOT_AT_RUNG)
            self.assertIn("hedge", plan[i].reason)

    def test_assignment_is_stable_when_the_sweep_grows(self):
        # THE critical property. `append_trials` gives new trials fresh indices;
        # the planner reruns from scratch every tick. If adding trial 8 moved
        # trial 3 to a different bracket, trial 3 would change rung ladders
        # mid-flight and a promoted trial could become pruned. Assignment must
        # depend on the index alone, never on the field.
        s_max = 3
        before = {i: hyperband_bracket(i, 0, s_max, 2) for i in range(8)}
        after = {i: hyperband_bracket(i, 0, s_max, 2) for i in range(20)}
        for i in range(8):
            self.assertEqual(before[i], after[i], f"trial {i} changed bracket")

    def test_partition_membership_stable_under_append(self):
        # Same property, at the partition level.
        cfg = _cfg(BracketSpec(kind="hyperband", seed=0))
        small = bracket_partition(self._trials(8), cfg)
        large = bracket_partition(self._trials(20), cfg)
        for key, b in small.items():
            for idx in b.members:
                self.assertIn(idx, large[key].members,
                              f"trial {idx} left bracket {key} when the sweep grew")

    def test_seed_changes_the_assignment(self):
        a = [hyperband_bracket(i, 0, 3, 2) for i in range(30)]
        b = [hyperband_bracket(i, 7, 3, 2) for i in range(30)]
        self.assertNotEqual(a, b)

    def test_assignment_is_deterministic(self):
        a = [hyperband_bracket(i, 0, 3, 2) for i in range(30)]
        b = [hyperband_bracket(i, 0, 3, 2) for i in range(30)]
        self.assertEqual(a, b)

    def test_every_bracket_gets_used_on_a_reasonable_field(self):
        cfg = _cfg(BracketSpec(kind="hyperband", seed=0))
        brackets = bracket_partition(self._trials(60), cfg)
        self.assertEqual(sorted(brackets), ["s=0", "s=1", "s=2", "s=3"])


class TestLastSurvivorPerBracket(unittest.TestCase):
    """Bracketing shrinks cohorts, so the small-m path is now the common one."""

    def test_never_prunes_a_whole_bracket(self):
        for mode in ("sync", "asha"):
            for eta in (2, 3):
                for m in range(1, 6):
                    for vals in itertools.permutations(range(m)):
                        cfg = _cfg(
                            BracketSpec(kind="params", keys=("g",)),
                            min_steps=1, budget=8, eta=eta, mode=mode)
                        # Two brackets: `a` is the field under test, `b` is a
                        # decoy that must not rescue or condemn anyone in `a`.
                        trials = [
                            TrialState(i, "running", _full(float(v), upto=9),
                                       params={"g": "a"})
                            for i, v in enumerate(vals)
                        ] + [
                            TrialState(100 + j, "running", _full(0.001, upto=9),
                                       params={"g": "b"})
                            for j in range(3)
                        ]
                        plan = plan_successive_halving(trials, cfg)
                        pruned = sum(
                            1 for p in plan
                            if p.action == Action.PRUNE and p.index < 100)
                        self.assertLess(
                            pruned, m,
                            f"all {m} pruned in bracket 'a' "
                            f"(mode={mode}, eta={eta}, values={vals})")

    def test_singleton_bracket_is_never_pruned(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("g",)))
        trials = [
            TrialState(0, "running", _full(999.0), params={"g": "alone"}),
            TrialState(1, "running", _full(0.1), params={"g": "other"}),
            TrialState(2, "running", _full(0.2), params={"g": "other"}),
        ]
        plan = _by_index(plan_successive_halving(trials, cfg))
        self.assertNotEqual(plan[0].action, Action.PRUNE)


class TestBracketWarnings(unittest.TestCase):
    def test_warns_when_every_bracket_is_a_singleton(self):
        # Bracketing on a continuous param: every trial alone, nothing can ever
        # be pruned, and SH silently does nothing. This is the failure the
        # warning exists to make loud.
        cfg = _cfg(BracketSpec(kind="params", keys=("lr",)))
        trials = [
            TrialState(i, "running", _full(float(i)), params={"lr": 0.1 * i})
            for i in range(6)
        ]
        warnings = bracket_warnings(bracket_partition(trials, cfg), cfg)
        self.assertEqual(len(warnings), 1)
        self.assertIn("no-op", warnings[0])
        self.assertIn("none of them can prune", warnings[0])

    def test_silent_on_a_healthy_partition(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        trials = [
            TrialState(i, "running", _full(float(i)),
                       params={"opt": "adam" if i < 3 else "sgd"})
            for i in range(6)
        ]
        self.assertEqual(bracket_warnings(bracket_partition(trials, cfg), cfg), [])

    def test_partial_warning(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("opt",)))
        trials = [
            TrialState(0, "running", _full(0.1), params={"opt": "adam"}),
            TrialState(1, "running", _full(0.2), params={"opt": "adam"}),
            TrialState(2, "running", _full(0.3), params={"opt": "sgd"}),   # alone
        ]
        warnings = bracket_warnings(bracket_partition(trials, cfg), cfg)
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 of 2", warnings[0])

    def test_hedge_does_not_trigger_a_warning(self):
        # s=0 has no rungs BY DESIGN — it must not be reported as degenerate.
        cfg = _cfg(BracketSpec(kind="hyperband", seed=0))
        trials = [TrialState(i, "running", _full(0.1 + i)) for i in range(40)]
        brackets = bracket_partition(trials, cfg)
        self.assertEqual(brackets["s=0"].rungs, [])
        self.assertEqual(bracket_warnings(brackets, cfg), [])

    def test_asha_needs_eta_per_bracket(self):
        cfg = _cfg(BracketSpec(kind="params", keys=("g",)), eta=3, mode="asha")
        trials = [
            TrialState(0, "running", _full(0.1), params={"g": "a"}),
            TrialState(1, "running", _full(0.2), params={"g": "a"}),
        ]
        b = bracket_partition(trials, cfg)["g='a'"]
        self.assertFalse(bracket_can_prune(b, cfg))   # 2 < eta=3

    def test_ready_trials_do_not_count_toward_the_cohort(self):
        # A bracket of 5 where 4 are `ready` has a cohort of 1 — it can't prune,
        # and the warning must be based on the cohort, not the member count.
        cfg = _cfg(BracketSpec(kind="params", keys=("g",)))
        trials = [TrialState(0, "running", _full(0.1), params={"g": "a"})] + [
            TrialState(i, "ready", (), params={"g": "a"}) for i in range(1, 5)
        ]
        b = bracket_partition(trials, cfg)["g='a'"]
        self.assertEqual(len(b.members), 5)
        self.assertEqual(len(b.cohort), 1)
        self.assertFalse(bracket_can_prune(b, cfg))


class TestIdempotence(unittest.TestCase):
    def test_planning_twice_agrees(self):
        for spec in (
            BracketSpec(kind="params", keys=("opt",)),
            BracketSpec(kind="hyperband", seed=3),
        ):
            cfg = _cfg(spec)
            trials = [
                TrialState(i, "running", _full(float(i)),
                           params={"opt": "adam" if i % 2 else "sgd"})
                for i in range(12)
            ]
            a = plan_successive_halving(trials, cfg)
            b = plan_successive_halving(trials, cfg)
            self.assertEqual(
                [(p.index, p.action, p.verdict, p.bracket) for p in a],
                [(p.index, p.action, p.verdict, p.bracket) for p in b],
            )


if __name__ == "__main__":
    unittest.main()
