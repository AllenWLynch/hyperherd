"""Tests for terminal output formatting."""

import io
import re
import unittest
from contextlib import redirect_stdout

from hyperherd.display import (
    _condense_case_block,
    format_short_value,
    print_status_table,
    trial_sort_key,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render(trials, *, brief=False, log_tails=None):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_status_table(trials, log_tails or {}, brief=brief)
    return _ANSI_RE.sub("", buf.getvalue())


class TestStatusTable(unittest.TestCase):
    def _trial(self, idx, status, name=None, params=None):
        t = {"index": idx, "status": status, "params": params or {}}
        if name is not None:
            t["experiment_name"] = name
        return t

    def test_shows_experiment_name_not_params(self):
        out = _render([
            self._trial(0, "running", name="mix-hi_lr-3e-05",
                        params={"dataset_config": "configs/train/data/x.yaml"}),
        ])
        self.assertIn("Name", out)
        self.assertNotIn("Params", out)
        self.assertIn("mix-hi_lr-3e-05", out)
        # The long param path that used to get cut off is not shown.
        self.assertNotIn("configs/train/data/x.yaml", out)

    def test_falls_back_to_params_without_name(self):
        out = _render([self._trial(0, "running", params={"lr": 0.01})])
        self.assertIn("lr=0.01", out)

    def test_brief_hides_ready_and_cancelled(self):
        out = _render([
            self._trial(0, "ready", name="r"),
            self._trial(1, "cancelled", name="c"),
            self._trial(2, "running", name="run"),
        ], brief=True)
        self.assertIn("run", out)
        self.assertNotIn("READY", out)
        self.assertNotIn("CANCELLED", out)

    def test_brief_sorts_active_first(self):
        out = _render([
            self._trial(0, "completed", name="done"),
            self._trial(1, "running", name="live"),
            self._trial(2, "failed", name="broke"),
        ], brief=True)
        rows = [ln for ln in out.splitlines() if "RUNNING" in ln or
                "FAILED" in ln or "COMPLETED" in ln]
        # running < failed < completed by status priority.
        self.assertTrue(rows[0].strip().startswith("1"))   # running
        self.assertIn("FAILED", rows[1])
        self.assertIn("COMPLETED", rows[2])

    def test_log_tail_ansi_stripped_no_bleed(self):
        # A wandb-style log tail with an opened-but-unterminated color must not
        # leak ANSI into the rendered table. Render WITHOUT our test-side strip
        # to prove the table itself emitted no foreign escape.
        colored = "\x1b[33mView run \x1b[1;34mmy-run-name that is quite long enough"
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_status_table(
                [self._trial(0, "running", name="run")], {0: colored},
            )
        raw = buf.getvalue()
        # The only ANSI allowed is the table's own Status colorize (which is
        # always paired with a reset); the borrowed log color must be gone.
        self.assertNotIn("\x1b[33m", raw)
        self.assertNotIn("\x1b[1;34m", raw)
        self.assertIn("View run", _ANSI_RE.sub("", raw))

    def test_sort_key_priority(self):
        self.assertLess(
            trial_sort_key({"status": "running", "index": 9}),
            trial_sort_key({"status": "completed", "index": 0}),
        )


class TestCondenseCaseBlock(unittest.TestCase):
    """The dry-run printer trims the baked lookup case block for readability."""

    def _script(self, n_trials: int) -> str:
        arms = []
        for i in range(n_trials):
            arms.append(f"  {i})")
            arms.append(f"    HYPERHERD_TRIAL_NAME=trial-{i}")
            arms.append(f"    OVERRIDES='lr={0.1 ** i}'")
            arms.append("    ;;")
        return "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --array=0-{}".format(n_trials - 1),
                "",
                'case "$SLURM_ARRAY_TASK_ID" in',
                *arms,
                "  *)",
                '    echo "no entry" >&2',
                "    exit 1",
                "    ;;",
                "esac",
                "bash launch.sh \"$OVERRIDES\"",
            ]
        )

    def test_short_block_unchanged(self):
        # 1-2 trials: condensing would actually be longer than the original.
        script = self._script(2)
        self.assertEqual(_condense_case_block(script), script)

    def test_long_block_elided(self):
        script = self._script(20)
        out = _condense_case_block(script)
        # First arm preserved.
        self.assertIn("  0)\n", out)
        self.assertIn("HYPERHERD_TRIAL_NAME=trial-0", out)
        # Middle arms gone.
        self.assertNotIn("HYPERHERD_TRIAL_NAME=trial-10", out)
        self.assertNotIn("HYPERHERD_TRIAL_NAME=trial-19", out)
        # Wildcard arm preserved (so the user sees the safety net).
        self.assertIn("  *)", out)
        self.assertIn("exit 1", out)
        # Elision marker is visible and counts what's hidden.
        self.assertIn("19 more trial arm(s) elided", out)

    def test_no_case_block_passthrough(self):
        # A script without a case block (shouldn't happen in practice, but
        # the condenser must not corrupt it).
        plain = "#!/bin/bash\necho hello\n"
        self.assertEqual(_condense_case_block(plain), plain)


class TestFormatShortValue(unittest.TestCase):
    def test_float_uses_4g(self):
        self.assertEqual(format_short_value(0.0001234), "0.0001234")
        self.assertEqual(format_short_value(1234567.89), "1.235e+06")

    def test_int_unchanged(self):
        self.assertEqual(format_short_value(42), "42")

    def test_string_unchanged(self):
        self.assertEqual(format_short_value("adam"), "adam")


if __name__ == "__main__":
    unittest.main()
