"""Tests for `herd monitor -p MSG`: the one-shot path that seeds the inbox
with a message and runs a single user-message tick with a console channel."""

import argparse
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import cmd_monitor


def _write_config(base):
    cfg = (
        "name: t\n"
        f"workspace: {base}\n"
        f"launcher: {os.path.join(base, 'launch.sh')}\n"
        "grid: [lr]\n"
        "parameters:\n"
        "  lr:\n"
        "    type: discrete\n"
        "    abbrev: lr\n"
        "    values: [0.1, 0.2]\n"
        "slurm:\n"
        "  partition: p\n"
        "  time: '00:10:00'\n"
        "  mem: 1G\n"
        "  cpus_per_task: 1\n"
    )
    with open(os.path.join(base, "hyperherd.yaml"), "w") as f:
        f.write(cfg)
    with open(os.path.join(base, "launch.sh"), "w") as f:
        f.write("#!/bin/bash\n")
    os.chmod(os.path.join(base, "launch.sh"), 0o755)


class _FakeResult:
    cost_usd = 0.01
    turns = 2
    halted = False
    halt_reason = None
    next_delay_seconds = 1800


class TestMonitorPrompt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp, [{"lr": 0.1}, {"lr": 0.2}], abbrevs={"lr": "lr"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _args(self, **over):
        base = dict(
            workspace=self.tmp, once=False, dry_run=False, prompt=None,
            trigger="boot", max_ticks=None, no_agent=False, force_discord=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_prompt_seeds_inbox_and_runs_user_message_tick(self):
        captured = {}

        async def fake_run_tick(workspace, trigger="scheduled", *, channel=None, **kw):
            # The inbox should already hold our message by the time the tick runs.
            inbox = os.path.join(self.tmp, ".hyperherd", "inbox.jsonl")
            with open(inbox) as f:
                captured["inbox"] = [json.loads(l) for l in f if l.strip()]
            captured["trigger"] = trigger
            captured["channel"] = channel
            return _FakeResult()

        with mock.patch(
            "hyperherd.monitor_agent.tick.run_tick", side_effect=fake_run_tick
        ), mock.patch("hyperherd.cli._apply_workspace_env"):
            rc = cmd_monitor(self._args(prompt="pause trial 1 please"))

        self.assertEqual(rc, 0)
        self.assertEqual(captured["trigger"], "user_message")
        self.assertEqual(captured["channel"].name, "console")
        self.assertEqual(len(captured["inbox"]), 1)
        msg = captured["inbox"][0]
        self.assertEqual(msg["text"], "pause trial 1 please")
        self.assertEqual(msg["source"], "cli")

    def test_prompt_does_not_run_preflight_or_daemon(self):
        async def fake_run_tick(workspace, trigger="scheduled", *, channel=None, **kw):
            return _FakeResult()

        with mock.patch(
            "hyperherd.monitor_agent.tick.run_tick", side_effect=fake_run_tick
        ), mock.patch("hyperherd.cli._apply_workspace_env"), \
             mock.patch("hyperherd.cli._monitor_preflight") as preflight, \
             mock.patch("hyperherd.monitor_agent.daemon.run_daemon") as daemon:
            rc = cmd_monitor(self._args(prompt="status?"))

        self.assertEqual(rc, 0)
        preflight.assert_not_called()
        daemon.assert_not_called()


if __name__ == "__main__":
    unittest.main()
