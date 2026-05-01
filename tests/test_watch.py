"""Tests for the watch daemon's pure components: state I/O, transition
detection, format adapters, and the Claude-summary fallback. The blocking
poll loop is exercised indirectly via `tick(once=True)` callers."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest, slurm, watch
from hyperherd.config import WatchConfig


def _trials(*pairs):
    """Build a minimal manifest list of (index, status) pairs."""
    return [
        {"index": i, "status": s, "experiment_name": f"exp{i}", "params": {"x": i}}
        for i, s in pairs
    ]


class TestState(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_state_roundtrip(self):
        # Empty before any save.
        self.assertEqual(watch.load_state(self.tmpdir), {})

        watch.save_state(self.tmpdir, {"last_seen": {"0": "running"}})
        loaded = watch.load_state(self.tmpdir)
        self.assertEqual(loaded["last_seen"], {"0": "running"})


class TestDetectEvents(unittest.TestCase):
    def test_first_tick_emits_nothing(self):
        # Daemon starting against an in-flight sweep should not replay history.
        trials = _trials((0, "completed"), (1, "failed"), (2, "running"))
        events, new_state = watch.detect_events(
            trials, state={}, enabled=["failed", "done"],
            heartbeat_seconds=None, now=1000.0,
        )
        self.assertEqual(events, [])
        self.assertEqual(new_state["last_seen"], {"0": "completed", "1": "failed", "2": "running"})

    def test_failed_fires_once_per_trial(self):
        prev = {"last_seen": {"0": "running", "1": "running"}}
        trials = _trials((0, "running"), (1, "failed"))
        events, new_state = watch.detect_events(
            trials, state=prev, enabled=["failed"],
            heartbeat_seconds=None, now=1000.0,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "trial_failed")
        self.assertEqual(events[0]["trial"]["index"], 1)

        # Same status next tick — no re-emit.
        events2, _ = watch.detect_events(
            trials, state=new_state, enabled=["failed"],
            heartbeat_seconds=None, now=1001.0,
        )
        self.assertEqual(events2, [])

    def test_cancelled_counts_as_failed_event(self):
        prev = {"last_seen": {"0": "running"}}
        trials = _trials((0, "cancelled"))
        events, _ = watch.detect_events(
            trials, state=prev, enabled=["failed"],
            heartbeat_seconds=None, now=1000.0,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "trial_failed")

    def test_done_fires_once_then_resets_on_resubmit(self):
        prev = {"last_seen": {"0": "running", "1": "running"}}
        trials = _trials((0, "completed"), (1, "completed"))
        events, new_state = watch.detect_events(
            trials, state=prev, enabled=["done"],
            heartbeat_seconds=None, now=1000.0,
        )
        self.assertEqual([e["event"] for e in events], ["sweep_done"])
        self.assertTrue(new_state["done_emitted"])

        # Tick again, no change → no re-emit.
        events2, new_state2 = watch.detect_events(
            trials, state=new_state, enabled=["done"],
            heartbeat_seconds=None, now=1001.0,
        )
        self.assertEqual(events2, [])
        self.assertTrue(new_state2["done_emitted"])

        # User resubmits trial 1 → done_emitted resets so the next
        # completion fires another `done`.
        trials_resub = _trials((0, "completed"), (1, "running"))
        _, new_state3 = watch.detect_events(
            trials_resub, state=new_state2, enabled=["done"],
            heartbeat_seconds=None, now=1002.0,
        )
        self.assertFalse(new_state3["done_emitted"])

        trials_done2 = _trials((0, "completed"), (1, "failed"))
        events4, _ = watch.detect_events(
            trials_done2, state=new_state3, enabled=["done"],
            heartbeat_seconds=None, now=1003.0,
        )
        self.assertEqual([e["event"] for e in events4], ["sweep_done"])

    def test_failed_event_disabled(self):
        # Two-trial sweep so a single failure isn't also a `done` trigger.
        prev = {"last_seen": {"0": "running", "1": "running"}}
        trials = _trials((0, "running"), (1, "failed"))
        events, _ = watch.detect_events(
            trials, state=prev, enabled=["done"],  # 'failed' not enabled
            heartbeat_seconds=None, now=1000.0,
        )
        self.assertEqual(events, [])

    def test_heartbeat_respects_interval(self):
        # First tick (no prior heartbeat): fires because totals differ from
        # the "no snapshot yet" baseline.
        trials = _trials((0, "running"), (1, "queued"))
        prev = {"last_seen": {"0": "running", "1": "queued"}}
        events, state1 = watch.detect_events(
            trials, state=prev, enabled=["heartbeat"],
            heartbeat_seconds=300, now=1000.0,
        )
        self.assertEqual([e["event"] for e in events], ["heartbeat"])
        self.assertEqual(state1["last_heartbeat_ts"], 1000.0)

        # Re-tick before the interval elapsed — no fire.
        events2, state2 = watch.detect_events(
            trials, state=state1, enabled=["heartbeat"],
            heartbeat_seconds=300, now=1100.0,
        )
        self.assertEqual(events2, [])
        self.assertEqual(state2["last_heartbeat_ts"], 1000.0)

    def test_heartbeat_skipped_when_totals_unchanged(self):
        trials = _trials((0, "running"))
        prev = {"last_seen": {"0": "running"}}
        # Prime: emits.
        _, state1 = watch.detect_events(
            trials, state=prev, enabled=["heartbeat"],
            heartbeat_seconds=60, now=1000.0,
        )
        # Way past the interval, but nothing changed — skip.
        events, state2 = watch.detect_events(
            trials, state=state1, enabled=["heartbeat"],
            heartbeat_seconds=60, now=10000.0,
        )
        self.assertEqual(events, [])
        # Timestamp is preserved (no false "we sent one").
        self.assertEqual(state2["last_heartbeat_ts"], 1000.0)


class TestRendering(unittest.TestCase):
    def test_render_failed(self):
        ev = {
            "event": "trial_failed",
            "trial": {"index": 7, "experiment_name": "lr-0.001", "status": "failed"},
            "totals": {"completed": 12, "failed": 1, "running": 4, "total": 19},
        }
        line = watch.render_line(ev, sweep_name="mnist")
        self.assertIn("[mnist]", line)
        self.assertIn("trial 7", line)
        self.assertIn("12/19 done", line)
        self.assertIn("4 running", line)
        self.assertIn("1 failed", line)

    def test_render_done(self):
        ev = {
            "event": "sweep_done",
            "trial": None,
            "totals": {"completed": 19, "total": 19},
        }
        line = watch.render_line(ev, sweep_name="mnist")
        self.assertIn("sweep complete", line)


class TestBuildRequest(unittest.TestCase):
    def setUp(self):
        self.event = {
            "event": "sweep_done",
            "trial": None,
            "totals": {"completed": 2, "total": 2},
        }

    def test_slack_format(self):
        req = watch.build_request("https://hooks.slack/x", "slack", self.event, "demo")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        body = json.loads(req.data)
        self.assertIn("text", body)
        self.assertIn("sweep complete", body["text"])

    def test_discord_format(self):
        req = watch.build_request("https://discord/x", "discord", self.event, "demo")
        body = json.loads(req.data)
        self.assertIn("content", body)

    def test_ntfy_format(self):
        req = watch.build_request("https://ntfy.sh/topic", "ntfy", self.event, "demo")
        self.assertEqual(req.get_header("Content-type"), "text/plain")
        self.assertIn(b"sweep complete", req.data)

    def test_raw_format(self):
        req = watch.build_request("https://example/x", "raw", self.event, "demo")
        body = json.loads(req.data)
        self.assertEqual(body["event"], "sweep_done")
        self.assertEqual(body["sweep"], "demo")
        self.assertIn("timestamp", body)

    def test_unknown_format_raises(self):
        with self.assertRaises(watch.WatchError):
            watch.build_request("https://x", "morse-code", self.event, "demo")


class TestBuildMessageRequest(unittest.TestCase):
    def test_slack_format_wraps_text(self):
        req = watch.build_message_request(
            "https://hooks.slack/x", "slack", "hello world", "demo"
        )
        self.assertEqual(req.get_header("Content-type"), "application/json")
        body = json.loads(req.data)
        self.assertEqual(body["text"], "[demo] hello world")

    def test_discord_format_uses_content_key(self):
        req = watch.build_message_request(
            "https://discord/x", "discord", "hi", "demo"
        )
        body = json.loads(req.data)
        self.assertEqual(body["content"], "[demo] hi")

    def test_ntfy_format_is_plain_text(self):
        req = watch.build_message_request(
            "https://ntfy.sh/topic", "ntfy", "ping", "demo"
        )
        self.assertEqual(req.get_header("Content-type"), "text/plain")
        self.assertEqual(req.data, b"[demo] ping")

    def test_raw_format_carries_message_event(self):
        req = watch.build_message_request(
            "https://example/x", "raw", "starting sweep", "demo"
        )
        body = json.loads(req.data)
        self.assertEqual(body["event"], "message")
        self.assertEqual(body["sweep"], "demo")
        self.assertEqual(body["text"], "starting sweep")
        self.assertIn("timestamp", body)

    def test_no_sweep_name_omits_prefix(self):
        req = watch.build_message_request(
            "https://ntfy.sh/topic", "ntfy", "bare", ""
        )
        self.assertEqual(req.data, b"bare")

    def test_unknown_format_raises(self):
        with self.assertRaises(watch.WatchError):
            watch.build_message_request("https://x", "smoke-signal", "hi", "demo")


class TestFailureSummarize(unittest.TestCase):
    """`_summarize_failure_with_claude` is opt-in eye-candy on failure events.
    Any error from the subprocess (missing binary, non-zero exit, timeout)
    must degrade silently to None so the daemon can still emit the raw
    stderr-tail fallback."""

    def _trial(self):
        return {"index": 0, "experiment_name": "exp0", "params": {}}

    def _failure(self):
        return {"cause": "TIMEOUT", "stderr_tail": "boom\n"}

    def test_no_claude_binary(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("claude")):
            out = watch._summarize_failure_with_claude(
                "demo", self._trial(), self._failure()
            )
        self.assertIsNone(out)

    def test_claude_nonzero_exit(self):
        proc = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch("subprocess.run", return_value=proc):
            out = watch._summarize_failure_with_claude(
                "demo", self._trial(), self._failure()
            )
        self.assertIsNone(out)

    def test_claude_success(self):
        proc = mock.Mock(returncode=0, stdout="Hit the wallclock.\n", stderr="")
        with mock.patch("subprocess.run", return_value=proc):
            out = watch._summarize_failure_with_claude(
                "demo", self._trial(), self._failure()
            )
        self.assertEqual(out, "Hit the wallclock.")


class TestStderrTail(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        self.log_dir = manifest.logs_path(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_err(self, idx, content):
        with open(os.path.join(self.log_dir, f"{idx}.err"), "w") as f:
            f.write(content)

    def test_no_log_file(self):
        text, truncated = watch._read_stderr_tail(self.tmpdir, 99)
        self.assertEqual(text, "")
        self.assertFalse(truncated)

    def test_short_log_returns_full(self):
        self._write_err(0, "line1\nline2\nline3\n")
        text, truncated = watch._read_stderr_tail(self.tmpdir, 0)
        self.assertEqual(text, "line1\nline2\nline3")
        self.assertFalse(truncated)

    def test_line_cap(self):
        # 30 short lines — only the last 20 should survive.
        self._write_err(0, "".join(f"line{i}\n" for i in range(30)))
        text, truncated = watch._read_stderr_tail(self.tmpdir, 0)
        self.assertTrue(truncated)
        first = text.split("\n", 1)[0]
        self.assertEqual(first, "line10")

    def test_byte_cap(self):
        # One absurdly long line — should be byte-capped even though it's
        # a single line.
        big = "x" * 5000 + "\n"
        self._write_err(0, big)
        text, truncated = watch._read_stderr_tail(self.tmpdir, 0)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text.encode()), watch._STDERR_TAIL_BYTES + 1)


class TestParseFailureInfo(unittest.TestCase):
    """slurm.parse_failure_info — pure parser for sacct output."""

    def test_timeout(self):
        out = "12345_7|TIMEOUT|0:0|None\n12345_7.batch|TIMEOUT|0:0|\n"
        info = slurm.parse_failure_info(out, "12345_7")
        self.assertEqual(info.state, "TIMEOUT")
        self.assertEqual(info.exit_code, 0)
        self.assertEqual(info.signal, 0)

    def test_out_of_memory(self):
        out = "12345_7|OUT_OF_MEMORY|0:9|None\n"
        info = slurm.parse_failure_info(out, "12345_7")
        self.assertEqual(info.state, "OUT_OF_MEMORY")
        self.assertEqual(info.signal, 9)

    def test_failed_with_exit_code(self):
        out = "12345_7|FAILED|1:0|None\n"
        info = slurm.parse_failure_info(out, "12345_7")
        self.assertEqual(info.state, "FAILED")
        self.assertEqual(info.exit_code, 1)
        self.assertEqual(info.signal, 0)

    def test_cancelled_carries_reason(self):
        out = "12345_7|CANCELLED+|0:0|JobOutOfTime\n"
        info = slurm.parse_failure_info(out, "12345_7")
        # `CANCELLED+` collapses to the first whitespace-delimited token.
        self.assertEqual(info.state, "CANCELLED+")
        self.assertEqual(info.reason, "JobOutOfTime")

    def test_target_not_found(self):
        out = "99999_0|FAILED|1:0|None\n"
        info = slurm.parse_failure_info(out, "12345_7")
        self.assertEqual(info.state, "UNKNOWN")  # default

    def test_ignores_step_rows(self):
        # Both rows present; parent wins, step is ignored.
        out = (
            "12345_7|TIMEOUT|0:0|None\n"
            "12345_7.batch|FAILED|1:0|\n"
        )
        info = slurm.parse_failure_info(out, "12345_7")
        self.assertEqual(info.state, "TIMEOUT")


class TestHumanCause(unittest.TestCase):
    def test_timeout(self):
        info = slurm.FailureInfo(state="TIMEOUT", exit_code=0, signal=0)
        self.assertEqual(watch._human_cause(info), "TIMEOUT")

    def test_oom(self):
        info = slurm.FailureInfo(state="OUT_OF_MEMORY")
        self.assertEqual(watch._human_cause(info), "OUT_OF_MEMORY")

    def test_signal_named(self):
        info = slurm.FailureInfo(state="FAILED", signal=11)
        self.assertEqual(watch._human_cause(info), "SIGSEGV")

    def test_signal_unnamed(self):
        info = slurm.FailureInfo(state="FAILED", signal=7)
        self.assertEqual(watch._human_cause(info), "signal 7")

    def test_exit_code(self):
        info = slurm.FailureInfo(state="FAILED", exit_code=42, signal=0)
        self.assertEqual(watch._human_cause(info), "exit code 42")

    def test_cancelled(self):
        info = slurm.FailureInfo(state="CANCELLED")
        self.assertEqual(watch._human_cause(info), "CANCELLED")


class TestFailureRendering(unittest.TestCase):
    def _failed_event(self, with_failure=True, with_summary=False):
        ev = {
            "event": "trial_failed",
            "trial": {
                "index": 7,
                "experiment_name": "lr-0.001_opt-sgd",
                "status": "failed",
                "params": {"lr": 0.001, "optimizer": "sgd"},
            },
            "totals": {"completed": 12, "failed": 1, "running": 4, "total": 19},
            "summary": "Hit the GPU memory limit." if with_summary else None,
        }
        if with_failure:
            ev["failure"] = {
                "cause": "TIMEOUT",
                "slurm_state": "TIMEOUT",
                "exit_code": 0,
                "signal": 0,
                "reason": "",
                "job_id": "12345",
                "stderr_tail": "Traceback\n  RuntimeError: boom",
                "stderr_truncated": False,
            }
        return ev

    def test_render_line_includes_cause(self):
        line = watch.render_line(self._failed_event(), sweep_name="mnist")
        self.assertIn("(TIMEOUT)", line)
        self.assertIn("trial 7", line)

    def test_render_line_no_cause_when_failure_missing(self):
        line = watch.render_line(
            self._failed_event(with_failure=False), sweep_name="mnist"
        )
        self.assertNotIn("(TIMEOUT)", line)
        self.assertIn("trial 7", line)

    def test_slack_body_includes_stderr_tail(self):
        req = watch.build_request(
            "https://slack/x", "slack", self._failed_event(), "mnist"
        )
        body = json.loads(req.data)
        self.assertIn("```", body["text"])
        self.assertIn("RuntimeError", body["text"])

    def test_slack_body_uses_summary_when_present(self):
        # When Claude summary is set, the stderr block should be replaced.
        req = watch.build_request(
            "https://slack/x",
            "slack",
            self._failed_event(with_summary=True),
            "mnist",
        )
        body = json.loads(req.data)
        self.assertIn("Hit the GPU memory limit.", body["text"])
        self.assertNotIn("```", body["text"])
        self.assertNotIn("RuntimeError", body["text"])

    def test_truncation_marker_in_code_block(self):
        ev = self._failed_event()
        ev["failure"]["stderr_truncated"] = True
        req = watch.build_request("https://slack/x", "slack", ev, "mnist")
        body = json.loads(req.data)
        self.assertIn("truncated", body["text"])

    def test_raw_includes_full_failure_block(self):
        req = watch.build_request(
            "https://example/x", "raw", self._failed_event(), "mnist"
        )
        body = json.loads(req.data)
        self.assertEqual(body["failure"]["cause"], "TIMEOUT")
        self.assertEqual(body["failure"]["stderr_tail"], "Traceback\n  RuntimeError: boom")

    def test_non_failure_event_has_no_extra_block(self):
        ev = {
            "event": "sweep_done",
            "trial": None,
            "totals": {"completed": 19, "total": 19},
            "summary": None,
        }
        req = watch.build_request("https://slack/x", "slack", ev, "mnist")
        body = json.loads(req.data)
        self.assertNotIn("```", body["text"])


class TestWatchConfigValidation(unittest.TestCase):
    def test_heartbeat_event_requires_interval(self):
        with self.assertRaises(Exception):
            # Heartbeat enabled but explicitly no interval — invalid combo.
            WatchConfig(events=["heartbeat"], heartbeat_minutes=None)

    def test_default_events_include_heartbeat(self):
        cfg = WatchConfig(webhook="https://x")
        self.assertEqual(cfg.events, ["failed", "done", "heartbeat"])
        self.assertEqual(cfg.heartbeat_minutes, 5)
        self.assertEqual(cfg.format, "raw")


class TestDefaultWebhook(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_topic_persists_across_calls(self):
        url1, topic1 = watch.resolve_default_webhook(self.tmpdir, "mnist sweep")
        url2, topic2 = watch.resolve_default_webhook(self.tmpdir, "mnist sweep")
        self.assertEqual(url1, url2)
        self.assertEqual(topic1, topic2)
        self.assertTrue(url1.startswith("https://ntfy.sh/herd-mnist-sweep-"))

    def test_topic_has_random_suffix(self):
        # Two fresh workspaces should get different random topics.
        other = tempfile.mkdtemp()
        try:
            manifest.init_workspace(other)
            _, topic_a = watch.resolve_default_webhook(self.tmpdir, "demo")
            _, topic_b = watch.resolve_default_webhook(other, "demo")
            self.assertNotEqual(topic_a, topic_b)
        finally:
            shutil.rmtree(other)

    def test_slug_falls_back_for_unsafe_name(self):
        url, topic = watch.resolve_default_webhook(self.tmpdir, "@@@!!!")
        # All non-alphanumerics collapse — slug becomes "sweep" placeholder.
        self.assertTrue(topic.startswith("herd-sweep-"))
        self.assertIn("/herd-sweep-", url)

    def test_topic_survives_event_writes(self):
        # Generate the topic, then run a tick-style state save and confirm
        # default_topic isn't clobbered by detect_events' rewrite.
        _, topic = watch.resolve_default_webhook(self.tmpdir, "demo")

        prev = watch.load_state(self.tmpdir)
        events, new_state = watch.detect_events(
            trials=_trials((0, "running")),
            state=prev,
            enabled=["failed"],
            heartbeat_seconds=None,
            now=1000.0,
        )
        watch.save_state(self.tmpdir, new_state)

        reloaded = watch.load_state(self.tmpdir)
        self.assertEqual(reloaded["default_topic"], topic)


if __name__ == "__main__":
    unittest.main()
