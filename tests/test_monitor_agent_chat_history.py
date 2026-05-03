"""Tests for the chat-history rolling buffer.

The buffer captures real conversation (agent `msg` posts and user inbox
messages) but excludes per-tick heartbeats (`tick_summary`). It's stored
at .hyperherd/chat-history.jsonl and trimmed to a small fixed size so
the agent can stitch its past questions to the user's replies across
ticks without prompt bloat.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from hyperherd.monitor_agent import state as state_mod
from hyperherd.monitor_agent.tools import (
    CHAT_HISTORY_FILENAME, CHAT_HISTORY_KEEP, record_chat_entry,
)


class TestChatHistoryRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _read_history(self):
        path = self.workspace / ".hyperherd" / CHAT_HISTORY_FILENAME
        if not path.is_file():
            return []
        return [
            json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()
        ]

    def test_record_round_trip(self):
        record_chat_entry(
            self.workspace,
            role="agent", text="Herd dog: hi", via="discord", author="Herd dog",
        )
        record_chat_entry(
            self.workspace,
            role="user", text="how's it going", via="discord", author="alice",
        )

        entries = self._read_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["role"], "agent")
        self.assertEqual(entries[0]["author"], "Herd dog")
        self.assertEqual(entries[1]["role"], "user")
        self.assertEqual(entries[1]["text"], "how's it going")

    def test_trims_to_keep_limit(self):
        """Once we exceed CHAT_HISTORY_KEEP, the oldest drop off."""
        for i in range(CHAT_HISTORY_KEEP + 5):
            record_chat_entry(
                self.workspace,
                role="agent", text=f"msg-{i}",
                via="discord", author="Herd dog",
            )

        entries = self._read_history()
        self.assertEqual(len(entries), CHAT_HISTORY_KEEP)
        # The last KEEP messages are the most recent ones; the earliest
        # should have been evicted.
        last_idx = CHAT_HISTORY_KEEP + 4  # we recorded 0..(K+4)
        first_kept = last_idx - CHAT_HISTORY_KEEP + 1
        self.assertEqual(entries[0]["text"], f"msg-{first_kept}")
        self.assertEqual(entries[-1]["text"], f"msg-{last_idx}")

    def test_drain_inbox_mirrors_to_chat_history(self):
        """When state.compute drains the inbox, each user message also
        lands in chat-history so the agent has cross-tick context."""
        # Seed an inbox with two user messages.
        inbox_path = self.workspace / ".hyperherd" / state_mod.INBOX_FILE
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        inbox_path.write_text(
            json.dumps({
                "timestamp": "2026-05-03T00:00:00",
                "source": "discord", "author": "alice", "text": "pause",
            }) + "\n" +
            json.dumps({
                "timestamp": "2026-05-03T00:00:01",
                "source": "discord", "author": "alice", "text": "actually go",
            }) + "\n"
        )

        # Drain — the helper itself doesn't need a snapshot.
        msgs = state_mod._drain_inbox(self.workspace)
        self.assertEqual(len(msgs), 2)

        history = self._read_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["text"], "pause")
        # Timestamps are preserved from the inbox entries.
        self.assertEqual(history[0]["timestamp"], "2026-05-03T00:00:00")
        # The inbox file is gone after drain (rename + unlink); the next
        # writer recreates it. Either absent or empty is acceptable.
        if inbox_path.exists():
            self.assertEqual(inbox_path.read_text(), "")


class TestDrainInboxAtomicity(unittest.TestCase):
    """The previous read-then-truncate path lost messages that arrived
    in the gap. Now uses rename → read → unlink so writes that happen
    *during* the drain still land on disk and aren't silently dropped.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)
        (self.workspace / ".hyperherd").mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _seed_inbox(self, lines):
        path = self.workspace / ".hyperherd" / state_mod.INBOX_FILE
        path.write_text("\n".join(lines) + "\n")

    def test_drain_uses_rename_not_truncate(self):
        """A writer that opens after the drain should see an empty
        (or absent) inbox.jsonl — its append creates a fresh file
        whose contents are NOT consumed by this drain."""
        import json as _json
        self._seed_inbox([
            _json.dumps({"timestamp": "t1", "source": "discord",
                         "author": "alice", "text": "msg1"}),
        ])

        # Simulate "writer appends after our read but before our cleanup"
        # by patching the unlink step to run a writer first.
        path = self.workspace / ".hyperherd" / state_mod.INBOX_FILE

        original_unlink = Path.unlink
        late_writes = []

        def unlink_with_writer_in_between(self_path):
            if self_path.name.endswith(".draining"):
                # Simulate a concurrent writer: appender opens the new
                # inbox.jsonl (which doesn't exist yet, so creates fresh).
                with open(path, "a") as f:
                    f.write(_json.dumps({
                        "timestamp": "t2", "source": "discord",
                        "author": "bob", "text": "msg2",
                    }) + "\n")
                late_writes.append("written")
            return original_unlink(self_path)

        from unittest import mock as _m
        with _m.patch.object(Path, "unlink", unlink_with_writer_in_between):
            msgs = state_mod._drain_inbox(self.workspace)

        # First drain returns msg1 (was in the file at rename time).
        self.assertEqual([m.text for m in msgs], ["msg1"])
        self.assertEqual(late_writes, ["written"])

        # Critically: msg2 is preserved — it's in the fresh inbox.jsonl
        # that the writer created after the rename. Next drain picks it up.
        msgs2 = state_mod._drain_inbox(self.workspace)
        self.assertEqual([m.text for m in msgs2], ["msg2"])


class TestStateReadsChatHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_read_chat_history_parses_entries(self):
        record_chat_entry(self.workspace, role="agent",
                          text="Herd dog: starting", via="discord",
                          author="Herd dog")
        record_chat_entry(self.workspace, role="user",
                          text="thanks", via="discord", author="alice")

        entries = state_mod._read_chat_history(self.workspace)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].role, "agent")
        self.assertEqual(entries[1].role, "user")
        self.assertEqual(entries[1].author, "alice")


class TestChatHistoryInvariant(unittest.TestCase):
    """Lock in the design rule: chat_history contains agent `msg` posts
    and user mentions/replies, and nothing else. In particular:

    - `tick_summary` writes nothing (it's the heartbeat, not conversation).
    - Slash commands never reach the chat history (they handle their
      own UI via Discord interactions, no record_chat_entry call).
    - The daemon's direct `channel.post` for the final-stop notification
      doesn't record either (it's a daemon-level event, not agent voice).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _hist(self):
        path = self.workspace / ".hyperherd" / "chat-history.jsonl"
        if not path.is_file():
            return []
        return [
            __import__("json").loads(ln)
            for ln in path.read_text().splitlines() if ln.strip()
        ]

    def test_only_msg_and_drained_inbox_appear(self):
        """The two legitimate writers — `record_chat_entry` from the msg
        tool, and the inbox-drain mirror in state.compute — both produce
        records. Nothing else writes to the file."""
        # Agent msg: records.
        record_chat_entry(self.workspace, role="agent",
                          text="Herd dog: a real reply",
                          via="discord", author="Herd dog")

        # User mention via inbox-drain (the only path that should write
        # role=user entries — slash commands and plain chatter never
        # reach _drain_inbox under the new on_message gating).
        import json as _json
        inbox_path = self.workspace / ".hyperherd" / state_mod.INBOX_FILE
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        inbox_path.write_text(_json.dumps({
            "timestamp": "2026-05-03T00:00:00",
            "source": "discord", "author": "alice",
            "text": "what's idx 3 doing?",
        }) + "\n")
        state_mod._drain_inbox(self.workspace)

        hist = self._hist()
        self.assertEqual(len(hist), 2)
        roles = [h["role"] for h in hist]
        self.assertEqual(set(roles), {"agent", "user"})
        # No "system" / "heartbeat" / "command" or other roles leak in.
        for h in hist:
            self.assertIn(h["role"], ("agent", "user"))


if __name__ == "__main__":
    unittest.main()
