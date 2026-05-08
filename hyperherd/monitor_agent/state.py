"""Deterministic per-tick state assembler.

Runs in pure Python before the agent is invoked. Pulls a fresh `herd
snapshot`, rotates the previous-tick snapshot for delta detection, drains
the inbox of any user messages received since last tick, and packages it
all into a single `TickState` the agent's `read_state()` tool returns.

The agent never calls `herd snapshot`, never reads the manifest, never
diffs anything itself — that's the point.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


SNAPSHOT_FILE = "last-snapshot.json"
PREV_SNAPSHOT_FILE = "last-snapshot.prev.json"
INBOX_FILE = "inbox.jsonl"
PLAN_FILE = "MONITOR_PLAN.md"
OUTBOUND_FILE = "last-outbound.jsonl"
USER_PROMPT_FILE = "PROMPT.md"   # workspace-root, user-authored standing instructions


TickTrigger = Literal["scheduled", "failure", "completion", "user_message", "boot"]


@dataclass
class FailureView:
    """Diff entry: a trial that newly entered `failed` since the last snapshot."""
    index: int
    experiment_name: Optional[str]
    slurm_state: Optional[str]
    stderr_tail: List[str] = field(default_factory=list)


@dataclass
class InboundMessage:
    """A Discord (or other source) message received since the last tick."""
    timestamp: str       # ISO-8601 UTC
    source: str          # "discord" / "slack" / etc.
    author: str
    text: str


@dataclass
class ChatEntry:
    """A recent message on either side of the conversation. Heartbeats
    (the obligatory per-tick `tick_summary` posts) are deliberately not
    recorded — only real conversation goes here so the agent can stitch
    questions to replies across ticks without noise."""
    timestamp: str
    role: str             # "user" | "agent"
    author: str           # Discord username for user; "agent" for the bot
    via: str              # "discord" / "webhook" / etc.
    text: str


@dataclass
class UserPromptView:
    """The workspace-level PROMPT.md, surfaced to the agent only when it
    differs from the last hash the agent acknowledged via
    `mark_user_prompt_read(hash)`. `status` is one of:

      - "new":       agent has never acknowledged a hash
      - "changed":   on-disk hash differs from the last acknowledged one
      - "unchanged": hashes match (in this case `text` is omitted to save tokens)
    """
    status: Literal["new", "changed", "unchanged"]
    sha256: str
    text: Optional[str]   # populated when status != "unchanged"


@dataclass
class TickState:
    """The single document the agent reads at the start of every tick."""
    sweep_name: str
    workspace: str                      # absolute path
    trigger: TickTrigger
    plan: str                           # MONITOR_PLAN.md contents (empty on first tick)
    totals: Dict[str, int]              # status -> count, plus "total"
    trials: List[Dict[str, Any]]        # straight from `herd snapshot`'s trials[]
    newly_failed: List[FailureView]
    newly_completed: List[int]
    newly_pruned: List[int]
    inbox: List[InboundMessage]
    chat_history: List[ChatEntry]   # rolling buffer of recent real messages
    user_prompt: Optional["UserPromptView"] = None   # PROMPT.md state (None if file absent)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form — what `read_state()` hands the agent."""
        return {
            "sweep_name": self.sweep_name,
            "workspace": self.workspace,
            "trigger": self.trigger,
            "plan": self.plan,
            "totals": self.totals,
            "trials": self.trials,
            "newly_failed": [asdict(f) for f in self.newly_failed],
            "newly_completed": self.newly_completed,
            "newly_pruned": self.newly_pruned,
            "inbox": [asdict(m) for m in self.inbox],
            "chat_history": [asdict(m) for m in self.chat_history],
            "user_prompt": asdict(self.user_prompt) if self.user_prompt else None,
        }


# --- snapshot rotation ------------------------------------------------------

def _hyperherd_dir(workspace: Path) -> Path:
    return workspace / ".hyperherd"


def refresh_snapshot(workspace: Path) -> Dict[str, Any]:
    """Public helper — refresh `.hyperherd/last-snapshot.json` without
    running an agent tick. Used by passive monitor mode (no agent loop)
    so the dashboard / heartbeat keep showing fresh totals."""
    return _rotate_and_capture(Path(workspace).resolve())


def _rotate_and_capture(workspace: Path) -> Dict[str, Any]:
    """Move the previous snapshot to .prev, fetch a fresh one, write it.

    Returns the parsed current snapshot dict. If no previous snapshot
    exists (first ever tick), the .prev file is left absent — diff
    helpers handle that case.
    """
    hh = _hyperherd_dir(workspace)
    hh.mkdir(parents=True, exist_ok=True)
    cur_path = hh / SNAPSHOT_FILE
    prev_path = hh / PREV_SNAPSHOT_FILE

    if cur_path.exists():
        shutil.copyfile(cur_path, prev_path)

    # Use `python -m hyperherd.cli` rather than the `herd` script so this
    # works regardless of whether `herd` is on PATH (e.g. during dev / tests
    # before `pip install -e .`).
    proc = subprocess.run(
        [sys.executable, "-m", "hyperherd.cli", "snapshot", str(workspace)],
        capture_output=True, text=True, check=True,
    )
    cur_path.write_text(proc.stdout)
    return json.loads(proc.stdout)


def _read_prev(workspace: Path) -> Optional[Dict[str, Any]]:
    prev_path = _hyperherd_dir(workspace) / PREV_SNAPSHOT_FILE
    if not prev_path.is_file():
        return None
    try:
        return json.loads(prev_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --- diff helpers -----------------------------------------------------------

def _indices_with_status(snapshot: Dict[str, Any], status: str) -> set:
    return {t["index"] for t in snapshot.get("trials", []) if t.get("status") == status}


def _diff_failed(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[FailureView]:
    """Indices that newly entered `failed` since the previous snapshot."""
    cur_failed = _indices_with_status(cur, "failed")
    prev_failed = _indices_with_status(prev, "failed") if prev else set()
    new_idx = cur_failed - prev_failed

    cur_trials = {t["index"]: t for t in cur.get("trials", [])}
    failed_stderr = {b["index"]: b for b in cur.get("failed_stderr", [])}

    out: List[FailureView] = []
    for idx in sorted(new_idx):
        trial = cur_trials.get(idx, {})
        block = failed_stderr.get(idx, {})
        out.append(FailureView(
            index=idx,
            experiment_name=trial.get("experiment_name"),
            slurm_state=trial.get("slurm_state"),
            stderr_tail=block.get("stderr_lines") or [],
        ))
    return out


def _diff_completed(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[int]:
    cur_done = _indices_with_status(cur, "completed")
    prev_done = _indices_with_status(prev, "completed") if prev else set()
    return sorted(cur_done - prev_done)


def _diff_pruned(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[int]:
    cur_pruned = _indices_with_status(cur, "pruned")
    prev_pruned = _indices_with_status(prev, "pruned") if prev else set()
    return sorted(cur_pruned - prev_pruned)


# --- plan + inbox -----------------------------------------------------------

def _read_plan(workspace: Path) -> str:
    path = _hyperherd_dir(workspace) / PLAN_FILE
    if not path.is_file():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _read_user_prompt(workspace: Path) -> Optional[UserPromptView]:
    """Load PROMPT.md from the workspace root and compare its hash to the
    last one the agent acknowledged via `mark_user_prompt_read`. Returns
    None if the file is absent.

    The acknowledged hash lives at `.hyperherd/user-prompt.sha256` (a
    single hex digest). When it matches the current file hash, we return
    a status="unchanged" view with `text=None` so the agent sees that
    nothing has changed without re-paying for the file's tokens every
    tick.
    """
    import hashlib

    path = workspace / USER_PROMPT_FILE
    if not path.is_file():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None

    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    ack_path = _hyperherd_dir(workspace) / "user-prompt.sha256"
    last_sha: Optional[str] = None
    if ack_path.is_file():
        try:
            last_sha = ack_path.read_text().strip() or None
        except OSError:
            last_sha = None

    if last_sha is None:
        return UserPromptView(status="new", sha256=sha, text=text)
    if last_sha != sha:
        return UserPromptView(status="changed", sha256=sha, text=text)
    return UserPromptView(status="unchanged", sha256=sha, text=None)


def _drain_inbox(workspace: Path) -> List[InboundMessage]:
    """Atomically drain inbox.jsonl: rename it aside, read it, delete it.

    The rename is the critical bit. The previous implementation did
    `read_text()` then `write_text("")` non-atomically, so any append that
    landed between those two steps was silently truncated on disk —
    Discord messages just disappeared.

    With rename: writers that already had the file open before the
    rename keep writing into the renamed file (POSIX fds track inodes,
    not paths), so we still capture their data. Writers that open AFTER
    the rename create a fresh inbox.jsonl, which is preserved for the
    next tick. Either way, no message is lost.

    Lines that fail to parse are silently dropped — we'd rather lose
    one message than abort the tick.
    """
    path = _hyperherd_dir(workspace) / INBOX_FILE
    if not path.is_file():
        return []

    drain_path = _hyperherd_dir(workspace) / (INBOX_FILE + ".draining")
    try:
        os.replace(path, drain_path)
    except OSError:
        # Couldn't rename — another drain in flight, or the file
        # vanished. Skip; next tick will retry.
        return []

    try:
        raw = drain_path.read_text()
    except OSError:
        raw = ""
    finally:
        try:
            drain_path.unlink()
        except OSError:
            pass

    msgs: List[InboundMessage] = []
    bad = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            msgs.append(InboundMessage(
                timestamp=d["timestamp"],
                source=d.get("source", "unknown"),
                author=d.get("author", ""),
                text=d.get("text", ""),
            ))
        except (json.JSONDecodeError, KeyError):
            bad += 1
            continue

    # Diagnostic log so when messages go missing the daemon log shows
    # exactly what state.compute saw.
    import logging
    _log = logging.getLogger(__name__)
    if msgs or bad:
        _log.info(
            "Drained inbox: %d message(s)%s",
            len(msgs),
            f" ({bad} unparseable line(s) dropped)" if bad else "",
        )

    # Mirror each inbound message into the chat history buffer so the
    # agent has cross-tick context once the inbox is drained.
    if msgs:
        try:
            from hyperherd.monitor_agent.tools import record_chat_entry
            for m in msgs:
                record_chat_entry(
                    workspace,
                    role="user", text=m.text, via=m.source,
                    author=m.author, timestamp=m.timestamp,
                )
        except Exception:
            pass

    return msgs


def _read_chat_history(workspace: Path) -> List[ChatEntry]:
    """Read chat-history.jsonl. Returns user+agent entries in chronological
    order. The file is maintained by `tools.record_chat_entry` and capped
    to the last few entries; we just parse and return."""
    from hyperherd.monitor_agent.tools import CHAT_HISTORY_FILENAME

    path = _hyperherd_dir(workspace) / CHAT_HISTORY_FILENAME
    if not path.is_file():
        return []
    try:
        raw = path.read_text()
    except OSError:
        return []

    out: List[ChatEntry] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(ChatEntry(
                timestamp=d.get("timestamp", ""),
                role=d.get("role", "?"),
                author=d.get("author", ""),
                via=d.get("via", "?"),
                text=d.get("text", ""),
            ))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


# --- public entrypoint ------------------------------------------------------

def compute(workspace: Path, trigger: TickTrigger = "scheduled") -> TickState:
    """Build the per-tick state document. One subprocess call to `herd
    snapshot`, one snapshot rotation, plus filesystem reads."""
    workspace = Path(workspace).resolve()
    cur = _rotate_and_capture(workspace)
    prev = _read_prev(workspace)

    return TickState(
        sweep_name=cur.get("sweep_name", "unknown"),
        workspace=str(workspace),
        trigger=trigger,
        plan=_read_plan(workspace),
        totals=cur.get("totals", {}),
        trials=cur.get("trials", []),
        newly_failed=_diff_failed(prev, cur),
        newly_completed=_diff_completed(prev, cur),
        newly_pruned=_diff_pruned(prev, cur),
        inbox=_drain_inbox(workspace),
        chat_history=_read_chat_history(workspace),
        user_prompt=_read_user_prompt(workspace),
    )
