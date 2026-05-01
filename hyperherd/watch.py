"""Polling daemon that posts trial state changes to a webhook.

Reads its settings from the `watch:` block in hyperherd.yaml. Stays stdlib-only
on purpose: webhook delivery is `urllib.request`, the optional Claude summary
shells out to the `claude` CLI via `subprocess`. No new package deps.

State (which trials we've already announced, when we last sent a heartbeat,
whether we've already fired `done` for this sweep) lives in
`<workspace>/.hyperherd/watch.json`.
"""

import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hyperherd import manifest, slurm
from hyperherd.config import Config


WATCH_STATE_FILE = "watch.json"

# Statuses we treat as "the trial has reached an end-state".
TERMINAL_STATUSES = ("completed", "failed", "cancelled")

# A heartbeat fires only if the totals snapshot has changed since the last
# heartbeat — keeps the daemon from spamming an unchanging sweep.
_HEARTBEAT_TOTAL_KEYS = ("ready", "submitted", "queued", "running",
                         "completed", "failed", "cancelled")


class WatchError(Exception):
    pass


# --- State file ---------------------------------------------------------------

def _state_path(workspace: str) -> str:
    return os.path.join(manifest.workspace_path(workspace), WATCH_STATE_FILE)


def load_state(workspace: str) -> Dict[str, Any]:
    path = _state_path(workspace)
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_state(workspace: str, state: Dict[str, Any]) -> None:
    path = _state_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# --- Event detection ----------------------------------------------------------

def _totals(trials: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in trials:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    counts["total"] = len(trials)
    return counts


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _all_terminal(trials: List[dict]) -> bool:
    return bool(trials) and all(t["status"] in TERMINAL_STATUSES for t in trials)


def detect_events(
    trials: List[dict],
    state: Dict[str, Any],
    enabled: List[str],
    heartbeat_seconds: Optional[int],
    now: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pure transition detector: takes the current manifest + last state and
    returns (events_to_emit, updated_state). No I/O, no webhook calls — that
    way the test suite can drive it directly without touching a network.

    Rules:
      - `failed`: per-trial, fires when a trial moves into 'failed' or
        'cancelled'. Skipped on the first ever tick (last_seen empty) so a
        daemon started after the fact doesn't replay history.
      - `done`: fires once when every trial is in a terminal status. Resets
        if any trial later moves back to live (e.g. a resubmit).
      - `heartbeat`: fires at most every `heartbeat_seconds`, only if totals
        differ from the snapshot we sent at the last heartbeat.
    """
    events: List[Dict[str, Any]] = []
    last_seen: Dict[str, str] = state.get("last_seen", {}) or {}
    first_tick = not last_seen

    current = {str(t["index"]): t["status"] for t in trials}
    by_index = {t["index"]: t for t in trials}
    totals = _totals(trials)

    # Per-trial 'failed' transitions.
    if "failed" in enabled and not first_tick:
        for idx_s, status in current.items():
            if status not in ("failed", "cancelled"):
                continue
            if last_seen.get(idx_s) == status:
                continue
            trial = by_index[int(idx_s)]
            events.append({
                "event": "trial_failed",
                "trial": _trial_payload(trial),
                "totals": totals,
            })

    # Sweep-level 'done'.
    done_emitted = bool(state.get("done_emitted", False))
    any_live_now = any(s not in TERMINAL_STATUSES for s in current.values())
    if any_live_now:
        done_emitted = False
    elif (
        "done" in enabled
        and not done_emitted
        and _all_terminal(trials)
        and not first_tick
    ):
        events.append({
            "event": "sweep_done",
            "trial": None,
            "totals": totals,
        })
        done_emitted = True

    # Heartbeat digest.
    last_heartbeat = float(state.get("last_heartbeat_ts", 0.0))
    last_hb_totals = state.get("last_heartbeat_totals")
    new_hb_totals: Optional[Dict[str, int]] = None
    if (
        "heartbeat" in enabled
        and heartbeat_seconds is not None
        and now - last_heartbeat >= heartbeat_seconds
    ):
        snapshot = {k: totals.get(k, 0) for k in _HEARTBEAT_TOTAL_KEYS}
        snapshot["total"] = totals.get("total", 0)
        if snapshot != last_hb_totals:
            events.append({
                "event": "heartbeat",
                "trial": None,
                "totals": totals,
            })
            new_hb_totals = snapshot

    # Carry forward any keys we don't manage (e.g. `default_topic`) so the
    # zero-config ntfy fallback's persisted topic survives daemon restarts.
    new_state = dict(state)
    new_state["last_seen"] = current
    new_state["done_emitted"] = done_emitted
    new_state["last_heartbeat_ts"] = (
        now if new_hb_totals is not None else last_heartbeat
    )
    new_state["last_heartbeat_totals"] = (
        new_hb_totals if new_hb_totals is not None else last_hb_totals
    )
    return events, new_state


def _trial_payload(trial: dict) -> Dict[str, Any]:
    return {
        "index": trial["index"],
        "experiment_name": trial.get("experiment_name", ""),
        "status": trial.get("status", "unknown"),
        "params": trial.get("params", {}),
    }


# --- Rendering ----------------------------------------------------------------

def render_line(event: Dict[str, Any], sweep_name: str) -> str:
    """One-line human summary used as the body for slack/discord/ntfy."""
    totals = event["totals"]
    parts = [f"{totals.get('completed', 0)}/{totals.get('total', 0)} done"]
    for key in ("running", "failed", "queued", "cancelled"):
        n = totals.get(key, 0)
        if n:
            parts.append(f"{n} {key}")
    summary_tail = ", ".join(parts)

    kind = event["event"]
    prefix = f"[{sweep_name}] " if sweep_name else ""
    if kind == "trial_failed":
        t = event["trial"]
        name = t.get("experiment_name") or f"trial {t['index']}"
        cause = (event.get("failure") or {}).get("cause")
        suffix = f" ({cause})" if cause else ""
        return (
            f"{prefix}trial {t['index']} ({name}) {t['status']}{suffix} — "
            f"{summary_tail}"
        )
    if kind == "sweep_done":
        return f"{prefix}sweep complete — {summary_tail}"
    if kind == "heartbeat":
        return f"{prefix}heartbeat — {summary_tail}"
    return f"{prefix}{kind} — {summary_tail}"


def _failure_text_block(event: Dict[str, Any]) -> str:
    """Extra body text for trial_failed events in slack/discord/ntfy.

    When `summarize: true` and the Claude call succeeded, the 1-2 sentence
    diagnosis goes in. Otherwise we attach the stderr tail in a triple-fenced
    block (or omit it entirely if the trial produced no stderr). Returns ''
    for non-failure events so callers can unconditionally append it.
    """
    if event.get("event") != "trial_failed":
        return ""
    summary = event.get("summary")
    if summary:
        return f"\n{summary}"
    failure = event.get("failure") or {}
    tail = failure.get("stderr_tail", "")
    if not tail:
        return ""
    truncated = failure.get("stderr_truncated")
    marker = "…(truncated, see logs)\n" if truncated else ""
    return f"\n```\n{marker}{tail}\n```"


# --- Webhook delivery ---------------------------------------------------------

def build_request(
    webhook: str,
    fmt: str,
    event: Dict[str, Any],
    sweep_name: str,
) -> urllib.request.Request:
    """Build a urllib Request for a single event. Pure — no network I/O."""
    line = render_line(event, sweep_name)
    extra = _failure_text_block(event)
    text_body = line + extra
    if fmt == "slack":
        body = json.dumps({"text": text_body}).encode()
        ctype = "application/json"
    elif fmt == "discord":
        body = json.dumps({"content": text_body}).encode()
        ctype = "application/json"
    elif fmt == "ntfy":
        body = text_body.encode()
        ctype = "text/plain"
    elif fmt == "raw":
        payload = dict(event)
        payload["sweep"] = sweep_name
        payload["timestamp"] = _now_iso()
        body = json.dumps(payload).encode()
        ctype = "application/json"
    else:
        raise WatchError(f"unknown format: {fmt!r}")

    return urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )


def build_message_request(
    webhook: str,
    fmt: str,
    text: str,
    sweep_name: str,
) -> urllib.request.Request:
    """Build a urllib Request for a free-form text message (`herd msg`).

    Mirrors `build_request`'s format dispatch so a manual message lands in the
    same channel and renders the same way as the daemon's events.
    """
    prefix = f"[{sweep_name}] " if sweep_name else ""
    body_text = prefix + text
    if fmt == "slack":
        body = json.dumps({"text": body_text}).encode()
        ctype = "application/json"
    elif fmt == "discord":
        body = json.dumps({"content": body_text}).encode()
        ctype = "application/json"
    elif fmt == "ntfy":
        body = body_text.encode()
        ctype = "text/plain"
    elif fmt == "raw":
        payload = {
            "event": "message",
            "sweep": sweep_name,
            "text": text,
            "timestamp": _now_iso(),
        }
        body = json.dumps(payload).encode()
        ctype = "application/json"
    else:
        raise WatchError(f"unknown format: {fmt!r}")

    return urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )


def post_message(
    webhook: str,
    fmt: str,
    text: str,
    sweep_name: str,
    timeout: float = 10.0,
) -> None:
    """POST a free-text message. Unlike `post_event`, delivery errors propagate
    so the interactive `herd msg` caller can show them."""
    req = build_message_request(webhook, fmt, text, sweep_name)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def post_event(
    webhook: str,
    fmt: str,
    event: Dict[str, Any],
    sweep_name: str,
    timeout: float = 10.0,
) -> None:
    """POST a single event. Logs and swallows failures so a flaky webhook can't
    take down the daemon — we'd rather miss one notification than lose the
    whole watcher mid-sweep.
    """
    req = build_request(webhook, fmt, event, sweep_name)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(
            f"watch: webhook POST failed ({event['event']}): {e}",
            file=sys.stderr,
        )


# --- Failure diagnosis --------------------------------------------------------

# Caps for the stderr tail attached to failure notifications. Slack/ntfy can
# accept much more, but Discord limits message bodies to 2000 chars total —
# ~1500 bytes of stderr leaves room for the line itself plus the code-fence
# wrapper. Cap on lines too so an ML training log full of long lines doesn't
# eat the whole budget.
_STDERR_TAIL_LINES = 20
_STDERR_TAIL_BYTES = 1500


def _read_stderr_tail(workspace: str, trial_index: int) -> tuple[str, bool]:
    """Return (text, truncated) for the trailing stderr of a trial.

    `truncated` is True when either the line cap or the byte cap kicked in,
    so the caller can render an ellipsis marker.
    """
    log_file = os.path.join(manifest.logs_path(workspace), f"{trial_index}.err")
    if not os.path.isfile(log_file):
        return "", False
    try:
        with open(log_file, "r", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        return "", False
    if not all_lines:
        return "", False

    truncated = len(all_lines) > _STDERR_TAIL_LINES
    tail_lines = all_lines[-_STDERR_TAIL_LINES:]
    text = "".join(tail_lines).rstrip()

    if len(text.encode()) > _STDERR_TAIL_BYTES:
        truncated = True
        # Trim from the front so the most recent (and most diagnostic) lines
        # survive. Cut on a line boundary if possible.
        encoded = text.encode()
        cut = encoded[-_STDERR_TAIL_BYTES:]
        try:
            text = cut.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            text = cut.decode("utf-8", errors="ignore")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    return text, truncated


def _human_cause(info: slurm.FailureInfo) -> str:
    """Render a SLURM FailureInfo as a short cause string for the summary line.

    Examples: 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'SIGSEGV',
    'exit code 1', 'CANCELLED'. Falls back to the raw state when nothing
    more specific is available.
    """
    state = info.state or "UNKNOWN"
    if state in ("TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"):
        return state
    if info.signal:
        # Common signals worth naming; everything else shows the number.
        named = {6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}
        return named.get(info.signal, f"signal {info.signal}")
    if info.exit_code is not None and info.exit_code != 0:
        return f"exit code {info.exit_code}"
    if state == "CANCELLED":
        return "CANCELLED"
    if state == "FAILED":
        return "FAILED"
    return state


def _diagnose_failure(workspace: str, trial: dict) -> Dict[str, Any]:
    """Build the `failure` block attached to a trial_failed event payload.

    Pulls the SLURM job_id from the manifest's job_ids ledger, queries sacct
    for state/exit/signal/reason, and reads the stderr tail. The resulting
    dict is what slack/discord/raw renderers consume. Always returns a dict
    — missing fields just become empty strings / None.
    """
    job_id = _latest_job_id_for_index(workspace, trial["index"])
    info = (
        slurm.query_failure_info(job_id, trial["index"])
        if job_id is not None
        else slurm.FailureInfo()
    )
    tail, truncated = _read_stderr_tail(workspace, trial["index"])
    return {
        "cause": _human_cause(info),
        "slurm_state": info.state,
        "exit_code": info.exit_code,
        "signal": info.signal,
        "reason": info.reason,
        "job_id": job_id,
        "stderr_tail": tail,
        "stderr_truncated": truncated,
    }


def _latest_job_id_for_index(workspace: str, index: int) -> Optional[str]:
    """Most recent SLURM job_id that included this trial's index, or None.

    Mirrors `cli._latest_job_id_for` but reads the ledger directly so this
    module doesn't have to import from cli."""
    for record in reversed(manifest.get_job_ids(workspace)):
        if index in record.get("indices", []):
            return record["slurm_job_id"]
    return None


def _summarize_failure_with_claude(
    sweep_name: str,
    trial: dict,
    failure: Dict[str, Any],
    timeout: float = 60.0,
) -> Optional[str]:
    """Shell out to `claude -p` for a 1-2 sentence diagnosis of a failure.

    Returns None on any error (claude missing, non-zero exit, timeout); the
    daemon falls back to the raw stderr tail in that case. Never raises.
    """
    payload = {
        "sweep": sweep_name,
        "trial": {
            "index": trial.get("index"),
            "experiment_name": trial.get("experiment_name", ""),
            "params": trial.get("params", {}),
        },
        "failure": {
            "cause": failure.get("cause"),
            "slurm_state": failure.get("slurm_state"),
            "exit_code": failure.get("exit_code"),
            "signal": failure.get("signal"),
            "reason": failure.get("reason"),
        },
        "stderr_tail": failure.get("stderr_tail", ""),
    }
    prompt = (
        "You are diagnosing a failed hyperparameter trial for a busy ML "
        "researcher. Below is the trial's metadata, SLURM exit info, and the "
        "tail of its stderr. Reply with 1-2 short sentences identifying the "
        "most likely root cause and the fix to try next. No preamble, no "
        "bullets, no code fences.\n\n"
        + json.dumps(payload)
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"watch: claude failure summary skipped: {e}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(
            f"watch: claude exited {proc.returncode}: {proc.stderr.strip()[:200]}",
            file=sys.stderr,
        )
        return None
    return proc.stdout.strip() or None


# --- Zero-config ntfy fallback ------------------------------------------------

_NTFY_BASE = "https://ntfy.sh"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Lowercase, hyphen-only slug. Used as a human hint in the random topic;
    not relied on for uniqueness — that comes from the random suffix."""
    s = _SLUG_RE.sub("-", (text or "sweep").lower()).strip("-")
    return s or "sweep"


def resolve_default_webhook(workspace: str, sweep_name: str) -> tuple[str, str]:
    """Return (webhook_url, topic) for the per-workspace ntfy fallback.

    On first call, generates a random topic and persists it to watch.json so
    the URL stays the same across daemon restarts (otherwise the user would
    re-subscribe every session). Topic format: `herd-{slug}-{random}` where
    the random suffix carries ~48 bits of entropy — enough that a public
    ntfy.sh broker can't be enumerated by guessing.
    """
    state = load_state(workspace)
    topic = state.get("default_topic")
    if not topic:
        topic = f"herd-{_slug(sweep_name)}-{secrets.token_urlsafe(8)}"
        state["default_topic"] = topic
        save_state(workspace, state)
    return f"{_NTFY_BASE}/{topic}", topic


def _print_default_webhook_banner(url: str) -> None:
    print(
        "watch.webhook is unset — falling back to a per-workspace ntfy.sh topic.\n"
        f"\n    {url}\n\n"
        "Subscribe by either:\n"
        f"  • Open {url} in a browser\n"
        "  • iOS/Android: install the ntfy app and add the topic\n"
        f"  • curl -s {url}/json   # streams events to the terminal\n\n"
        "Anyone with this URL can read the notifications, so treat it as a\n"
        "secret. To use a private webhook (Slack, Discord, your own ntfy\n"
        "server, ...) set `watch.webhook` in hyperherd.yaml.\n",
        file=sys.stderr,
    )


# --- Daemon loop --------------------------------------------------------------

def _refresh_status(workspace: str) -> None:
    """Pull fresh SLURM state into the manifest. Imported lazily to dodge a
    cli ↔ watch import cycle."""
    from hyperherd.cli import _sync_slurm_status
    _sync_slurm_status(workspace)


def tick(config: Config, now: Optional[float] = None) -> List[Dict[str, Any]]:
    """One poll: refresh SLURM, detect events, post them, persist state.

    Returns the list of events that were emitted on this tick.
    """
    if now is None:
        now = time.time()

    watch_cfg = config.watch
    if not watch_cfg.webhook:
        # Defensive — `run()` resolves the fallback before calling tick(); this
        # only fires if a caller invokes tick() directly without going through
        # run(). Don't silently generate a topic here, since tick() shouldn't
        # have side effects beyond reading/writing the state file.
        raise WatchError(
            "watch.webhook is not set; call watch.run() to apply the ntfy "
            "fallback, or configure watch.webhook in hyperherd.yaml"
        )

    _refresh_status(config.workspace)
    trials = manifest.load_manifest(config.workspace)
    state = load_state(config.workspace)

    heartbeat_seconds = (
        watch_cfg.heartbeat_minutes * 60
        if watch_cfg.heartbeat_minutes is not None
        else None
    )

    events, new_state = detect_events(
        trials=trials,
        state=state,
        enabled=list(watch_cfg.events),
        heartbeat_seconds=heartbeat_seconds,
        now=now,
    )

    for event in events:
        # Failure events get a diagnostic block (SLURM cause + stderr tail);
        # successes and heartbeats don't, since the user said Claude/diag adds
        # no value there.
        event["summary"] = None
        if event["event"] == "trial_failed":
            event["failure"] = _diagnose_failure(config.workspace, event["trial"])
            if watch_cfg.summarize:
                event["summary"] = _summarize_failure_with_claude(
                    config.name, event["trial"], event["failure"]
                )
        # Timestamped stdout line so a foreground / nohup-redirected daemon
        # leaves an audit trail of what fired and when, independent of the
        # webhook delivery itself.
        print(
            f"[{_now_iso()}] {event['event']} — "
            f"{render_line(event, config.name)}",
            flush=True,
        )
        post_event(watch_cfg.webhook, watch_cfg.format, event, config.name)

    save_state(config.workspace, new_state)
    return events


def run(config: Config, once: bool = False, pidfile: Optional[str] = None) -> None:
    if not config.watch.webhook:
        url, _ = resolve_default_webhook(config.workspace, config.name)
        config.watch.webhook = url
        config.watch.format = "ntfy"
        _print_default_webhook_banner(url)
    else:
        print(
            f"[{_now_iso()}] watch: posting to {config.watch.webhook} "
            f"(format={config.watch.format}, interval={config.watch.interval_seconds}s)",
            flush=True,
        )

    if pidfile:
        with open(pidfile, "w") as f:
            f.write(f"{os.getpid()}\n")

    try:
        while True:
            try:
                tick(config)
            except Exception as e:
                # Never let a transient parse / FS error kill the daemon.
                print(f"watch: tick failed: {e}", file=sys.stderr)
            if once:
                return
            time.sleep(config.watch.interval_seconds)
    finally:
        if pidfile and os.path.isfile(pidfile):
            try:
                os.unlink(pidfile)
            except OSError:
                pass
