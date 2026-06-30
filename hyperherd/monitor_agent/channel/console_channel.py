"""A `MessageChannel` that prints to the terminal.

Used by `herd monitor -p "<message>"`: a one-shot, no-Discord path where the
agent's replies should land on stdout instead of a chat surface. Inbound is a
no-op (the single message is seeded into the inbox by the CLI before the tick),
and the various post* methods just print.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional


class ConsoleChannel:
    """Outbound-only channel that echoes the agent's messages to stdout."""

    name = "console"

    async def start(self) -> None:  # pragma: no cover - trivial
        pass

    async def stop(self) -> None:  # pragma: no cover - trivial
        pass

    async def post(self, body: str) -> None:
        print(f"\n🐎 {body}\n")

    async def post_file(self, path, *, body: Optional[str] = None) -> None:
        if body:
            print(f"\n🐎 {body}")
        paths = path if isinstance(path, list) else [path]
        for p in paths:
            print(f"   [file] {p}")
        print()

    async def post_to_trial_thread(
        self,
        trial_index: int,
        body: Optional[str] = None,
        *,
        file_path: "Optional[Path]" = None,
        thread_seed_text: Optional[str] = None,
    ) -> None:
        prefix = f"\n🐎 [trial {trial_index}]"
        if body:
            print(f"{prefix} {body}")
        else:
            print(prefix)
        if file_path is not None:
            print(f"   [file] {file_path}")
        print()

    def set_inbound_handler(self, handler) -> None:  # pragma: no cover - no-op
        pass

    def set_stop_handler(self, handler) -> None:  # pragma: no cover - no-op
        pass

    def set_info_handler(self, handler) -> None:  # pragma: no cover - no-op
        pass

    def thinking(self):
        @contextlib.asynccontextmanager
        async def _cm():
            yield
        return _cm()
