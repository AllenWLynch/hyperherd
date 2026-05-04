# Logger MCP integrations

The autonomous monitor's `compute_metric` tool reads `log_result(...)` streams written to disk by your trial code. That covers most "is this run going off the rails" questions cheaply, with no extra setup.

For richer queries — "compare these three runs," "what does the val_loss curve look like across the sweep" — you can wire a vendor MCP server (wandb, mlflow, ClickUp, your own) directly into the agent's tool surface. HyperHerd doesn't wrap or abstract these; the agent just gets the vendor's tools as `mcp__<name>__<tool>` and learns to use them.

## When to add an MCP

You don't need one if:

- Your trial code calls `hyperherd.log_result(name, value, step=...)` for metrics you care about. `compute_metric(idx, name)` aggregates these directly. Cheap and deterministic.

You probably want one if:

- You log to wandb / mlflow / etc. and don't want to also write `log_result` calls in your trainer. The MCP gives the agent direct read access to runs.
- You want the agent to do open-ended exploration ("which trials have val_loss curves that look similar to idx 7?") that `compute_metric` can't express.

## How to add one

Two pieces: a new `mcp_servers` block in your `hyperherd.yaml`, and the relevant secret in the daemon's environment.

```yaml
# hyperherd.yaml
mcp_servers:
  - name: wandb
    command: uvx
    args:
      - --from
      - git+https://github.com/wandb/wandb-mcp-server
      - wandb-mcp-server
    env:
      WANDB_API_KEY: ${WANDB_API_KEY}
```

```bash
# In the shell that launches the daemon:
export WANDB_API_KEY='your-key-here'
herd monitor my_workspace
```

Restart the daemon if it was already running. The agent's tool list now includes `mcp__wandb__get_run_history_tool`, `mcp__wandb__compare_runs_tool`, etc.

`${VAR}` references in the `env:` block are expanded from the daemon process's environment at startup. No secrets in YAML.

## Adding multiple MCPs

```yaml
mcp_servers:
  - name: wandb
    command: uvx
    args: [--from, git+https://github.com/wandb/wandb-mcp-server, wandb-mcp-server]
    env:
      WANDB_API_KEY: ${WANDB_API_KEY}

  - name: my-internal
    command: /usr/local/bin/internal-mcp
    args: [--read-only]
    env:
      DB_URL: ${DB_URL}
```

Each `name` becomes the agent's prefix (`mcp__wandb__*`, `mcp__my-internal__*`). The agent can use any of them; the skill teaches conservative defaults — prefer `compute_metric` for routine aggregates, reach for the MCP only for queries it can't answer.

## What the agent does with them

Two behavioral guardrails in the skill:

1. **`compute_metric` first.** It's in-process, deterministic, returns small typed dicts. The MCP is a fallback for things `compute_metric` can't express.
2. **Trust local streams over remote ones.** If `compute_metric` says a run looks fine but the wandb MCP says it diverged, the agent flags the discrepancy in `msg` rather than acting on the remote.

The agent doesn't know about the MCP at boot — it discovers the tools when the daemon starts. If you add an MCP and want the agent to actually use it for a specific question, mention it in chat: "@HerdDog can you fetch the val_loss trajectory for idx 3 from wandb and tell me if it looks healthy?"

## Troubleshooting

**Tools don't appear in `/info` or `/help`.** Slash commands are HyperHerd-side; they don't show MCP tools. Look for `mcp__<name>__*` log lines when the daemon starts up — the SDK prints them on tool registration.

**"Command not found"** when the daemon launches. The MCP's `command:` (e.g. `uvx`, `npx`) needs to be on the daemon's `PATH`. Check with `which uvx` from the same shell that runs `herd monitor`.

**MCP starts but returns errors on every call.** Usually a credential issue. Check the daemon log for the MCP subprocess's stderr — most servers print their auth error there. Then verify the env var was set in the launching shell (`echo $WANDB_API_KEY` should print non-empty).

**Cost per tick spikes after adding the MCP.** Vendor MCPs vary widely in tool-result size. Check the `compute_metric` skill guidance — if the agent is fetching raw history through the MCP when a simple aggregate would do, tighten the skill or remove the MCP for that workspace.
