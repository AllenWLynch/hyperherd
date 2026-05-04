"""Parse and validate hyperherd YAML configuration files."""

import os
import re
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator


# Tokens used as filename components in `experiment_name` (and thus output
# directory paths). Restrict to alphanumerics, dot, underscore, hyphen — the
# intersection of "safe in a path" and "safe in a space-separated key=value
# override string" across launchers.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class DiscreteParameter(BaseModel):
    type: Literal["discrete"]
    abbrev: Optional[str] = None
    values: List[Any] = Field(min_length=1)
    labels: Optional[List[str]] = None
    default: Optional[Any] = None

    @model_validator(mode="after")
    def _validate_default(self):
        if self.default is not None and self.default not in self.values:
            raise ValueError(
                f"default '{self.default}' is not in values: {self.values}"
            )
        return self

    @model_validator(mode="after")
    def _validate_labels(self):
        if self.labels is None:
            # Slashes in values would produce ugly / unsafe experiment names.
            # Require labels so the user picks a clean display token.
            for v in self.values:
                if isinstance(v, str) and "/" in v:
                    raise ValueError(
                        f"value {v!r} contains '/'; provide a `labels:` list "
                        f"so experiment names use a short display token "
                        f"instead of the raw value"
                    )
            return self
        if len(self.labels) != len(self.values):
            raise ValueError(
                f"labels has {len(self.labels)} entries but values has "
                f"{len(self.values)}; they must match 1-to-1"
            )
        for label in self.labels:
            if not isinstance(label, str) or not label:
                raise ValueError(f"labels must be non-empty strings, got: {label!r}")
            if "/" in label:
                raise ValueError(f"labels may not contain '/', got: {label!r}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"labels must be unique, got: {self.labels}")
        return self

    def label_for(self, value: Any) -> Optional[str]:
        """Return the user-provided display label for a value, if labels are set."""
        if self.labels is None:
            return None
        for v, label in zip(self.values, self.labels):
            if v == value:
                return label
        return None


class ContinuousParameter(BaseModel):
    type: Literal["continuous"]
    abbrev: Optional[str] = None
    low: float
    high: float
    scale: Literal["linear", "log"] = "linear"
    steps: int = Field(default=5, ge=1)
    default: Optional[float] = None

    @model_validator(mode="after")
    def _validate_range(self):
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.scale == "log" and self.low <= 0:
            raise ValueError(f"log scale requires low > 0, got {self.low}")
        if self.default is not None:
            if self.default < self.low or self.default > self.high:
                raise ValueError(
                    f"default {self.default} is outside range [{self.low}, {self.high}]"
                )
        return self


ParameterSpec = Union[DiscreteParameter, ContinuousParameter]


# Valid operator keys for `when` operator-maps.
WHEN_OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le", "in", "not_in"}


def _coerce_numeric(value):
    """Recursively coerce strings that parse cleanly as int/float to numbers.

    Strings like "1e-3", "0.001", "42" become numbers; anything else
    (including bare True/False, real text, leading-zero IDs) is left alone.
    Applied to nested dicts and lists so it reaches operator-map values
    (`{le: "1e-3"}`) and exclude lists (`[1e-3]`).
    """
    if isinstance(value, str):
        s = value.strip()
        # Leave decorative/leading-zero strings (like '007', '0x10') as strings.
        if not s or s.lower() in {"true", "false", "null", "none"}:
            return value
        try:
            f = float(s)
        except (TypeError, ValueError):
            return value
        # Prefer int when the parsed float is integer and the source had no
        # decimal/exponent (so "1.0" stays float, "1" becomes int).
        if "." not in s and "e" not in s.lower() and float(s).is_integer():
            try:
                return int(s)
            except ValueError:
                return f
        return f
    if isinstance(value, list):
        return [_coerce_numeric(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_numeric(v) for k, v in value.items()}
    return value


class Constraint(BaseModel):
    name: str = "unnamed"
    when: Dict[str, Any] = Field(min_length=1)
    exclude: Optional[Dict[str, List[Any]]] = None
    force: Optional[Dict[str, Any]] = None
    set: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _need_action(self):
        if not self.exclude and not self.force and not self.set:
            raise ValueError(
                f"Constraint '{self.name}': must have at least one of "
                f"'exclude', 'force', or 'set'"
            )
        return self

    @model_validator(mode="after")
    def _validate_when_matchers(self):
        for param, matcher in self.when.items():
            if param == "expr":
                # Validated separately; needs the global parameter set.
                if not isinstance(matcher, str) or not matcher.strip():
                    raise ValueError(
                        f"Constraint '{self.name}': 'when.expr' must be a non-empty string"
                    )
                continue
            if isinstance(matcher, dict):
                if len(matcher) != 1:
                    raise ValueError(
                        f"Constraint '{self.name}': operator-map for '{param}' "
                        f"must have exactly one operator key, got {list(matcher.keys())}"
                    )
                op = next(iter(matcher.keys()))
                if op not in WHEN_OPERATORS:
                    raise ValueError(
                        f"Constraint '{self.name}': unknown operator '{op}' for "
                        f"'{param}'. Valid operators: {sorted(WHEN_OPERATORS)}"
                    )
                if op in ("in", "not_in") and not isinstance(matcher[op], list):
                    raise ValueError(
                        f"Constraint '{self.name}': operator '{op}' on '{param}' "
                        f"requires a list value"
                    )
        return self

    @model_validator(mode="after")
    def _validate_set_keys(self):
        if self.set:
            for k, v in self.set.items():
                if not isinstance(k, str) or not k:
                    raise ValueError(
                        f"Constraint '{self.name}': 'set' keys must be non-empty strings"
                    )
                if isinstance(v, dict) and set(v.keys()) == {"expr"}:
                    if not isinstance(v["expr"], str) or not v["expr"].strip():
                        raise ValueError(
                            f"Constraint '{self.name}': 'set.{k}.expr' must be "
                            f"a non-empty string"
                        )
        return self

    @model_validator(mode="before")
    @classmethod
    def _normalize_exclude(cls, data):
        if isinstance(data, dict) and "exclude" in data and data["exclude"]:
            for k, v in data["exclude"].items():
                if not isinstance(v, list):
                    data["exclude"][k] = [v]
        return data

    @model_validator(mode="before")
    @classmethod
    def _coerce_numeric_strings(cls, data):
        """Coerce strings that parse as numbers (e.g. YAML 1.1 reads `1e-3`
        as a string because there's no decimal/sign).

        Without this, a `when: {lr: {le: 1e-3}}` clause silently fails to
        match because `_match_one`'s comparison branch requires int/float on
        both sides. Same trap in `exclude` list values and any `force`/`set`
        literal that's intended to be numeric.
        """
        if not isinstance(data, dict):
            return data
        for field in ("when", "exclude", "force", "set"):
            section = data.get(field)
            if isinstance(section, dict):
                data[field] = {k: _coerce_numeric(v) for k, v in section.items()}
        return data


class SlurmConfig(BaseModel):
    # Required: there's no universal "default" partition — every cluster
    # has its own naming. Forcing a value here means `herd run --dry-run`
    # fails loudly on an un-edited template, which is what we want.
    partition: str
    time: str = "01:00:00"
    mem: str = "8G"
    cpus_per_task: int = 1
    gres: Optional[str] = None
    max_concurrent: Optional[int] = Field(default=None, ge=1)
    extra_args: List[str] = Field(default_factory=list)


class DiscordConfig(BaseModel):
    """Settings for the agent-SDK monitor's Discord channel.

    The token comes from the `DISCORD_BOT_TOKEN` environment variable —
    don't put secrets in YAML. If `guild_id` is set and the env var is
    present, the daemon connects on startup and uses Discord for both
    outbound posts and inbound user messages. Otherwise it ignores this
    section entirely.
    """

    guild_id: Optional[str] = None
    """Discord server (guild) ID. Required to enable the Discord channel."""

    channel_id: Optional[str] = None
    """Pin to a specific existing channel; skips auto-create."""

    channel_name: Optional[str] = None
    """Override the sweep-derived channel name. Discord lowercases it
    automatically; non [a-z0-9-] chars get stripped."""

    dashboard_refresh_seconds: int = 60
    """How often the daemon edits the live-dashboard message. Set to 0
    to disable the dashboard (you'll still have the periodic 🐾
    heartbeat posts and the slash commands)."""


class McpServerConfig(BaseModel):
    """One external MCP server, wired into the agent's tool surface.

    Mirrors the Claude Agent SDK's external-MCP shape — the entries are
    passed straight through to `ClaudeAgentOptions.mcp_servers` so the
    agent sees the server's tools as `mcp__<name>__<tool>`. Use this to
    plug in vendor MCP servers (wandb, mlflow, ClickUp, custom in-house
    ones) without HyperHerd having to wrap or abstract them.

    `env` values are looked up from the environment of the daemon
    process — no secrets in YAML.
    """

    name: str
    """Name the agent sees in tool prefixes (e.g. `wandb` →
    `mcp__wandb__*`)."""

    command: str
    """Executable to launch — `uvx`, `npx`, an absolute path, etc."""

    args: List[str] = Field(default_factory=list)
    """Argv tail passed to `command`."""

    env: Dict[str, str] = Field(default_factory=dict)
    """Environment variables to set in the subprocess. Values are
    typically env-var references like `${WANDB_API_KEY}` — the daemon
    expands these from its own environment at startup."""


class Config(BaseModel):
    name: str
    workspace: str = ""  # set by load_config from the config file's directory

    # Grid field: which parameters to grid over.
    #   - None (omitted): one-at-a-time from defaults
    #   - "all": Cartesian product of all parameters
    #   - list of param names: grid those, defaults for the rest
    grid: Optional[Union[Literal["all"], List[str]]] = None

    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    mcp_servers: List[McpServerConfig] = Field(default_factory=list)

    # Extra override tokens appended to every trial's argument string. The
    # format is whatever the launcher expects — for Hydra trainers this is
    # `key=value`; for a launcher that uses `parse_overrides()`, anything
    # parseable by it. The string is split on whitespace by the shell when
    # the sbatch script forwards it, so each entry should be one token.
    static_overrides: List[str] = Field(default_factory=list)
    launcher: str = ""

    parameters: Dict[str, ParameterSpec] = Field(min_length=1)
    conditions: List[Constraint] = Field(
        default_factory=list,
        validation_alias=AliasChoices("conditions", "constraints"),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_hydra_static_overrides(cls, data):
        """Lift `hydra.static_overrides` to top-level `static_overrides`.

        The `hydra:` section was the historical home for static overrides
        when the project assumed Hydra trainers exclusively. The field is
        launcher-agnostic now, so it lives at the top level — but we keep
        the old key working as a silent alias. If both are set, the
        top-level one wins (and we ignore the nested one).
        """
        if not isinstance(data, dict):
            return data
        hydra = data.get("hydra")
        if isinstance(hydra, dict) and "static_overrides" in hydra:
            data.setdefault("static_overrides", hydra["static_overrides"])
        # Drop the legacy section so pydantic doesn't reject it as an unknown field.
        data.pop("hydra", None)
        return data

    @model_validator(mode="after")
    def _validate_abbrev_safety(self):
        """Each parameter's experiment-name token must be safe in a file path.

        The token is the explicit `abbrev` if set, otherwise the parameter
        name. Either way it ends up in `experiment_name` (e.g. `lr-0.001`),
        which becomes a directory component for outputs/checkpoints. Reject
        anything outside [A-Za-z0-9._-] — `/` would corrupt paths, whitespace
        would corrupt the override string, `=` collides with the override
        syntax, etc.
        """
        for name, spec in self.parameters.items():
            if spec.abbrev is not None:
                if not _SAFE_TOKEN_RE.match(spec.abbrev):
                    raise ValueError(
                        f"parameter '{name}': abbrev {spec.abbrev!r} contains "
                        f"characters that would corrupt file paths or override "
                        f"syntax; allowed characters are letters, digits, "
                        f"'.', '_', '-'"
                    )
            else:
                if not _SAFE_TOKEN_RE.match(name):
                    raise ValueError(
                        f"parameter name {name!r} contains characters unsafe "
                        f"for filenames or override syntax (allowed: letters, "
                        f"digits, '.', '_', '-'). Set an explicit `abbrev:` "
                        f"on this parameter so the experiment name stays clean."
                    )
        return self

    @model_validator(mode="after")
    def _validate_param_name_chars(self):
        """Reject parameter names that would break override-string parsing.

        Even with an abbrev set, the parameter name itself flows into the
        override string as `name=value`, and the whole string is single-
        quoted into the generated sbatch case block. Quotes / equals signs /
        whitespace / shell metachars in names would corrupt either the
        override format or its shell quoting.

        Hydra-style `+foo`/`++foo`/`~foo` prefixes and dotted paths like
        `model.lr` are intentionally permitted — those are normal in Hydra.
        Filename-unsafe-but-otherwise-fine chars like `/` are caught earlier
        by `_validate_abbrev_safety` (which fires when no abbrev is set).
        """
        forbidden = set("'\"`=\\\n\r\t ")
        for name in self.parameters:
            bad = sorted(set(name) & forbidden)
            if bad:
                raise ValueError(
                    f"parameter name {name!r} contains forbidden characters "
                    f"{bad}; these would break override-string parsing or "
                    f"shell quoting"
                )
        return self

    @model_validator(mode="after")
    def _validate_grid_and_defaults(self):
        param_names = set(self.parameters.keys())

        if self.grid == "all":
            # No defaults needed
            pass
        elif isinstance(self.grid, list):
            # Validate grid param names exist
            unknown = set(self.grid) - param_names
            if unknown:
                raise ValueError(f"grid references unknown parameters: {sorted(unknown)}")
            # Defaults required for non-grid params
            non_grid = param_names - set(self.grid)
            for name in non_grid:
                if self.parameters[name].default is None:
                    raise ValueError(
                        f"parameter '{name}' needs a default value "
                        f"(it is not in the grid)"
                    )
        else:
            # grid is None -> one-at-a-time, all params need defaults
            for name, spec in self.parameters.items():
                if spec.default is None:
                    raise ValueError(
                        f"parameter '{name}' needs a default value "
                        f"(grid is not set, so all parameters need defaults)"
                    )

        # Expression namespace: param names with +/~ prefixes stripped.
        # Validate uniqueness here once for the whole config.
        from hyperherd.expr import (
            ExprError,
            sanitized_namespace,
            validate_expr,
            validate_namespace_keys,
        )
        try:
            validate_namespace_keys(param_names)
        except ExprError as e:
            raise ValueError(str(e)) from e
        expr_names = set(sanitized_namespace({n: None for n in param_names}).keys())

        # Validate condition references
        for constraint in self.conditions:
            for ref in constraint.when:
                if ref == "expr":
                    try:
                        validate_expr(constraint.when["expr"], expr_names)
                    except ExprError as e:
                        raise ValueError(
                            f"Constraint '{constraint.name}': {e}"
                        ) from e
                    continue
                if ref not in param_names:
                    raise ValueError(
                        f"Constraint '{constraint.name}': 'when' references "
                        f"unknown parameter '{ref}'"
                    )
            if constraint.exclude:
                for ref in constraint.exclude:
                    if ref not in param_names:
                        raise ValueError(
                            f"Constraint '{constraint.name}': 'exclude' references "
                            f"unknown parameter '{ref}'"
                        )
            if constraint.force:
                for ref in constraint.force:
                    if ref not in param_names:
                        raise ValueError(
                            f"Constraint '{constraint.name}': 'force' references "
                            f"unknown parameter '{ref}'"
                        )
            if constraint.set:
                for k, v in constraint.set.items():
                    if isinstance(v, dict) and set(v.keys()) == {"expr"}:
                        try:
                            validate_expr(v["expr"], expr_names)
                        except ExprError as e:
                            raise ValueError(
                                f"Constraint '{constraint.name}': set.{k}: {e}"
                            ) from e
            # Note: `set` keys are arbitrary Hydra paths and intentionally not
            # validated against `parameters` — that's the whole point.

        return self

    def get_param(self, name: str) -> ParameterSpec:
        return self.parameters[name]

    @property
    def param_names(self) -> List[str]:
        return list(self.parameters.keys())

    @property
    def abbrevs(self) -> Dict[str, str]:
        return {
            name: (spec.abbrev if spec.abbrev else name)
            for name, spec in self.parameters.items()
        }

    @property
    def labels(self) -> Dict[str, Dict[Any, str]]:
        """Per-parameter value→label mappings for experiment naming.

        Only includes discrete parameters that declared `labels:`. Parameters
        without explicit labels are absent (not an empty dict) so the name
        builder falls back to stringifying the raw value.
        """
        result: Dict[str, Dict[Any, str]] = {}
        for name, spec in self.parameters.items():
            if isinstance(spec, DiscreteParameter) and spec.labels is not None:
                result[name] = dict(zip(spec.values, spec.labels))
        return result

    @property
    def defaults(self) -> Optional[Dict[str, Any]]:
        """Build defaults dict from per-parameter default fields.

        Returns None if no parameters have defaults set.
        """
        d = {}
        for name, spec in self.parameters.items():
            if spec.default is not None:
                d[name] = spec.default
        return d if d else None


class ConfigError(Exception):
    pass


CONFIG_FILENAME = "hyperherd.yaml"


def load_config(path: str) -> Config:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, CONFIG_FILENAME)
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must be a YAML mapping")

    # Workspace is the directory containing the config file
    config_dir = os.path.dirname(path)
    raw["workspace"] = config_dir

    # Resolve launcher path relative to config dir
    launcher = raw.get("launcher", "")
    if launcher and not os.path.isabs(launcher):
        raw["launcher"] = os.path.normpath(os.path.join(config_dir, launcher))

    try:
        return Config.model_validate(raw)
    except Exception as e:
        raise ConfigError(str(e)) from e
