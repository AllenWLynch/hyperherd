"""Parse and validate hyperwhip YAML configuration files."""

import os
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator


class DiscreteParameter(BaseModel):
    type: Literal["discrete"]
    abbrev: str
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
    abbrev: str
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
            for k in self.set:
                if not isinstance(k, str) or not k:
                    raise ValueError(
                        f"Constraint '{self.name}': 'set' keys must be non-empty strings"
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


class SlurmConfig(BaseModel):
    partition: str = "default"
    time: str = "01:00:00"
    mem: str = "8G"
    cpus_per_task: int = 1
    gres: Optional[str] = None
    max_concurrent: Optional[int] = Field(default=None, ge=1)
    extra_args: List[str] = Field(default_factory=list)


class HydraConfig(BaseModel):
    static_overrides: List[str] = Field(default_factory=list)


class Config(BaseModel):
    name: str
    workspace: str = ""  # set by load_config from the config file's directory

    # Grid field: which parameters to grid over.
    #   - None (omitted): one-at-a-time from defaults
    #   - "all": Cartesian product of all parameters
    #   - list of param names: grid those, defaults for the rest
    grid: Optional[Union[Literal["all"], List[str]]] = None

    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    hydra: HydraConfig = Field(default_factory=HydraConfig)
    launcher: str = ""

    parameters: Dict[str, ParameterSpec] = Field(min_length=1)
    conditions: List[Constraint] = Field(
        default_factory=list,
        validation_alias=AliasChoices("conditions", "constraints"),
    )

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

        # Validate condition references
        for constraint in self.conditions:
            for ref in constraint.when:
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
        return {name: spec.abbrev for name, spec in self.parameters.items()}

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


CONFIG_FILENAME = "hyperwhip.yaml"


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
