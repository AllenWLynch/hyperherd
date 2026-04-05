"""Parse and validate hyperwhip YAML configuration files."""

import os
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class DiscreteParameter(BaseModel):
    type: Literal["discrete"]
    abbrev: str
    values: List[Any] = Field(min_length=1)


class ContinuousParameter(BaseModel):
    type: Literal["continuous"]
    abbrev: str
    low: float
    high: float
    scale: Literal["linear", "log"] = "linear"
    steps: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _validate_range(self):
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.scale == "log" and self.low <= 0:
            raise ValueError(f"log scale requires low > 0, got {self.low}")
        return self


ParameterSpec = Union[DiscreteParameter, ContinuousParameter]


class Constraint(BaseModel):
    name: str = "unnamed"
    when: Dict[str, Any] = Field(min_length=1)
    exclude: Optional[Dict[str, List[Any]]] = None
    force: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _need_exclude_or_force(self):
        if not self.exclude and not self.force:
            raise ValueError(f"Constraint '{self.name}': must have 'exclude' and/or 'force'")
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

    # Default values for parameters. Required when grid != "all".
    defaults: Optional[Dict[str, Any]] = None

    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    hydra: HydraConfig = Field(default_factory=HydraConfig)
    launcher: str = ""

    # parameters stored as a dict of name -> spec during parsing,
    # but we expose them as a list with the name embedded
    parameters: Dict[str, ParameterSpec] = Field(min_length=1)
    constraints: List[Constraint] = Field(default_factory=list)

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
            if non_grid:
                if not self.defaults:
                    raise ValueError(
                        f"defaults required for non-grid parameters: {sorted(non_grid)}"
                    )
                missing = non_grid - set(self.defaults.keys())
                if missing:
                    raise ValueError(f"defaults missing for parameters: {sorted(missing)}")
        else:
            # grid is None -> one-at-a-time, all params need defaults
            if not self.defaults:
                raise ValueError("defaults required when grid is not set (one-at-a-time mode)")
            missing = param_names - set(self.defaults.keys())
            if missing:
                raise ValueError(f"defaults missing for parameters: {sorted(missing)}")

        # Validate constraint references
        for constraint in self.constraints:
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

        return self

    def get_param(self, name: str) -> ParameterSpec:
        return self.parameters[name]

    @property
    def param_names(self) -> List[str]:
        return list(self.parameters.keys())

    @property
    def abbrevs(self) -> Dict[str, str]:
        return {name: spec.abbrev for name, spec in self.parameters.items()}


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
