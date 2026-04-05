"""Parse and validate hyperwhip YAML configuration files."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ParameterSpec:
    name: str
    type: str  # "discrete" or "continuous"
    values: Optional[List[Any]] = None  # for discrete
    low: Optional[float] = None  # for continuous
    high: Optional[float] = None
    scale: str = "linear"  # "linear" or "log"
    steps: int = 5  # discretization steps for continuous


@dataclass
class Constraint:
    name: str
    when: Dict[str, Any]  # param_name -> value that triggers this constraint
    exclude: Optional[Dict[str, List[Any]]] = None  # param_name -> values to exclude
    force: Optional[Dict[str, Any]] = None  # param_name -> forced value


@dataclass
class SlurmConfig:
    partition: str = "default"
    time: str = "01:00:00"
    mem: str = "8G"
    cpus_per_task: int = 1
    gres: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)


@dataclass
class HydraConfig:
    command: str = "python train.py"
    config_path: Optional[str] = None
    static_overrides: List[str] = field(default_factory=list)


@dataclass
class SearchConfig:
    mode: str = "grid"  # "grid" or "axes"
    defaults: Optional[Dict[str, Any]] = None  # required for axes mode


@dataclass
class Config:
    name: str
    workspace: str
    search: SearchConfig
    slurm: SlurmConfig
    hydra: HydraConfig
    launcher: str
    parameters: List[ParameterSpec]
    constraints: List[Constraint]


class ConfigError(Exception):
    pass


def _parse_parameter(name: str, spec: dict) -> ParameterSpec:
    ptype = spec.get("type")
    if ptype not in ("discrete", "continuous"):
        raise ConfigError(
            f"Parameter '{name}': type must be 'discrete' or 'continuous', got '{ptype}'"
        )
    if ptype == "discrete":
        values = spec.get("values")
        if not values or not isinstance(values, list):
            raise ConfigError(f"Parameter '{name}': discrete type requires a 'values' list")
        return ParameterSpec(name=name, type="discrete", values=values)
    else:
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            raise ConfigError(f"Parameter '{name}': continuous type requires 'low' and 'high'")
        return ParameterSpec(
            name=name,
            type="continuous",
            low=float(low),
            high=float(high),
            scale=spec.get("scale", "linear"),
            steps=spec.get("steps", 5),
        )


def _parse_constraint(raw: dict) -> Constraint:
    name = raw.get("name", "unnamed")
    when = raw.get("when")
    if not when or not isinstance(when, dict):
        raise ConfigError(f"Constraint '{name}': 'when' must be a non-empty dict")
    exclude = raw.get("exclude")
    force = raw.get("force")
    if not exclude and not force:
        raise ConfigError(f"Constraint '{name}': must have 'exclude' and/or 'force'")
    if exclude:
        for k, v in exclude.items():
            if not isinstance(v, list):
                exclude[k] = [v]
    return Constraint(name=name, when=when, exclude=exclude, force=force)


def load_config(path: str) -> Config:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must be a YAML mapping")

    name = raw.get("name")
    if not name:
        raise ConfigError("Config must have a 'name' field")

    workspace = raw.get("workspace", f"./{name}")
    config_dir = os.path.dirname(path)
    if not os.path.isabs(workspace):
        workspace = os.path.normpath(os.path.join(config_dir, workspace))

    # Search config
    search_raw = raw.get("search", {})
    search = SearchConfig(
        mode=search_raw.get("mode", "grid"),
        defaults=search_raw.get("defaults"),
    )
    if search.mode not in ("grid", "axes"):
        raise ConfigError(f"search.mode must be 'grid' or 'axes', got '{search.mode}'")

    # SLURM config
    slurm_raw = raw.get("slurm", {})
    slurm = SlurmConfig(
        partition=slurm_raw.get("partition", "default"),
        time=slurm_raw.get("time", "01:00:00"),
        mem=slurm_raw.get("mem", "8G"),
        cpus_per_task=slurm_raw.get("cpus_per_task", 1),
        gres=slurm_raw.get("gres"),
        extra_args=slurm_raw.get("extra_args", []),
    )

    # Hydra config
    hydra_raw = raw.get("hydra", {})
    hydra = HydraConfig(
        command=hydra_raw.get("command", "python train.py"),
        config_path=hydra_raw.get("config_path"),
        static_overrides=hydra_raw.get("static_overrides", []),
    )

    # Launcher
    launcher = raw.get("launcher", "")
    if launcher and not os.path.isabs(launcher):
        launcher = os.path.normpath(os.path.join(config_dir, launcher))

    # Parameters
    params_raw = raw.get("parameters", {})
    if not params_raw:
        raise ConfigError("Config must define at least one parameter")
    parameters = [_parse_parameter(name, spec) for name, spec in params_raw.items()]

    # Validate axes mode has defaults
    if search.mode == "axes":
        if not search.defaults:
            raise ConfigError("axes mode requires 'search.defaults' for all parameters")
        param_names = {p.name for p in parameters}
        missing = param_names - set(search.defaults.keys())
        if missing:
            raise ConfigError(f"axes mode missing defaults for: {missing}")

    # Constraints
    constraints_raw = raw.get("constraints", [])
    constraints = [_parse_constraint(c) for c in constraints_raw]

    return Config(
        name=name,
        workspace=workspace,
        search=search,
        slurm=slurm,
        hydra=hydra,
        launcher=launcher,
        parameters=parameters,
        constraints=constraints,
    )
