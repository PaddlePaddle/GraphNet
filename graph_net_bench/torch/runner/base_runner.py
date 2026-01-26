from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum


class RunnerType(Enum):
    LOCAL = "local"
    PROCESS = "process"
    REMOTE = "remote"


@dataclass
class ExecutionConfig:
    """Configuration specific to model execution."""

    compiler: str = "inductor"
    device: str = "cuda"
    op_lib: str = "default"
    warmup: int = 5
    trials: int = 10
    seed: int = 123
    log_prompt: str = "graph-net-runner-log"
    backend_config: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class RunnerStrategyConfig:
    """Configuration for runner strategy selection."""

    runner_type: RunnerType = RunnerType.LOCAL
    remote_machine: str = "localhost"
    remote_port: int = 50052
    subprocess_timeout: int = 600

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunnerStrategyConfig":
        runner_type_str = d.get("runner_type", "local")
        try:
            runner_type = RunnerType(runner_type_str.lower())
        except ValueError:
            runner_type = RunnerType.LOCAL

        return cls(
            runner_type=runner_type,
            remote_machine=d.get("machine", "localhost"),
            remote_port=d.get("port", 50052),
            subprocess_timeout=d.get("subprocess_timeout", 600),
        )


@dataclass
class RunnerConfig:
    """Unified configuration combining execution and strategy configs."""

    execution: ExecutionConfig
    strategy: RunnerStrategyConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunnerConfig":
        execution_config = ExecutionConfig.from_dict(d)
        strategy_config = RunnerStrategyConfig.from_dict(d)
        return cls(execution=execution_config, strategy=strategy_config)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.execution.to_dict(),
            "runner_type": self.strategy.runner_type.value,
            "machine": self.strategy.remote_machine,
            "port": self.strategy.remote_port,
            "subprocess_timeout": self.strategy.subprocess_timeout,
        }


@dataclass
class RunResult:
    """Result of a single backend run."""

    success: bool = False
    outputs: Optional[Tuple[Any, ...]] = None
    time_stats: Dict[str, Any] = field(default_factory=dict)
    log_content: str = ""
    error_message: str = ""

    output_path: Optional[Path] = None
    log_path: Optional[Path] = None


class BaseRunner(ABC):
    """Abstract base class for model execution runners."""

    def __init__(self, config: RunnerConfig):
        self.config = config

    @abstractmethod
    def run(self, model_path: str, output_dir: str) -> RunResult:
        """
        Execute model evaluation and return results.

        Args:
            model_path: Path to model directory (containing model.py, graph_net.json, etc.)
            output_dir: Directory to store outputs and logs

        Returns:
            RunResult containing outputs, timing stats, and logs
        """
        pass

    def _get_output_path(self, output_dir: str, model_path: str) -> Path:
        from graph_net_bench.torch import utils

        return Path(utils.get_output_path(output_dir, model_path))

    def _get_log_path(self, output_dir: str, model_path: str) -> Path:
        from graph_net_bench.torch import utils

        return Path(utils.get_log_path(output_dir, model_path))


def _get_runner_class(runner_type: RunnerType) -> type:
    """Get runner class by type with lazy imports."""
    if runner_type == RunnerType.LOCAL:
        from .local_runner import LocalRunner

        return LocalRunner
    if runner_type == RunnerType.PROCESS:
        from .process_runner import ProcessRunner

        return ProcessRunner
    if runner_type == RunnerType.REMOTE:
        from .remote_runner import RemoteRunner

        return RemoteRunner
    raise ValueError(f"Unknown runner_type: {runner_type}")


def create_runner(config: RunnerConfig) -> BaseRunner:
    """Factory function to create appropriate runner based on config."""
    runner_class = _get_runner_class(config.strategy.runner_type)
    return runner_class(config)
