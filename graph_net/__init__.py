__all__ = ["torch", "paddle"]

from importlib import import_module
from typing import TYPE_CHECKING, Any, List


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from . import torch as torch  # type: ignore
    from . import paddle as paddle  # type: ignore
