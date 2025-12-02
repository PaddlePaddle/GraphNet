"""
Pass manager for dtype conversion passes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from graph_net.imp_util import load_module

if TYPE_CHECKING:
    from graph_net.torch.multi_dtype_passes.pass_base import DtypeConversionPass


def get_dtype_conversion_pass(pass_name: str) -> type[DtypeConversionPass]:
    """
    Load a dtype conversion pass by name.

    Args:
        pass_name: Name of the pass file (without .py extension)

    Returns:
        Pass class (not instance)
    """
    import graph_net.torch.multi_dtype_passes as mdpass

    py_module = load_module(f"{os.path.dirname(mdpass.__file__)}/{pass_name}.py")
    return py_module.ConcretePass
