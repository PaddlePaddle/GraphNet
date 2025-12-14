"""
Pass manager for dtype generalization passes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from graph_net.imp_util import load_module

if TYPE_CHECKING:
    from graph_net.torch.dtype_gen_passes.pass_base import DtypeGeneralizationPass


def get_dtype_generalization_pass(pass_name: str) -> type[DtypeGeneralizationPass]:
    """
    Load a dtype generalization pass by name.

    Args:
        pass_name: Name of the pass file (without .py extension)

    Returns:
        Pass class (not instance)
    """
    import graph_net.torch.dtype_gen_passes as dgpass

    py_module = load_module(f"{os.path.dirname(dgpass.__file__)}/{pass_name}.py")
    return py_module.ConcretePass
