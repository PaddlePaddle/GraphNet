"""
Dtype generalization pass for bfloat16.

This pass converts float32 tensors to bfloat16.
"""

from graph_net.torch.dtype_gen_passes.dtype_generalization_pass import (
    ConcretePass as BaseConcretePass,
)

# Weights that must remain float32 for numerical stability
FLOAT32_PRESERVED_WEIGHTS = {
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "bn_parameters_weight",
    "bn_parameters_bias",
    "ln_parameters_weight",
    "ln_parameters_bias",
}


class ConcretePass(BaseConcretePass):
    """
    FX Graph pass that converts dtypes to bfloat16.
    """

    def __init__(self, *args, **kwargs):
        # Override target_dtype to bfloat16
        super().__init__(
            target_dtype="bfloat16",
            preserve_weights=FLOAT32_PRESERVED_WEIGHTS,
            *args,
            **kwargs,
        )
