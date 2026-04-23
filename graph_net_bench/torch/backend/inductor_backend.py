import sys
import torch
from .graph_compiler_backend import GraphCompilerBackend


# Predefined Inductor config templates for GraphNet.
# Each template maps to a set of torch._inductor.config options.
_INDUCTOR_CONFIG_TEMPLATES = {
    "triton": {
        # Default Triton code generation (Inductor's default behavior).
        # Explicitly disable cpp_wrapper to ensure Triton backend.
        "cpp_wrapper": False,
    },
    "cpp_wrapper": {
        # Use C++ wrapper for generated kernels instead of Python wrapper.
        # Reference: torch._inductor.config.cpp_wrapper
        "cpp_wrapper": True,
    },
    "cudagraphs": {
        # Enable CUDA Graphs to reduce kernel launch overhead.
        # Reference: torch._inductor.config.triton.cudagraphs
        # NOTE: CUDA Graphs does not support dynamic shapes or graph breaks, so it is
        # highly recommended to use only for inference with fixed input shapes.
        # Prefer using mode="reduce-overhead" for official support.
        "triton.cudagraphs": True,
    },
    "max_autotune": {
        # Enable comprehensive autotuning across all backends.
        "max_autotune": True,
        "max_autotune_gemm": True,
        "coordinate_descent_tuning": True,
        "epilogue_fusion": True,
    },
    "freezing": {
        # Enable model freezing to inline weights as constants.
        # After freezing, weights can no longer be updated.
        # Reference: torch._inductor.config.freezing
        "freezing": True,
    },
    "tma": {
        # Enable persistent matmul kernels with TMA (Tensor Memory Accelerator).
        # Reference: torch._inductor.config.triton.enable_persistent_tma_matmul
        # NOTE: This config has graceful fallback behavior:
        #   - On NVIDIA H100+ (Hopper, CC >= 9.0): Enables TMA persistent kernels
        #   - On other GPUs (A100, AMD, etc.): Enables non-TMA persistent kernels as fallback
        "triton.enable_persistent_tma_matmul": True,
    },
}


class InductorBackend(GraphCompilerBackend):
    """Inductor backend with configurable config template selection.

    Supported config keys:
        fullgraph (bool): Whether to compile the entire model into a single graph.
            Defaults to False. Passed to torch.compile.
        dynamic (bool | None): Whether to enable dynamic shape support.
            Defaults to None. Passed to torch.compile.
        graph_net_inductor_config_template (str): One of "triton", "cpp_wrapper",
            "cudagraphs", "max_autotune", "freezing", "tma".
            GraphNet predefined Inductor config templates.
        mode (str): torch.compile mode. One of "default", "reduce-overhead",
            "max-autotune". Passed directly to torch.compile.
        options (dict): Direct torch.compile options dict.

        NOTE: `torch.compile` does not allow `mode` and `options` to be specified together.
        Therefore `graph_net_inductor_config_template` (which provides `options`),
        `mode` and `options` are mutually exclusive with each other.

    Reference:
        - PyTorch compile: https://pytorch.org/docs/stable/generated/torch.compile.html
        - Inductor configs: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
    """

    def __init__(self, config):
        super().__init__(config)
        self._config_template = config.get("graph_net_inductor_config_template")
        self._mode = config.get("mode")
        self._fullgraph = config.get("fullgraph", False)
        self._dynamic = config.get("dynamic")
        self._options = config.get("options")

        # Mutually exclusive: exactly one of template / mode / options can be specified
        provided = [self._config_template, self._mode, self._options]
        if sum(1 for x in provided if x is not None) > 1:
            raise ValueError(
                "Specify exactly one of 'graph_net_inductor_config_template', "
                "'mode', or 'options', not multiple."
            )

    def _build_options(self):
        if self._options:
            return self._options

        if self._config_template:
            if self._config_template not in _INDUCTOR_CONFIG_TEMPLATES:
                raise ValueError(
                    f"Unknown config template: {self._config_template!r}. "
                    f"Available: {sorted(_INDUCTOR_CONFIG_TEMPLATES.keys())}"
                )
            return _INDUCTOR_CONFIG_TEMPLATES[self._config_template]

        return {}

    def __call__(self, model):
        final_options = self._build_options()

        if self._config_template is not None or self._options is not None:
            print(
                f"[InductorBackend] graph_net_inductor_config_template={self._config_template!r}, "
                f"mode={self._mode!r}, options={final_options}",
                file=sys.stderr,
                flush=True,
            )

        return torch.compile(
            model,
            backend="inductor",
            fullgraph=self._fullgraph,
            dynamic=self._dynamic,
            **({"mode": self._mode} if self._mode is not None else {}),
            **({"options": final_options} if final_options else {}),
        )

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
