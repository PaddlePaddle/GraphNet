import sys
import torch
from .graph_compiler_backend import GraphCompilerBackend


# Predefined Inductor config templates.
# Each template maps to a set of torch._inductor.config overrides.
# Reference: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
#
# Note: These are extension to PyTorch's official "mode" concept.
# PyTorch modes: "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
# These templates provide additional config combinations for specific use cases.
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
    "cutlass": {
        # Enable max-autotune to potentially use CUTLASS-based GEMM kernels.
        # CUTLASS backend requires separate installation.
        # Reference: torch._inductor.config.max_autotune_gemm_backends
        "max_autotune": True,
        "max_autotune_gemm": True,
        "epilogue_fusion": True,
        "coordinate_descent_tuning": True,
    },
    "aten": {
        # Enable autotune fallback to ATen kernels for debugging.
        # This causes Inductor to fall back to ATen (eager) kernels
        # when autotuning finds them faster. Useful for debugging.
        # Reference: torch._inductor.config.autotune_fallback_to_aten
        "autotune_fallback_to_aten": True,
    },
    "cudagraphs": {
        # Enable CUDA Graphs to reduce kernel launch overhead.
        # Reference: torch._inductor.config.triton.cudagraphs
        # Note: Prefer using mode="reduce-overhead" for official support.
        "triton.cudagraphs": True,
    },
    "max_autotune": {
        # Enable comprehensive autotuning across all backends.
        # Equivalent to torch.compile(mode="max-autotune") with extra options.
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
        # NOTE: This config has graceful fallback behavior:
        #   - On NVIDIA H100+ (Hopper, CC >= 9.0): Enables TMA persistent kernels
        #   - On other GPUs (A100, AMD, etc.): Enables non-TMA persistent kernels as fallback
        # Reference: torch._inductor.config.triton.enable_persistent_tma_matmul
        "triton.enable_persistent_tma_matmul": True,
    },
}

# Map template names to torch.compile mode strings where applicable.
# Reference: https://pytorch.org/docs/stable/generated/torch.compile.html
_TEMPLATE_TO_COMPILE_MODE = {
    "cudagraphs": "reduce-overhead",
    "max_autotune": "max-autotune",
}


def _set_nested_attr(config_module, key, value):
    """Set a possibly nested attribute on a config module.

    For example, key="triton.cudagraphs" sets config_module.triton.cudagraphs = value.
    """
    parts = key.split(".")
    obj = config_module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class InductorBackend(GraphCompilerBackend):
    """Inductor backend with configurable config template selection.

    Supported config keys:
        template (str): One of "triton", "cpp_wrapper", "cutlass", "aten",
            "cudagraphs", "max_autotune", "freezing", "tma".
            Applies a predefined set of Inductor config overrides.
            Note: These are extensions to PyTorch's official "mode" concept.
        mode (str): torch.compile mode. One of "default", "reduce-overhead",
            "max-autotune", "max-autotune-no-cudagraphs".
            If a template implies a mode, that is used unless explicitly overridden.
        freezing (bool): Enable/disable model freezing before compilation.
        inductor_config (dict): Arbitrary torch._inductor.config overrides.
            Keys can be dotted paths (e.g. "triton.cudagraphs").
            These are applied last and override everything else.

    Reference:
        - PyTorch modes: https://pytorch.org/docs/stable/generated/torch.compile.html
        - Inductor configs: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
    """

    def __init__(self, config):
        super().__init__(config)
        self._template = config.get("template", None)
        self._mode = config.get("mode", None)
        self._freezing = config.get("freezing", None)
        self._inductor_config = config.get("inductor_config", {})

    def _build_inductor_overrides(self):
        """Collect all Inductor config overrides from template + explicit config."""
        overrides = {}

        # 1. Apply template defaults
        if self._template is not None:
            if self._template not in _INDUCTOR_CONFIG_TEMPLATES:
                raise ValueError(
                    f"Unknown Inductor config template: {self._template!r}. "
                    f"Available templates: {sorted(_INDUCTOR_CONFIG_TEMPLATES.keys())}"
                )
            overrides.update(_INDUCTOR_CONFIG_TEMPLATES[self._template])

        # 2. Apply top-level convenience flags
        if self._freezing is not None:
            overrides["freezing"] = self._freezing

        # 3. Apply explicit inductor_config overrides (highest priority)
        overrides.update(self._inductor_config)

        return overrides

    def _resolve_compile_mode(self):
        """Determine the torch.compile mode string."""
        if self._mode is not None:
            return self._mode
        if self._template in _TEMPLATE_TO_COMPILE_MODE:
            return _TEMPLATE_TO_COMPILE_MODE[self._template]
        return "default"

    def __call__(self, model):
        import torch._inductor.config as inductor_config

        overrides = self._build_inductor_overrides()
        compile_mode = self._resolve_compile_mode()

        if self._template or self._inductor_config:
            print(
                f"[InductorBackend] template={self._template!r}, mode={compile_mode!r}, "
                f"overrides={overrides}",
                file=sys.stderr,
                flush=True,
            )

        # Apply Inductor config overrides
        for key, value in overrides.items():
            _set_nested_attr(inductor_config, key, value)

        return torch.compile(model, backend="inductor", mode=compile_mode)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
