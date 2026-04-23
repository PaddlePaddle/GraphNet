import sys
import torch
from .graph_compiler_backend import GraphCompilerBackend


class InductorBackend(GraphCompilerBackend):
    """Inductor backend wrapping ``torch.compile(backend="inductor")``.

    Config keys (all optional, forwarded to ``torch.compile``):
        mode (str): Compilation mode, e.g. "default", "reduce-overhead",
            "max-autotune".
        options (dict): Inductor-specific options mapping to
            ``torch._inductor.config`` keys. Mutually exclusive with *mode*.
        fullgraph (bool): Compile the entire model as a single graph.
            Defaults to False.
        dynamic (bool | None): Enable dynamic-shape tracing.
            Defaults to None (use torch defaults).

    See Also:
        https://pytorch.org/docs/stable/generated/torch.compile.html
    """

    def __init__(self, config):
        super().__init__(config)
        self._mode = config.get("mode")
        self._fullgraph = config.get("fullgraph", False)
        self._dynamic = config.get("dynamic")
        self._options = config.get("options")

        if self._mode is not None and self._options is not None:
            raise ValueError(
                "'mode' and 'options' are mutually exclusive; "
                "specify at most one of them."
            )

    def __call__(self, model):
        if self._mode is not None or self._options is not None:
            print(
                f"[InductorBackend] mode={self._mode!r}, options={self._options}",
                file=sys.stderr,
                flush=True,
            )

        return torch.compile(
            model,
            backend="inductor",
            fullgraph=self._fullgraph,
            dynamic=self._dynamic,
            **({"mode": self._mode} if self._mode is not None else {}),
            **({"options": self._options} if self._options else {}),
        )

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
