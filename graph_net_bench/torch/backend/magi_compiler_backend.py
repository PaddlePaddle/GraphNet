import inspect
import importlib.util
import os
import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import magi_compiler
except ImportError:
    magi_compiler = None


def _load_dynamic_arg_dims(model: torch.nn.Module) -> dict[str, list[int]]:
    try:
        import sympy

        model_file = getattr(model.__class__, "__graph_net_file_path__", None)
        if model_file is None:
            model_file = inspect.getfile(model.__class__)
        constraints_path = os.path.join(
            os.path.dirname(model_file), "input_tensor_constraints.py"
        )
        if not os.path.isfile(constraints_path):
            return {}

        spec = importlib.util.spec_from_file_location("_constraints", constraints_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        dynamic_dims: dict[str, list[int]] = {}
        for shape, arg_name in mod.dynamic_dim_constraint_input_shapes:
            dims = [i for i, s in enumerate(shape) if isinstance(s, sympy.Basic)]
            if dims:
                dynamic_dims[arg_name] = dims

        return dynamic_dims
    except Exception:
        return {}


class MagiCompilerBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model):
        dynamic_arg_dims = self.config.get("dynamic_arg_dims", None)
        config_patch = self.config.get("config_patch", None)
        if dynamic_arg_dims is None:
            dynamic_arg_dims = _load_dynamic_arg_dims(model)

        original_cls = model.__class__
        single_use_cls = type(original_cls.__name__, (original_cls,), {})

        if dynamic_arg_dims:
            compiled_cls = magi_compiler.magi_compile(
                single_use_cls,
                dynamic_arg_dims=dynamic_arg_dims,
                config_patch=config_patch,
            )
        else:

            def _static_config_patch(cfg):
                cfg.compile_mode = magi_compiler.config.CompileMode.TORCH_COMPILE
                cfg.backend = "inductor"
                if config_patch is not None:
                    cfg = config_patch(cfg)
                return cfg

            compiled_cls = magi_compiler._api._magi_compile_class(
                single_use_cls, {}, None, _static_config_patch, None
            )

        model.__class__ = compiled_cls
        return model

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
