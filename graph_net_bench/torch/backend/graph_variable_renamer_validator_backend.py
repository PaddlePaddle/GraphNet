import torch
from pathlib import Path
from typing import Dict
from graph_net.tensor_meta import TensorMeta
import os
import importlib.util
from .graph_compiler_backend import GraphCompilerBackend


class RenamedModelAdapter(torch.nn.Module):
    def __init__(self, renamed_model: torch.nn.Module, mapping: Dict[str, str]):
        super().__init__()
        self.model = renamed_model
        self.mapping = mapping
        if hasattr(renamed_model, "__graph_net_file_path__"):
            self.__graph_net_file_path__ = renamed_model.__graph_net_file_path__

    def forward(self, **kwargs):
        new_kwargs = self._convert_by_name_mapping(kwargs)
        return self.model(**new_kwargs)

    def _convert_by_name_mapping(self, kwargs):
        new_kwargs = {}
        for old_name, value in kwargs.items():
            if old_name in self.mapping:
                new_name = self.mapping[old_name]
                new_kwargs[new_name] = value
        return new_kwargs


class GraphVariableRenamerValidatorBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)

    def _get_rename_mapping(self, model_dir: Path):
        mapping = {}
        for meta_file in ["input_meta.py", "weight_meta.py"]:
            meta_path = model_dir / meta_file
            if not meta_path.exists():
                continue
            metas = TensorMeta.unserialize_from_py_file(str(meta_path))
            for m in metas:
                if m.original_name:
                    mapping[m.original_name] = m.name
        return mapping

    def _load_model_instance(self, path: str, device: str) -> torch.nn.Module:
        class_name = "GraphModule"
        model_file = os.path.join(path, "model.py")

        spec = importlib.util.spec_from_file_location(class_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ModelClass = getattr(module, class_name)
        instance = ModelClass().to(device)
        return instance

    def _make_config(
        self,
        model_path_prefix: str,
        renamed_root: str,
    ):
        return {
            "model_path_prefix": model_path_prefix,
            "renamed_root": renamed_root,
        }

    def _get_rel_model_path(self, model_path) -> str:
        model_path = os.path.realpath(model_path)
        model_path_prefix = os.path.realpath(self.config["model_path_prefix"])
        assert model_path.startswith(model_path_prefix)
        rel_model_path = model_path[len(model_path_prefix) :]
        if rel_model_path.startswith("/"):
            rel_model_path = rel_model_path[1:]
        assert not rel_model_path.startswith("/")
        return rel_model_path

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        config = self._make_config(**self.config)
        model_path = os.path.dirname(model.__class__.__graph_net_file_path__)
        model_name = os.path.basename(model_path)
        rel_model_path = self._get_rel_model_path(model_path)
        renamed_parent_dir = os.path.join(config["renamed_root"], rel_model_path)

        print(f"[GraphVariableRenamerValidatorBackend] Processing: {model_name}")
        print(
            f"[GraphVariableRenamerValidatorBackend] Loading from: {renamed_parent_dir}"
        )

        device = model.__class__.__graph_net_device__
        renamed_model = self._load_model_instance(renamed_parent_dir, device)
        mapping = self._get_rename_mapping(Path(renamed_parent_dir))
        assert mapping, f"Mapping is empty for {model_name} at {renamed_parent_dir}"
        adapter = RenamedModelAdapter(renamed_model, mapping)
        return adapter.eval()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
