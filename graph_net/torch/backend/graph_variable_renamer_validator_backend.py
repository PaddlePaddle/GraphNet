import torch
from pathlib import Path
from typing import Dict
from graph_net.tensor_meta import TensorMeta
import os
import importlib.util


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


class GraphVariableRenamerValidatorBackend:
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
        renamed_dentry: str = "_renamed",
    ):
        return {
            "model_path_prefix": model_path_prefix,
            "renamed_root": renamed_root,
            "renamed_dentry": renamed_dentry,
        }

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        config = self._make_config(**self.config)
        model_path = os.path.dirname(model.__class__.__graph_net_file_path__)
        model_name = os.path.basename(model_path)
        renamed_dir_name = f"{model_name}_renamed"
        renamed_model_dir = os.path.join(config["renamed_root"], renamed_dir_name)

        print(f"[GraphVariableRenamerValidatorBackend] Processing: {model_name}")
        print(
            f"[GraphVariableRenamerValidatorBackend] Loading from: {renamed_model_dir}"
        )

        device = model.__class__.__graph_net_device__
        renamed_model = self._load_model_instance(renamed_model_dir, device)
        mapping = self._get_rename_mapping(Path(renamed_model_dir))
        assert (
            mapping
        ), f"Mapping is empty for {renamed_dir_name} at {renamed_model_dir}"
        adapter = RenamedModelAdapter(renamed_model, mapping)
        return adapter.eval()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
