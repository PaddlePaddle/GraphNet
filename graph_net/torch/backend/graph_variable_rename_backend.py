import torch
import graph_net
from pathlib import Path
from typing import Any, Dict
from graph_net.imp_util import load_module
from graph_net.torch.graph_variable_renamer import GraphVariableRenamer
from graph_net.tensor_meta import TensorMeta


class RenamedModelAdapter(torch.nn.Module):
    def __init__(self, renamed_model, mapping):
        super().__init__()
        self.model = renamed_model
        self.mapping = mapping

    def forward(self, **kwargs):
        new_kwargs = {}
        for old_name, value in kwargs.items():
            if old_name in self.mapping:
                new_name = self.mapping[old_name]
                new_kwargs[new_name] = value

        return self.model(**new_kwargs)


class GraphVariableRenameBackend:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.config = config
        self.workspace_path = Path(
            self.config.get("workspace_path", "./tmp/graph_variable_rename_workspace")
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    def _get_default_paths(self):
        lib_root = Path(graph_net.__file__).parent
        default_util_path = str(lib_root / "torch/constraint_util.py")
        return default_util_path

    def _get_rename_mapping(self, dst_model_dir: Path) -> Dict[str, str]:
        mapping = {}

        input_meta_path = dst_model_dir / "input_meta.py"
        if input_meta_path.exists():
            metas = TensorMeta.unserialize_from_py_file(str(input_meta_path))
            for m in metas:
                if m.original_name:
                    mapping[m.original_name] = m.name

        weight_meta_path = dst_model_dir / "weight_meta.py"
        if weight_meta_path.exists():
            metas = TensorMeta.unserialize_from_py_file(str(weight_meta_path))
            for m in metas:
                if m.original_name:
                    mapping[m.original_name] = m.name

        return mapping

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        print("\n[GraphVariableRenameBackend] Starting rename process...")

        if not hasattr(model.__class__, "__graph_net_file_path__"):
            raise ValueError(
                "Input model must be a GraphNet model with __graph_net_file_path__ attribute."
            )

        src_file_path = Path(model.__class__.__graph_net_file_path__).resolve()
        src_model_dir = src_file_path.parent
        model_rel_path = src_model_dir.name
        model_path_prefix = str(src_model_dir.parent)
        default_util_path = self._get_default_paths()
        data_input_predicator_filepath = self.config.get(
            "data_input_predicator_filepath", default_util_path
        )
        data_input_predicator_class_name = self.config.get(
            "data_input_predicator_class_name", "NaiveDataInputPredicator"
        )
        data_input_predicator_config = self.config.get(
            "data_input_predicator_config", {}
        )
        model_runnable_predicator_filepath = self.config.get(
            "model_runnable_predicator_filepath", default_util_path
        )
        model_runnable_predicator_class_name = self.config.get(
            "model_runnable_predicator_class_name", "ModelRunnablePredicator"
        )
        model_runnable_predicator_config = self.config.get(
            "model_runnable_predicator_config", {}
        )

        output_dir = str(self.workspace_path)

        renamer_config = {
            "output_dir": output_dir,
            "model_path_prefix": model_path_prefix,
            "data_input_predicator_filepath": data_input_predicator_filepath,
            "data_input_predicator_class_name": data_input_predicator_class_name,
            "data_input_predicator_config": data_input_predicator_config,
            "model_runnable_predicator_filepath": model_runnable_predicator_filepath,
            "model_runnable_predicator_class_name": model_runnable_predicator_class_name,
            "model_runnable_predicator_config": model_runnable_predicator_config,
        }

        print(f"[Backend Info] Model Source Dir: {src_model_dir}")
        print(f"[Backend Info] Calculated Prefix: {model_path_prefix}")

        try:
            renamer = GraphVariableRenamer(renamer_config)
            renamer(model_rel_path)
        except Exception as e:
            print(f"[Error] GraphVariableRenamer execution failed: {e}")
            raise e

        dst_model_dir = self.workspace_path / model_rel_path
        print(f"[Success] Renamed model saved to {dst_model_dir}")

        renamed_core_model = self._load_model(dst_model_dir)
        name_mapping = self._get_rename_mapping(dst_model_dir)

        adapter_model = RenamedModelAdapter(renamed_core_model, name_mapping)
        adapter_model.eval()

        return adapter_model

    def _load_model(self, model_dir: Path) -> torch.nn.Module:
        model_py_path = model_dir / "model.py"
        if not model_py_path.exists():
            raise FileNotFoundError(f"Renamed model not found at {model_py_path}")

        py_module = load_module(str(model_py_path))

        if hasattr(py_module, "GraphModule"):
            GraphModule = getattr(py_module, "GraphModule")
            GraphModule.__graph_net_file_path__ = str(model_py_path)
            model = GraphModule()
            return model
        else:
            raise ValueError(f"GraphModule class not found in {model_py_path}")

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
