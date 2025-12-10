import torch
from pathlib import Path
from typing import Dict, Any
from graph_net.imp_util import load_module
from graph_net.tensor_meta import TensorMeta


class RenamedModelAdapter(torch.nn.Module):
    def __init__(self, renamed_model: torch.nn.Module, mapping: Dict[str, str]):
        super().__init__()
        self.model = renamed_model
        self.mapping = mapping
        if hasattr(renamed_model, "__graph_net_file_path__"):
            self.__graph_net_file_path__ = renamed_model.__graph_net_file_path__

    def forward(self, **kwargs):
        new_kwargs = {}
        for old_name, value in kwargs.items():
            if old_name in self.mapping:
                new_name = self.mapping[old_name]
                new_kwargs[new_name] = value
        return self.model(**new_kwargs)


class GraphVariableRenamerValidatorBackend:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _get_rename_mapping(self, model_dir: Path) -> Dict[str, str]:
        mapping = {}
        if not model_dir.exists():
            print(f"[ValidatorBackend] Error: Model dir does not exist: {model_dir}")
            return mapping

        for meta_file in ["input_meta.py", "weight_meta.py"]:
            meta_path = model_dir / meta_file
            if meta_path.exists():
                try:
                    metas = TensorMeta.unserialize_from_py_file(str(meta_path))
                    for m in metas:
                        if m.original_name:
                            mapping[m.original_name] = m.name
                except Exception as e:
                    print(
                        f"[ValidatorBackend] Warning: Failed to parse {meta_path}: {e}"
                    )
        return mapping

    def _load_renamed_model(
        self, model_dir: Path, device: torch.device
    ) -> torch.nn.Module:
        model_py_path = model_dir / "model.py"
        if not model_py_path.exists():
            raise FileNotFoundError(f"Renamed model not found at {model_py_path}")

        py_module = load_module(str(model_py_path))

        if not hasattr(py_module, "GraphModule"):
            raise ValueError(f"GraphModule class not found in {model_py_path}")

        GraphModule = getattr(py_module, "GraphModule")
        GraphModule.__graph_net_file_path__ = str(model_py_path)

        model = GraphModule()
        model.to(device)
        model.eval()
        return model

    def __call__(self, original_model: torch.nn.Module) -> torch.nn.Module:
        renamed_root = self.config.get("renamed_root")
        if not renamed_root:
            raise ValueError("Config 'renamed_root' is missing!")

        default_prefix = str(Path(__file__).resolve().parent.parent.parent.parent)
        model_path_prefix = self.config.get("model_path_prefix", default_prefix)

        if not hasattr(original_model, "__graph_net_file_path__"):
            raise ValueError("Original model missing __graph_net_file_path__")

        orig_abs_path = Path(original_model.__class__.__graph_net_file_path__).resolve()
        orig_model_dir = orig_abs_path.parent

        try:
            rel_model_path = orig_model_dir.relative_to(model_path_prefix)
        except ValueError:
            print(
                f"[ValidatorBackend] Warning: Model path {orig_model_dir} is not under prefix {model_path_prefix}. Fallback to leaf name."
            )
            rel_model_path = orig_model_dir.name

        renamed_model_dir = Path(renamed_root) / rel_model_path

        print(f"[ValidatorBackend] Original Path: {orig_model_dir}")
        print(f"[ValidatorBackend] Relative Path: {rel_model_path}")
        print(f"[ValidatorBackend] Loading Renamed: {renamed_model_dir}")

        try:
            device = next(original_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        renamed_core_model = self._load_renamed_model(renamed_model_dir, device)
        mapping = self._get_rename_mapping(renamed_model_dir)

        if not mapping:
            print(
                f"[ValidatorBackend] Warning: Mapping is empty for {rel_model_path}. Check input_meta.py generation."
            )

        adapter = RenamedModelAdapter(renamed_core_model, mapping)
        adapter.to(device)
        adapter.eval()

        return adapter

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
