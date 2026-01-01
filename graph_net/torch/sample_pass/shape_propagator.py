from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
from graph_net.torch.fx_graph_module_util import (
    get_torch_module_and_inputs,
)
from pathlib import Path
import json
import torch
from torch.fx.passes.shape_prop import ShapeProp


class ShapePropagator(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        device: str = "auto",
        output_json_file_name: str = "shape_prop.json",
        shape_prop_json_key: str = "op_name_and_tensor_output_shape_list",
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        file_name = self.config["output_json_file_name"]
        return self.naive_sample_handled(rel_model_path, search_file_name=file_name)

    def resume(self, rel_model_path: str):
        model_path = Path(self.config["model_path_prefix"]) / rel_model_path
        device = self._choose_device(self.config["device"])
        shape_prop = FxGraphShapePropagator(model_path, device)
        op_and_shapes = shape_prop.infer_op_name_and_tensor_output_shape_list()
        json_obj = {
            self.config["shape_prop_json_key"]: op_and_shapes,
        }
        op_and_shapes_json = json.dumps(json_obj, indent=4)
        output_dir_path = Path(self.config["output_dir"]) / rel_model_path
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / self.config["output_json_file_name"]
        output_file_path.write_text(op_and_shapes_json)

    def _choose_device(self, device) -> str:
        if device in ["cpu", "cuda"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"


class FxGraphShapePropagator:
    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path
        self.device = device

    def infer_op_name_and_tensor_output_shape_list(self):
        data = [
            (self._get_op_name(node), self._get_tensor_output_shape(node))
            for node in self._shape_propagated_nodes()
        ]
        return data

    def _get_tensor_output_shape(self, node):
        meta = node.meta.get("tensor_meta")
        if meta is None:
            return None
        if not hasattr(meta, "shape"):
            return None
        if not isinstance(meta.shape, (list, tuple)):
            return None
        return meta.shape

    def _get_op_name(self, node):
        if node.op == "call_method":
            return f"Tensor.{node.target}"
        elif node.op == "call_function":
            return getattr(node.target, "__name__", str(node.target))
        else:
            return node.op

    def _shape_propagated_nodes(self):
        model_path = str(self.model_path)
        module, inputs = get_torch_module_and_inputs(
            model_path, use_dummy_inputs=False, device=self.device
        )
        gm = parse_immutable_model_path_into_sole_graph_module(
            model_path, device=self.device
        )
        ShapeProp(gm).propagate(*inputs)
        for node in gm.graph.nodes:
            yield node
