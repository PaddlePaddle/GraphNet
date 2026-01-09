import os
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch.nn as nn

from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module_without_varify
from graph_net.sample_pass.sample_pass import SamplePass


class OpNamesExtractor(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def sample_handled(self, rel_model_path: str) -> bool:
        return self.naive_sample_handled(
            rel_model_path, search_file_name="op_names.txt"
        )

    def resume(self, rel_model_path: str):
        torch.cuda.empty_cache()
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        op_names = self._extract_ops(model_path)
        output_path = self._get_output_path(rel_model_path)
        output_path.write_text("\n".join(op_names))
        print(f"Save op-names to {str(output_path)}")

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def _get_output_path(self, rel_model_path: str):
        output_path_dir = Path(self.config["output_dir"]) / rel_model_path
        output_path_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_path_dir / "op_names.txt"
        return output_path

    def _extract_ops(self, model_path: str) -> List[str]:
        extractor = TypicalSequenceExtractor()
        model, inputs = get_torch_module_and_inputs(model_path)
        compiled_model, _ = parse_sole_graph_module_without_varify(model, inputs)
        extractor.extract_compiler(compiled_model, inputs)
        ops_info = extractor.extract_node

        return [op["target_name"] for op in ops_info]


class TypicalSequenceExtractor:
    def __init__(self):
        self.extract_node = []

    def _extract_operators_from_graph(
        self, gm: nn.Module, example_inputs: List[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        operator_list = []
        named_modules = dict(gm.named_modules())

        for node in gm.graph.nodes:
            if node.op not in ("call_method", "call_function", "call_module"):
                continue

            if node.op == "call_module":
                target_name = type(named_modules[node.target]).__name__
            elif node.op == "call_method":
                target_name = f"Tensor.{node.target}"
            elif node.op == "call_function":
                target_name = getattr(node.target, "__name__", str(node.target))
            else:
                raise NotImplementedError()
            operator_list.append(
                {
                    "op_type": node.op,
                    "target": node.target,
                    "name": node.name,
                    "target_name": target_name,
                }
            )

        return operator_list

    def extract_compiler(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        operator = self._extract_operators_from_graph(gm, inputs)
        self.extract_node = operator
        return gm.forward
