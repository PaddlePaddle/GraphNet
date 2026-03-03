import os
from pathlib import Path
from typing import List
import paddle

from graph_net import imp_util
from graph_net.paddle import utils
from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = imp_util.load_module(file_path, "unnamed")
    model_class = getattr(module, class_name, None)
    setattr(model_class, "__graph_net_file_path__", os.path.normpath(file_path))
    return model_class


def get_input_spec(model_path):
    inputs_params_list = utils.load_converted_list_from_text(f"{model_path}")
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


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

    def _get_static_program(self, model_path):
        model_class = load_class_from_file(
            os.path.join(model_path, "model.py"), "GraphModule"
        )
        model = model_class()
        model.eval()

        static_model = paddle.jit.to_static(
            model,
            input_spec=get_input_spec(model_path),
            full_graph=True,
            backend=None,
        )
        static_model.eval()
        program = static_model.forward.concrete_program.main_program
        return program

    def _extract_ops(self, model_path: str) -> List[str]:
        program = self._get_static_program(model_path)

        operator_list = []
        for block in program.blocks:
            for op in block.ops:
                if op.name() == "pd_op.data":
                    pass
                elif op.name().startswith("pd_op."):
                    operator_list.append(op.name().replace("pd_op.", ""))
                elif not op.name().startswith("builtin."):
                    assert False, f"Unrecognized op: {op}"
        return operator_list
