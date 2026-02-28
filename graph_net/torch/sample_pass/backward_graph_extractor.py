import os
import inspect
from pathlib import Path

import torch
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func

from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor
from graph_net.torch.fx_graph_module_util import (
    get_torch_module_and_inputs,
    _get_tensor_metas,
)


class BackwardGraphExtractor:
    def __init__(self, model_name, model_path, output_dir, device):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device

    def __call__(self):
        module, forward_inputs = get_torch_module_and_inputs(
            self.model_path, use_dummy_inputs=False, device=self.device
        )
        module.train()

        forward_inputs = self.set_requires_grad_for_forward_inputs(
            self.model_path, module, forward_inputs
        )
        gm_holder, backward_inputs = self.capture_graph(module, forward_inputs)
        self.get_extractor("forward")(gm_holder["forward_gm"], forward_inputs)
        self.get_extractor("backward")(gm_holder["backward_gm"], backward_inputs)

    def get_extractor(self, suffix):
        return BuiltinGraphExtractor(
            name=f"{self.model_name}_{suffix}",
            dynamic=False,
            mut_graph_codes=[],
            placeholder_auto_rename=False,
            workspace_path=self.output_dir,
        )

    def capture_graph(self, module, forward_inputs):
        gm_holder = {}
        backward_inputs = []

        def forward_compiler(fx_gm, forward_inputs):
            gm_holder["forward_gm"] = fx_gm
            return fx_gm

        def backward_compiler(fx_gm, forward_inputs):
            gm_holder["backward_gm"] = fx_gm

            placeholders = [n for n in fx_gm.graph.nodes if n.op == "placeholder"]
            origin_forward = fx_gm.forward

            def wrapped_forward(*args):
                for node, arg in zip(placeholders, args):
                    backward_inputs.append(arg.detach().clone())
                return origin_forward(*args)

            fx_gm.forward = wrapped_forward
            return make_boxed_func(fx_gm)

        compiled = aot_module_simplified(
            module,
            forward_inputs,
            fw_compiler=forward_compiler,
            bw_compiler=backward_compiler,
        )
        outs = compiled(*forward_inputs)
        outs = [outs] if isinstance(outs, torch.Tensor) else outs
        valid_pairs = [
            (out, torch.ones_like(out))
            for out in outs
            if isinstance(out, torch.Tensor) and out.requires_grad
        ]

        if valid_pairs:
            tensors, grads = zip(*valid_pairs)
            torch.autograd.backward(tensors, grads)
            gm_holder["backward_gm"] = self._remove_none_from_output(
                gm_holder["backward_gm"]
            )

        return gm_holder, backward_inputs

    def _remove_none_from_output(self, gm):
        output_node = next(
            (n for n in gm.graph.nodes if n.op == "output"),
            None,
        )
        outs = (
            output_node.args[0]
            if output_node and isinstance(output_node.args, (tuple, list))
            else output_node.args
        )
        if isinstance(outs, (tuple, list)):
            new_outs = tuple(out for out in outs if out is not None)
            if new_outs != outs:
                output_node.args = (new_outs,)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm

    def _requires_grad(self, name, tensor):
        if not tensor.is_floating_point():
            return False

        nograd_parameter_keywords = [
            "running_mean",
            "running_var",
            "num_batches_tracked",
            "mask",
            "indices",
            "position_ids",
            "anchor",
        ]
        for keyword in nograd_parameter_keywords:
            if keyword in name:
                return False

        return True

    def set_requires_grad_for_forward_inputs(
        self, model_path, graph_module, example_inputs
    ):
        tensor_metas = _get_tensor_metas(model_path)
        name2tensor_meta = {
            tensor_meta.name: tensor_meta for tensor_meta in tensor_metas
        }
        for input_idx, name in enumerate(
            inspect.signature(graph_module.forward).parameters
        ):
            tensor = example_inputs[input_idx]
            tensor_meta = name2tensor_meta[name]
            original_name = (
                tensor_meta.original_name
                if hasattr(tensor_meta, "original_name") and tensor_meta.original_name
                else name
            )
            tensor.requires_grad = self._requires_grad(original_name, tensor)
            # print(f"{name}, {original_name}, requires_grad:{tensor.requires_grad}")
        return example_inputs


class BackwardGraphExtractorPass(SamplePass, ResumableSamplePassMixin):
    """SamplePass wrapper to generate Torch unittests via model_path_handler."""

    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        device: str = "auto",
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        return self.naive_sample_handled(rel_model_path, search_file_name="model.py")

    def resume(self, rel_model_path: str):
        model_path_prefix = Path(self.config["model_path_prefix"])
        model_name = f"{os.path.basename(rel_model_path)}"
        model_path = model_path_prefix / rel_model_path
        output_dir = (
            Path(self.config["output_dir"])
            / os.path.dirname(rel_model_path)
            / model_name
        )
        device = self._choose_device(self.config["device"])
        extractor = BackwardGraphExtractor(model_name, model_path, output_dir, device)
        extractor()

    def _choose_device(self, device) -> str:
        if device in ["cpu", "cuda"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
