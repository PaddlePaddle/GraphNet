import os
import torch
import shutil
import inspect
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
from graph_net.tensor_meta import TensorMeta
from pathlib import Path
from graph_net.torch.utils import apply_templates
from graph_net.imp_util import load_module
from graph_net.hash_util import get_sha256_hash


class GraphVariableRenamer:
    """
    Used by graph_net.model_path_handler
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)
        self.data_input_predicator = self._make_data_input_predicator(self.config)
        self.model_runnable_predicator = self._make_model_runnable_predicator(
            self.config
        )

    def _make_data_input_predicator(self, config):
        module = load_module(config["data_input_predicator_filepath"])
        cls = getattr(module, config["data_input_predicator_class_name"])
        return cls(config["data_input_predicator_config"])

    def _make_model_runnable_predicator(self, config):
        module = load_module(config["model_runnable_predicator_filepath"])
        cls = getattr(module, config["model_runnable_predicator_class_name"])
        return cls(config["model_runnable_predicator_config"])

    def _make_config(
        self,
        data_input_predicator_filepath,
        model_runnable_predicator_filepath,
        output_dir="./tmp/graph_variable_renamer_dir",
        filter_path=None,
        filter_config=None,
        post_extract_process_path=None,
        post_extract_process_class_name=None,
        post_extract_process_config=None,
        data_input_predicator_class_name="DataInputPredicator",
        model_runnable_predicator_class_name="ModelRunner",
        data_input_predicator_config=None,
        model_runnable_predicator_config=None,
        model_path_prefix="",
        **kwargs,
    ):
        if post_extract_process_config is None:
            post_extract_process_config = {}
        if data_input_predicator_config is None:
            data_input_predicator_config = {}
        if model_runnable_predicator_config is None:
            model_runnable_predicator_config = {}
        return {
            "output_dir": output_dir,
            "filter_path": filter_path,
            "filter_config": filter_config if filter_config is not None else {},
            "post_extract_process_path": post_extract_process_path,
            "post_extract_process_class_name": post_extract_process_class_name,
            "post_extract_process_config": post_extract_process_config,
            "data_input_predicator_filepath": data_input_predicator_filepath,
            "data_input_predicator_class_name": data_input_predicator_class_name,
            "data_input_predicator_config": data_input_predicator_config,
            "model_runnable_predicator_filepath": model_runnable_predicator_filepath,
            "model_runnable_predicator_class_name": model_runnable_predicator_class_name,
            "model_runnable_predicator_config": model_runnable_predicator_config,
            "model_path_prefix": model_path_prefix,
        }

    def __call__(self, rel_model_path):
        src_model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        module, inputs = get_torch_module_and_inputs(src_model_path)
        gm = parse_sole_graph_module(module, inputs)
        gm = self.rename_graph_variables(gm, inputs, src_model_path)
        model_name = os.path.basename(rel_model_path.rstrip(os.sep))
        new_rel_path = f"{model_name}_renamed"
        dst_model_path = os.path.realpath(
            os.path.join(self.config["output_dir"], new_rel_path)
        )
        Path(dst_model_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_model_path, dst_model_path, dirs_exist_ok=True)
        self._update_model_py_file(gm, dst_model_path)
        self._update_weight_meta_py_file(src_model_path, dst_model_path)
        self._update_input_meta_py_file(src_model_path, dst_model_path)
        self._try_run(dst_model_path)

    def _try_run(self, model_path):
        assert self.model_runnable_predicator(
            model_path
        ), f"{model_path} is not a runnable model"

    def _update_model_py_file(self, graph_module, model_path):
        py_code = apply_templates(graph_module.code)
        (Path(model_path) / "model.py").write_text(py_code)
        file_hash = get_sha256_hash(py_code)
        (Path(model_path) / "graph_hash.txt").write_text(file_hash)

    def _update_weight_meta_py_file(self, src_model_path, dst_model_path):
        old_name_to_new_name = self._get_original_name_to_new_name(
            src_model_path, dst_model_path
        )
        tensor_metas = TensorMeta.unserialize_from_py_file(
            os.path.join(src_model_path, "weight_meta.py"),
        )
        for weight_meta in tensor_metas:
            assert weight_meta.name in old_name_to_new_name
            if weight_meta.original_name is None:
                weight_meta.original_name = weight_meta.name
            weight_meta.name = old_name_to_new_name[weight_meta.name]
        py_code = "\n\n".join(
            [weight_meta.serialize_to_py_str() for weight_meta in tensor_metas]
        )
        (Path(dst_model_path) / "weight_meta.py").write_text(py_code)

    def _update_input_meta_py_file(self, src_model_path, dst_model_path):
        old_name_to_new_name = self._get_original_name_to_new_name(
            src_model_path, dst_model_path
        )
        tensor_metas = TensorMeta.unserialize_from_py_file(
            os.path.join(src_model_path, "input_meta.py"),
        )
        for input_meta in tensor_metas:
            assert input_meta.name in old_name_to_new_name
            if input_meta.original_name is None:
                input_meta.original_name = input_meta.name
            input_meta.name = old_name_to_new_name[input_meta.name]
        py_code = "\n\n".join(
            [input_meta.serialize_to_py_str() for input_meta in tensor_metas]
        )
        (Path(dst_model_path) / "input_meta.py").write_text(py_code)

    def _get_original_name_to_new_name(self, src_model_path, dst_model_path):
        src_model = self._get_model(src_model_path)
        dst_model = self._get_model(dst_model_path)
        old_name_and_new_name_pairs = zip(
            self._get_input_names_from_signature(src_model),
            self._get_input_names_from_signature(dst_model),
            strict=True,
        )
        return {
            old_name: new_name for old_name, new_name in old_name_and_new_name_pairs
        }

    def _get_model(self, model_path):
        py_module = load_module(os.path.join(model_path, "model.py"))
        GraphModule = getattr(py_module, "GraphModule")
        GraphModule.__graph_net_file_path__ = py_module.__graph_net_file_path__
        return GraphModule()

    def _get_input_names_from_signature(self, module):
        return inspect.signature(module.forward).parameters

    def rename_graph_variables(
        self, gm: torch.fx.GraphModule, sample_inputs, model_path
    ):
        counters = {"in": 0, "w": 0, "tmp": 0}
        # graph may not have input, only contain weights
        arg_iter = iter(sample_inputs) if sample_inputs else iter([])
        for node in gm.graph.nodes:
            self._process_single_node(node, arg_iter, counters, model_path)
        gm.graph.lint()
        gm.recompile()
        return gm

    def _process_single_node(self, node, arg_iter, counters, model_path):
        if "original_name" not in node.meta:
            node.meta["original_name"] = node.name
        if node.op == "placeholder":
            self._handle_placeholder(node, arg_iter, counters, model_path)
        elif node.op == "get_attr":
            self._apply_rename(node, "w", counters)
        elif node.op != "output":
            self._apply_rename(node, "tmp", counters)
        else:
            # Do nothing
            pass

    def _handle_placeholder(self, node, arg_iter, counters, model_path):
        real_arg = next(arg_iter, None)
        is_weight = self._is_weight_node(node, real_arg, model_path)
        prefix = "w" if is_weight else "in"
        self._apply_rename(node, prefix, counters, update_target=True)

    def _apply_rename(self, node, prefix, counters, update_target=False):
        new_name = f"{prefix}_{counters[prefix]}"
        counters[prefix] += 1
        node.name = new_name
        if update_target:
            node.target = new_name

    def _is_weight_node(self, node, real_arg, model_path):
        is_not_data_input = not self.data_input_predicator(model_path, node.name)
        is_parameter_type = (
            node.type is not None
            and isinstance(node.type, type)
            and issubclass(node.type, torch.nn.parameter.Parameter)
        )
        is_parameter_value = isinstance(real_arg, torch.nn.Parameter)
        return is_not_data_input or is_parameter_type or is_parameter_value
