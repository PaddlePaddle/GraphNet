import os
import json
import importlib.util

import paddle
from athena.graphnet_samples import GraphnetSample, RunGeneration
from graph_net import imp_util
from graph_net.paddle import utils


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = imp_util.load_module(file_path, "unnamed")
    model_class = getattr(module, class_name, None)
    return model_class


def write_to_file(filepath, content):
    print(f"Write to {filepath}")
    with open(filepath, "w") as f:
        f.write(content)


def generate_model_wrapper_class(model_dump_path, data_arg_names):
    graph_module_wrapper_class_template = """
import paddle

class GraphModuleWrapper(paddle.nn.Layer):
    def __init__(self, graph_module):
        super().__init__()
        self.graph_module = graph_module

    def set_parameters(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, paddle.nn.parameter.Parameter):
                setattr(self, name, value)

    def forward(self, ${DATA_ARG_NAMES}):
        param_dict = { name: param for name, param in self.named_parameters() }
        outputs = self.graph_module(${DATA_ARG_VALUE_PAIRS}, **param_dict)
        return outputs
"""

    data_arg_value_pairs = [f"{name}={name}" for name in data_arg_names]
    graph_module_wrapper_class_code_str = graph_module_wrapper_class_template.replace(
        "${DATA_ARG_NAMES}", ", ".join(data_arg_names)
    ).replace("${DATA_ARG_VALUE_PAIRS}", ", ".join(data_arg_value_pairs))
    print(graph_module_wrapper_class_code_str)

    file_path = os.path.join(model_dump_path, "graph_module_wrapper.py")
    write_to_file(file_path, graph_module_wrapper_class_code_str)
    model_class = load_class_from_file(
        file_path=file_path, class_name="GraphModuleWrapper"
    )
    return model_class


# used as configuration of python -m graph_net.paddle.run_model
class RunModelDecorator:
    def __init__(self, config):
        self.config = self.make_config(**config)

    def __call__(self, model):
        return extract(**self.config)(model)

    def make_config(
        self,
        name=None,
        dynamic=False,
        input_spec=None,
        custom_extractor_path: str = None,
        custom_extractor_config: dict = None,
    ):
        assert name is not None
        return {
            "name": name,
            "dynamic": dynamic,
            "input_spec": input_spec,
            "extractor_config": {
                "custom_extractor_path": custom_extractor_path,
                "custom_extractor_config": custom_extractor_config,
            },
        }


class GraphExtractor:
    def __init__(
        self,
        model,
        name,
        dynamic=False,
        input_spec=None,
        workspace_path=None,
    ):
        self.model = model
        self.name = name
        self.dynamic = dynamic
        self.input_spec = input_spec
        assert not self.dynamic, "dynamic=True is not supported now!"

        self.num_subgraphs = 0
        self.num_samples_of_all_subgraphs = 0
        self.subgraph_idx2samples = None

        dump_path = os.environ.get("GRAPH_NET_PIR_DUMP_WORKSPACE", "/tmp")
        self.dump_path = os.path.abspath(dump_path)

        workspace_path = (
            workspace_path
            if workspace_path is not None
            else os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        )
        self.workspace_path = os.path.abspath(workspace_path)
        if not self.workspace_path:
            raise EnvironmentError(
                "Environment variable 'GRAPH_NET_EXTRACT_WORKSPACE' is not set."
            )

    def prepare_to_extract(self, model_dump_path):
        os.makedirs(model_dump_path, exist_ok=True)
        new_flags = {
            "FLAGS_logging_trunc_pir_py_code": 1,
            "FLAGS_logging_pir_py_code_int_tensor_element_limit": 64,
            "FLAGS_logging_pir_py_code_dir": model_dump_path,
        }
        old_flags = paddle.get_flags(list(new_flags.keys()))

        print(f"Set pir dumping path to {model_dump_path}")
        paddle.set_flags(new_flags)
        return old_flags

    def run_model_with_dump_enabled(self, model_dump_path, **input_dict):
        # Get model dump path
        old_flags = self.prepare_to_extract(model_dump_path)

        param_dict = {
            k: v
            for k, v in input_dict.items()
            if isinstance(v, paddle.nn.parameter.Parameter)
        }
        data_dict = {k: v for k, v in input_dict.items() if k not in param_dict}

        input_spec = self.input_spec
        if self.input_spec is None:
            input_spec = [
                paddle.static.InputSpec(value.shape, value.dtype, name=name)
                for name, value in data_dict.items()
                if isinstance(value, paddle.Tensor)
            ]
        else:
            assert len(input_spec) == len(data_dict)

        if param_dict:
            model_wrapper_class = generate_model_wrapper_class(
                model_dump_path, data_dict.keys()
            )
            wrapped_model = model_wrapper_class(self.model)
            wrapped_model.set_parameters(**param_dict)
        else:
            wrapped_model = self.model

        # Run the static model
        static_model = paddle.jit.to_static(
            wrapped_model,
            input_spec=input_spec,
            full_graph=True,
            backend=None,
        )
        static_model.eval()
        program = static_model.forward.concrete_program.main_program
        # print(program)
        static_model(**data_dict)

        # Restore the environment
        paddle.set_flags(old_flags)
        return static_model

    def translate_pir_program_to_sample_codes(
        self, model_dump_path, split_positions=None
    ):
        ir_programs_path = os.path.join(model_dump_path, "exec_programs.py")
        example_inputs_path = os.path.join(
            model_dump_path, "programs_example_input_tensor_meta.py"
        )
        assert os.path.isfile(
            ir_programs_path
        ), f"{ir_programs_path} is not a regular file."
        assert os.path.isfile(
            example_inputs_path
        ), f"{example_inputs_path} is not a regular file."

        # Arguments for graph decomposer
        op_example_inputs_path = (
            os.path.join(model_dump_path, "op_example_input_tensor_meta.py")
            if split_positions
            else None
        )
        all_samples = RunGeneration(
            model_name=self.name,
            ir_programs=ir_programs_path,
            example_inputs=example_inputs_path,
            op_example_inputs=op_example_inputs_path,
            split_positions=split_positions,
            eval_mode=True,
        )

        self.subgraph_idx2samples = {}
        for sample in all_samples:
            if sample.subgraph_idx not in self.subgraph_idx2samples.keys():
                self.subgraph_idx2samples[sample.subgraph_idx] = []
            self.subgraph_idx2samples[sample.subgraph_idx].append(sample)

        self.num_subgraphs = len(self.subgraph_idx2samples)
        self.num_samples_of_all_subgraphs = len(all_samples)
        assert self.num_subgraphs > 0
        return self.subgraph_idx2samples

    def write_sample_to_file(self, subgraph_path, sample):
        if not os.path.exists(subgraph_path):
            os.makedirs(subgraph_path, exist_ok=True)
        write_to_file(f"{subgraph_path}/model.py", sample.model)
        write_to_file(f"{subgraph_path}/weight_meta.py", sample.weight_meta)
        write_to_file(f"{subgraph_path}/input_meta.py", sample.input_meta)
        with open(os.path.join(subgraph_path, "graph_net.json"), "w") as f:
            json.dump(sample.metadata, f, indent=4)

    def __call__(self, **input_dict):
        # 1. Run the model to dump pir programs
        model_dump_path = os.path.join(self.dump_path, self.name)
        static_model = self.run_model_with_dump_enabled(model_dump_path, **input_dict)

        # 2. Convert pir programs to graphnet samples
        self.translate_pir_program_to_sample_codes(
            model_dump_path, split_positions=None
        )

        # 3. Save to model_path
        model_path = os.path.join(self.workspace_path, self.name)
        for subgraph_idx, samples in self.subgraph_idx2samples.items():
            assert len(samples) == 1
            if self.num_samples_of_all_subgraphs == 1:
                subgraph_path = model_path
            else:
                subgraph_path = os.path.join(model_path, f"subgraph_{subgraph_idx}")
            self.write_sample_to_file(subgraph_path, samples[0])

        print(
            f"Graph and tensors for '{self.name}' extracted successfully to: {model_path}"
        )
        return static_model


def extract(name, dynamic=False, input_spec=None, extractor_config: dict = None):
    """
    Extract computation graphs from PaddlePaddle nn.Layer.
    The extracted computation graphs will be saved into directory of env var $GRAPH_NET_EXTRACT_WORKSPACE.

    Args:
        name (str): The name of the model, used as the directory name for saving.
        dynamic (bool): Enable dynamic shape support in paddle.jit.to_static.
        input_spec (list[InputSpec] | tuple[InputSpec]): InputSpec for input tensors, which includes tensor's name, shape and dtype.
            When dynamic is False, input_spec can be inferred automatically.

    Returns:
        wrapper or decorator
    """

    extractor_config = make_extractor_config(extractor_config)

    def get_graph_extractor_maker():
        custom_extractor_path = extractor_config["custom_extractor_path"]
        custom_extractor_config = extractor_config["custom_extractor_config"]
        if custom_extractor_path is None:
            return GraphExtractor

        cls = load_class_from_file(custom_extractor_path, "GraphExtractor")
        return lambda *args, **kwargs: cls(custom_extractor_config, *args, **kwargs)

    def wrapper(model: paddle.nn.Layer):
        assert isinstance(model, paddle.nn.Layer), f"{type(model)=}"
        extractor = get_graph_extractor_maker()(model, name, dynamic, input_spec)
        return extractor

    def decorator(module_class):
        def constructor(*args, **kwargs):
            return wrapper(module_class(*args, **kwargs))

        return constructor

    def decorator_or_wrapper(obj):
        if isinstance(obj, paddle.nn.Layer):
            return wrapper(obj)
        elif issubclass(obj, paddle.nn.Layer):
            return decorator(obj)
        else:
            raise NotImplementedError(
                "Only paddle.nn.Layer instance or subclass supported."
            )

    return decorator_or_wrapper


def make_extractor_config(extractor_config):
    kwargs = extractor_config if extractor_config is not None else {}
    return make_extractor_config_impl(**kwargs)


def make_extractor_config_impl(
    custom_extractor_path: str = None, custom_extractor_config: dict = None
):
    config = custom_extractor_config if custom_extractor_config is not None else {}
    return {
        "custom_extractor_path": custom_extractor_path,
        "custom_extractor_config": config,
    }
