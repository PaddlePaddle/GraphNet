import os
import json

os.environ["ENABLE_CINN_IN_DY2ST"] = "0"
# os.environ["FLAGS_logging_trunc_pir_py_code"] = "1"
# os.environ["FLAGS_logging_pir_py_code_int_tensor_element_limit"] = "64"
os.environ["FLAGS_logging_pir_py_code_dir"] = "/tmp/dump"

import paddle
from athena.module_op_unittests_for_graphnet import GraphnetSample, generate_samples
from graph_net.paddle import utils


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

        self.subgraph_counter = 0
        self.dump_path = os.environ.get("GRAPH_NET_PIR_DUMP_WORKSPACE", "/tmp")
        self.workspace_path = (
            workspace_path
            if workspace_path is not None
            else os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        )
        if not self.workspace_path:
            raise EnvironmentError(
                "Environment variable 'GRAPH_NET_EXTRACT_WORKSPACE' is not set."
            )

    def prepare_to_extract(self, model_dump_path):
        os.makedirs(model_dump_path, exist_ok=True)
        old_flags = paddle.get_flags(
            [
                "FLAGS_logging_trunc_pir_py_code",
                "FLAGS_logging_pir_py_code_int_tensor_element_limit",
                "FLAGS_logging_pir_py_code_dir",
            ]
        )
        print(f"Set pir dumping path to {model_dump_path}")
        paddle.set_flags(
            {
                "FLAGS_logging_trunc_pir_py_code": 1,
                "FLAGS_logging_pir_py_code_int_tensor_element_limit": 64,
                "FLAGS_logging_pir_py_code_dir": model_dump_path,
            }
        )
        return old_flags

    def write_to_file(self, filepath, content):
        print(f"Write to {filepath}")
        with open(filepath, "w") as f:
            f.write(content)

    def __call__(self, **input_dict):
        # 1. Get model dump path
        model_dump_path = os.path.join(self.dump_path, self.name)
        old_flags = self.prepare_to_extract(model_dump_path)

        if self.input_spec is None:
            self.input_spec = [
                paddle.static.InputSpec(value.shape, value.dtype, name=name)
                for name, value in input_dict.items()
                if isinstance(value, paddle.Tensor)
            ]

        # 2. Run the model to dump pir programs
        static_model = paddle.jit.to_static(
            self.model, input_spec=self.input_spec, full_graph=True
        )
        static_model(**input_dict)

        # 3. Convert pir programs to graphnet samples
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

        graphnet_samples = generate_samples(
            model_name=self.name,
            ir_programs=ir_programs_path,
            example_inputs=example_inputs_path,
            eval_mode=True,
        )

        # 4. Save to model_path
        model_path = os.path.join(self.workspace_path, self.name)
        self.subgraph_counter = len(graphnet_samples)
        for i, sample in enumerate(graphnet_samples):
            subgraph_path = (
                model_path
                if self.subgraph_counter == 1
                else os.path.join(model_path, f"subgraph_{i}")
            )
            if not os.path.exists(subgraph_path):
                os.makedirs(subgraph_path, exist_ok=True)
            self.write_to_file(f"{subgraph_path}/model.py", sample.model)
            self.write_to_file(f"{subgraph_path}/weight_meta.py", sample.weight_meta)
            self.write_to_file(f"{subgraph_path}/input_meta.py", sample.input_meta)
            with open(os.path.join(subgraph_path, "graph_net.json"), "w") as f:
                json.dump(sample.metadata, f, indent=4)

        print(
            f"Graph and tensors for '{self.name}' extracted successfully to: {model_path}"
        )

        # 5. Restore the environment
        paddle.set_flags(old_flags)
        return static_model


def extract(name, dynamic=False, input_spec=None):
    def wrapper(model: paddle.nn.Layer):
        assert isinstance(model, paddle.nn.Layer), f"{type(model)=}"
        return GraphExtractor(model, name, dynamic, input_spec)

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
