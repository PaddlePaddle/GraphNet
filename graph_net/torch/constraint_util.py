import sys
import os
import graph_net
import logging

logger = logging.getLogger(__name__)


class NaiveDataInputPredicator:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path, input_var_name: str) -> bool:
        return not (
            "_self_" in input_var_name or "_instance_modules_" in input_var_name
        )


class RenamedDataInputPredicator:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path, input_var_name: str) -> bool:
        return not input_var_name.startswith("w_")


class ModelRunnablePredicator:
    def __init__(self, config):
        if config is None:
            config = {}

        decorator_config = {"use_dummy_inputs": True}
        self.predicator = RunModelPredicator(decorator_config)

    def __call__(self, model_path):
        return self.predicator(model_path)


class ShapePropagatablePredicator:
    def __init__(self, config=None):
        if config is None:
            config = {}

        graph_net_root = os.path.dirname(graph_net.__file__)
        decorator_config = {
            "decorator_path": f"{graph_net_root}/torch/shape_prop.py",
            "decorator_class_name": "ShapePropagate",
            "use_dummy_inputs": True,
        }
        self.predicator = RunModelPredicator(decorator_config)

    def __call__(self, model_path):
        return self.predicator(model_path)


class RunModelPredicator:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config

    def __call__(self, model_path):
        import json
        import base64

        json_string = json.dumps(self.config)
        json_bytes = json_string.encode("utf-8")
        b64_encoded_bytes = base64.b64encode(json_bytes)
        decorator_config = b64_encoded_bytes.decode("utf-8")
        cmd = f"{sys.executable} -m graph_net.torch.run_model --model-path {model_path} --decorator-config {decorator_config}"
        return os.system(cmd) == 0
