import sys
import os


class NaiveDataInputPredicator:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path, input_var_name: str) -> bool:
        return not ("_self_" in input_var_name)


class ModelRunnablePredicator:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path):
        cmd = f"{sys.executable} -m graph_net.torch.run_model --model-path {model_path}"
        return os.system(cmd) == 0


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
