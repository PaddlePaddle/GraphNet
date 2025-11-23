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
