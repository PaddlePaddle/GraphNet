from pathlib import Path
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
import os
import sys


class CheckGraphModuleParsable:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)
        self.num_handled_models = 0

    def _make_config(
        self,
        model_path_prefix,
        output_dir,
        resume=False,
        limits_handled_models=None,
    ):
        return {
            "model_path_prefix": model_path_prefix,
            "output_dir": output_dir,
            "resume": resume,
            "limits_handled_models": limits_handled_models,
        }

    def __call__(self, rel_model_path):
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        if self.config["resume"] and self._is_model_path_handled(rel_model_path):
            return
        parse_immutable_model_path_into_sole_graph_module(model_path)
        output_dir = Path(self.config["output_dir"]) / rel_model_path
        output_dir.mkdir(parents=True, exist_ok=True)
        self._inc_num_handled_models()

    def _is_model_path_handled(self, rel_model_path):
        return (Path(self.config["output_dir"]) / rel_model_path).exists()

    def _inc_num_handled_models(self):
        self.num_handled_models += 1
        limits = self.config["limits_handled_models"]
        if limits is not None:
            if self.num_handled_models >= limits:
                print(
                    "`num_handled_models` exceeds config `limits_handled_models`",
                    file=sys.stderr,
                )
                sys.exit(0)
