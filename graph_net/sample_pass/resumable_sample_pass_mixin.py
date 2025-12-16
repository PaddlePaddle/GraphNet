import abc
import sys
from graph_net.sample_pass.sample_pass_mixin import SamplePassMixin
from pathlib import Path
import os


class ResumableSamplePassMixin(SamplePassMixin):
    def __init__(self, *args, **kwargs):
        self.num_handled_models = 0

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def sample_handled(self, rel_model_path: str) -> bool:
        dst_model_path = Path(self.config["output_dir"]) / rel_model_path
        if not dst_model_path.exists():
            return False
        num_model_py_files = len(list(dst_model_path.rglob("model.py")))
        assert num_model_py_files <= 1
        return num_model_py_files == 1

    @abc.abstractmethod
    def resume(self, rel_model_path: str):
        raise NotImplementedError()

    def resumable_handle_sample(self, rel_model_path: str):
        assert os.path.realpath(self.config["model_path_prefix"]) != os.path.realpath(
            self.config["output_dir"]
        )
        if self.config["resume"] and self.sample_handled(rel_model_path):
            return
        self.resume(rel_model_path)
        self._inc_num_handled_models_or_exit()

    def _inc_num_handled_models_or_exit(self):
        if self.config["limits_handled_models"] is None:
            return
        self.num_handled_models += 1
        if self.num_handled_models >= self.config["limits_handled_models"]:
            print("limits_handled_models expired.", flush=True)
            sys.exit(0)
