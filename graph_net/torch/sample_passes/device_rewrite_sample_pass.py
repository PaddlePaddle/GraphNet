from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.sample_pass.only_model_file_rewrite_sample_pass_mixin import (
    OnlyModelFileRewriteSamplePassMixin,
)
from graph_net.torch import utils
from pathlib import Path


class DeviceRewriteSamplePass(
    SamplePass, ResumableSamplePassMixin, OnlyModelFileRewriteSamplePassMixin
):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        device: str,
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def resume(self, rel_model_path: str):
        return self.copy_sample_and_handle_model_py_file(rel_model_path)

    def handle_model_py_file(self, rel_model_path: str) -> str:
        src_model_path = Path(self.config["model_path_prefix"]) / rel_model_path
        model_py_code = (src_model_path / "model.py").read_text()
        device = self.config["device"]
        return utils.modify_code_by_device(model_py_code, device)
