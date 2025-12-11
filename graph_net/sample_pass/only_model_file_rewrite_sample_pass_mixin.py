import abc
import shutil
from graph_net.sample_pass.sample_pass_mixin import SamplePassMixin
from pathlib import Path


class OnlyModelFileRewriteSamplePassMixin(SamplePassMixin):
    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
    ):
        pass

    @abc.abstractmethod
    def handle_model_py_file(self, rel_model_path: str) -> str:
        """
        return rewrited model.py file contents
        """
        raise NotImplementedError()

    def copy_sample_and_handle_model_py_file(self, rel_model_path: str):
        src_model_path = Path(self.config["model_path_prefix"]) / rel_model_path
        dst_model_path = Path(self.config["output_dir"]) / rel_model_path
        dst_model_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_model_path, dst_model_path, dirs_exist_ok=True)
        model_py_code = self.handle_model_py_file(rel_model_path)
        (dst_model_path / "model.py").write_text(model_py_code)
