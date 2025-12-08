from pathlib import Path
from graph_net.imp_util import load_module
import graph_net.graph_net_json_file_util as gn_json


class UpdateSymDimReifier:
    def __init__(self, config):
        self.config = self.make_config(**config)

    def make_config(
        self,
        model_path_prefix,
        reifier_factory_path,
        reifier_factory_class_name,
        reifier_factory_config=None,
        resume=True,
    ):
        if reifier_factory_config is None:
            reifier_factory_config = {}
        return {
            "reifier_factory_path": reifier_factory_path,
            "reifier_factory_class_name": reifier_factory_class_name,
            "reifier_factory_config": reifier_factory_config,
            "model_path_prefix": model_path_prefix,
            "resume": resume,
        }

    def __call__(self, model_path):
        model_path_obj = Path(self.config["model_path_prefix"]) / model_path
        model_path = str(model_path_obj)
        input_tensor_cstr_filepath = model_path_obj / "input_tensor_constraints.py"
        if not input_tensor_cstr_filepath.exists():
            return
        if self.config["resume"] and self._found_reified_dims(model_path):
            return
        reifier_factory_class = self._get_reifier_factory_class()
        reifier_factory_instance = reifier_factory_class(
            config=self.config["reifier_factory_config"], model_path=model_path
        )
        matched_reifier_name = reifier_factory_instance.get_matched_reifier_name()
        if matched_reifier_name is None:
            return
        assert isinstance(matched_reifier_name, str), f"{type(matched_reifier_name)=}"
        gn_json.update_json(
            model_path, gn_json.kSymbolicDimensionReifier, matched_reifier_name
        )

    def _get_reifier_factory_class(self):
        py_module = load_module(self.config["reifier_factory_path"])
        return getattr(py_module, self.config["reifier_factory_class_name"])

    def _found_reified_dims(self, model_path):
        json = gn_json.read_json(model_path)

        if gn_json.kSymbolicDimensionReifier not in json:
            return False

        return json[gn_json.kSymbolicDimensionReifier] is not None
