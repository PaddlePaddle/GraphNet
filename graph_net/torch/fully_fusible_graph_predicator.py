import torch
import traceback
import logging
from graph_net.imp_util import load_module
from graph_net.torch.decompose_util import fold_range_to_submodule
from graph_net.torch.graph_decomposer import NaiveDecomposerExtractor
from graph_net.torch.graph_fusibility_status import (
    GraphFusibilityStatus,
    GraphFusibility,
)
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)

logger = logging.getLogger(__name__)


class FullyFusibleGraphPredicator:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        handler_config = self.config["handler_config"]
        self.decomposer_extractor = NaiveDecomposerExtractor(handler_config)

    def __call__(self, model_path):
        try:
            self.decomposer_extractor(model_path)
        except GraphFusibilityStatus as status:
            if status.graph_fusibility == GraphFusibility.kFullyFusible:
                return True
            elif status.graph_fusibility == GraphFusibility.kNotFullyFusible:
                return False
            else:
                raise NotImplementedError(f"{status.graph_fusibility=}")
        except Exception:
            print("\n--- Custom Error Handler ---")
            traceback.print_exc()
            print("--------------------------\n")
        return False


class FullyFusibleSubGraphPredicator:
    def __init__(self, config):
        if config is None:
            config = {}
        self.config = self._make_config(**config)
        self.nn_module_fully_fusible_decorator = (
            self._make_nn_module_fully_fusible_decorator(config)
        )
        model_path = self.config["model_path"]
        module, inputs = get_torch_module_and_inputs(model_path)
        self.traced_module = parse_immutable_model_path_into_sole_graph_module(
            model_path
        )
        self.inputs = inputs

    def _make_nn_module_fully_fusible_decorator(self, config):
        py_module = load_module(self.config["nn_module_fully_fusible_decorator_path"])
        decorator_cls = getattr(
            py_module, self.config["nn_module_fully_fusible_decorator_class_name"]
        )
        return decorator_cls(self.config["nn_module_fully_fusible_decorator_config"])

    def _make_config(
        self,
        model_path,
        nn_module_fully_fusible_decorator_path,
        nn_module_fully_fusible_decorator_class_name,
        nn_module_fully_fusible_decorator_config=None,
    ):
        if nn_module_fully_fusible_decorator_config is None:
            nn_module_fully_fusible_decorator_config = {}
        return {
            "model_path": model_path,
            "nn_module_fully_fusible_decorator_path": nn_module_fully_fusible_decorator_path,
            "nn_module_fully_fusible_decorator_class_name": nn_module_fully_fusible_decorator_class_name,
            "nn_module_fully_fusible_decorator_config": nn_module_fully_fusible_decorator_config,
        }

    def __call__(self, start_node_idx, end_node_idx):
        try:
            rewrited_gm: torch.fx.GraphModule = fold_range_to_submodule(
                self.traced_module,
                start_node_idx=start_node_idx,
                end_node_idx=end_node_idx,
                submodule_hook=self.nn_module_fully_fusible_decorator,
            )
            rewrited_gm(*self.inputs)
        except GraphFusibilityStatus as status:
            if status.graph_fusibility == GraphFusibility.kFullyFusible:
                return True
            elif status.graph_fusibility == GraphFusibility.kNotFullyFusible:
                return False
            else:
                raise NotImplementedError(f"{status.graph_fusibility=}")
        except Exception:
            print("\n--- Custom Error Handler ---")
            traceback.print_exc()
            print("--------------------------\n")
        return False
