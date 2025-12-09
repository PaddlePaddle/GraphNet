import traceback
import logging
from graph_net.torch.graph_decomposer import NaiveDecomposerExtractor
from graph_net.torch.graph_fusibility_status import (
    GraphFusibilityStatus,
    GraphFusibility,
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
