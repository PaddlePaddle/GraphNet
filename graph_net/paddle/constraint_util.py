import logging

logger = logging.getLogger(__name__)


class NaiveDataInputPredicator:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path, input_var_name: str) -> bool:
        return not input_var_name.startswith("parameter_")
