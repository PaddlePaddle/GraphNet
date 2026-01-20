import abc
from graph_net.declare_config_mixin import DeclareConfigMixin


class SamplePass(abc.ABC, DeclareConfigMixin):
    def __init__(self, config=None):
        from graph_net.sample_pass.sample_pass_mixin import SamplePassMixin

        self.init_config(
            config=config, constraint_base_classes=(SamplePass, SamplePassMixin)
        )
        super().__init__()

    @abc.abstractmethod
    def __call__(self, rel_model_path: str):
        raise NotImplementedError()
