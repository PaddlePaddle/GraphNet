import abc
import copy
import inspect


class SamplePass(abc.ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self._check_config_declaration_valid()
        self.config = self._make_config_by_config_declare(config)

    @abc.abstractmethod
    def declare_config(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, rel_model_path: str):
        raise NotImplementedError()

    def _recursively_check_mixin_declare_config(self, base_class):
        from graph_net.sample_pass.sample_pass_mixin import SamplePassMixin

        if issubclass(base_class, (SamplePass, SamplePassMixin)):
            check_is_base_signature(
                base_class=base_class,
                derived_class=type(self),
                method_name="declare_config",
            )
        for sub_class in base_class.__bases__:
            self._recursively_check_mixin_declare_config(sub_class)

    def _check_config_declaration_parameters(self):
        sig = inspect.signature(self.declare_config)
        for name, param in sig.parameters.items():
            assert param.annotation in {
                int,
                bool,
                float,
                str,
                list,
                dict,
            }, f"{name=} {param.annotation}"
            assert param.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_KEYWORD,
            }, f"{name=} {param.kind=}"

    def _check_config_declaration_valid(self):
        self._recursively_check_mixin_declare_config(type(self))
        self._check_config_declaration_parameters()

    def _make_config_by_config_declare(self, config):
        sig = inspect.signature(self.declare_config)
        mut_config = copy.deepcopy(config)
        for name, param in sig.parameters.items():
            self._complete_default(name, param, mut_config)
            class_name = type(self).__name__
            assert name in mut_config, f"{name=} {class_name=}"

        def get_extra_config_fields():
            return set(name for name, _ in mut_config.items()) - set(
                name for name, _ in sig.parameters.items()
            )

        no_varadic_keyword = all(
            param.kind != inspect.Parameter.VAR_KEYWORD
            for _, param in sig.parameters.items()
        )
        if no_varadic_keyword:
            no_extra_config_fields = all(
                name in sig.parameters for name, _ in mut_config.items()
            )
            assert no_extra_config_fields, f"{get_extra_config_fields()=}"
        return mut_config

    def _complete_default(self, name, param, mut_config):
        if param.default is inspect.Parameter.empty:
            return
        mut_config[name] = copy.deepcopy(param.default)


def check_is_base_signature(base_class, derived_class, method_name):
    base = getattr(base_class, method_name)
    derived = getattr(derived_class, method_name)
    base_parameters = inspect.signature(base).parameters
    derived_parameters = inspect.signature(derived).parameters
    assert len(derived_parameters) >= len(base_parameters)
    for name, param in base_parameters.items():
        assert name in base_parameters, f"{name=}"
        assert param == base_parameters[name]
