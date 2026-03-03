import os
import torch
import torch.fx
from torch.fx.subgraph_rewriter import replace_pattern
from torch.fx.passes.infra.pass_manager import PassManager, PassResult
import random
import string
import inspect
import json
from collections import OrderedDict
from pathlib import Path
import importlib.util as imp
from .graph_compiler_backend import GraphCompilerBackend


class PassMgrBackend(GraphCompilerBackend):
    def __init__(self, config: dict):
        assert isinstance(config, dict)
        super().__init__(self._make_config(**config))
        self.pass_manager = self.make_pass_manager()

    def _make_config(
        self,
        input_pass_rule_dir: str,
        output_pass_rule_dir: str,
        output_pass_pattern_limit: int,
        output_pass_replacement_func_limit: int,
        **kwargs,
    ):
        sorted_input_pass_rule_names = self._get_sorted_input_pass_rule_names(
            input_pass_rule_dir, output_pass_rule_dir
        )
        sorted_output_pass_rule_names = self._get_sorted_output_pass_rule_names(
            output_pass_rule_dir
        )
        return {
            "input_pass_rule_dir": input_pass_rule_dir,
            "output_pass_rule_dir": output_pass_rule_dir,
            "output_pass_pattern_limit": output_pass_pattern_limit,
            "output_pass_replacement_func_limit": output_pass_replacement_func_limit,
            "sorted_input_pass_rule_names": sorted_input_pass_rule_names,
            "sorted_output_pass_rule_names": sorted_output_pass_rule_names,
        }

    def _get_sorted_output_pass_rule_names(self, output_pass_rule_dir):
        output_pass_file_path = (
            Path(output_pass_rule_dir) / "sorted_output_pass_rule_names.json"
        )
        if not output_pass_file_path.exists():
            return []
        with open(output_pass_file_path) as f:
            rule_names = json.load(f)
        assert isinstance(rule_names, list)
        return rule_names

    def _get_sorted_input_pass_rule_names(
        self, input_pass_rule_dir, output_pass_rule_dir
    ):
        input_pass_file_path = (
            Path(input_pass_rule_dir) / "sorted_input_pass_rule_names.json"
        )
        if input_pass_file_path.exists():
            with open(input_pass_file_path) as f:
                default_input_rule_names = json.load(f)
        else:
            default_input_rule_names = []
        assert isinstance(default_input_rule_names, list)
        customized_input_pass_file_path = (
            Path(output_pass_rule_dir) / "sorted_input_pass_rule_names.json"
        )
        if not customized_input_pass_file_path.exists():
            return default_input_rule_names
        with open(customized_input_pass_file_path) as f:
            customized_input_rule_names = json.load(f)
        assert set(default_input_rule_names) == set(customized_input_rule_names)
        return customized_input_rule_names

    def __call__(self, model):
        return torch.compile(model, backend=self.torch_compile_backend)

    def torch_compile_backend(self, gm: torch.fx.GraphModule, sample_inputs: list):
        pass_result = self.pass_manager(gm)
        return pass_result.graph_module

    def make_pass_manager(self):
        return PassManager(passes=self.get_passes())

    def get_passes(self):
        return [
            create_pass(pass_name=pass_name, pass_rule=pass_rule)
            for pass_name, pass_rule in self._get_named_pass_rules()
        ]

    def _get_named_pass_rules(self):
        name2output_pass_rules = OrderedDict(
            (Path(inspect.getfile(rule)).stem, rule)
            for rule in self._get_output_pass_rules()
        )
        name2input_pass_rules = OrderedDict(
            (Path(inspect.getfile(rule)).stem, rule)
            for rule in self._get_input_pass_rules()
        )
        for name in name2input_pass_rules.keys():
            if name not in name2output_pass_rules:
                continue
            name2input_pass_rules[name] = name2output_pass_rules[name]
            del name2output_pass_rules[name]
        return [*name2input_pass_rules.items(), *name2output_pass_rules.items()]

    def _get_input_pass_rules(self):
        input_pass_rule_dir = self.config["input_pass_rule_dir"]
        sorted_input_pass_rule_names = self.config["sorted_input_pass_rule_names"]
        return [
            self._find_rule(dir_path=input_pass_rule_dir, name=name)
            for name in sorted_input_pass_rule_names
        ]

    def _get_output_pass_rules(self):
        output_pass_rule_dir = self.config["output_pass_rule_dir"]
        sorted_output_pass_rule_names = self.config["sorted_output_pass_rule_names"]
        rules = [
            self._find_rule(dir_path=output_pass_rule_dir, name=name)
            for name in sorted_output_pass_rule_names
        ]
        rules = self._bound_by_replacement_func_limit(rules)
        rules = self._bound_by_pattern_limit(rules)
        return rules

    def _bound_by_replacement_func_limit(self, rules):
        allowed_replacement_funcs = self._get_allowed_replacement_funcs(rules)
        return [
            rule for rule in rules if rule.replacement_func in allowed_replacement_funcs
        ]

    def _get_allowed_replacement_funcs(self, rules):
        replacement_func_limit = self.config["output_pass_replacement_func_limit"]
        replacement_func2none = OrderedDict([])
        for rule in rules:
            replacement_func2none[rule.replacement_func] = None
        replacement_funcs = list(replacement_func2none.keys())
        if len(replacement_funcs) <= replacement_func_limit:
            return set(replacement_funcs)
        indices = random.sample(range(len(replacement_funcs)), replacement_func_limit)
        indices.sort()
        return set(replacement_funcs[index] for index in indices)

    def _bound_by_pattern_limit(self, rules):
        pattern_limit = self.config["output_pass_pattern_limit"]
        if len(rules) <= pattern_limit:
            return rules
        indices = random.sample(range(len(rules)), pattern_limit)
        indices.sort()
        return [rules[i] for i in indices]

    def _find_rule(self, dir_path, name):
        return load_py_module(os.path.join(dir_path, f"{name}.py"), name=name)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class PatternReplacementPass:
    def __init__(self, pass_rule):
        arg_names = list(inspect.signature(pass_rule.pattern).parameters.keys())

        @self.reset_func_arg_names(arg_names)
        def replacement(*args):
            return pass_rule.replacement_func()(*pass_rule.replacement_args(*args))

        self.pattern = pass_rule.pattern
        self.replacement = replacement

    @classmethod
    def reset_func_arg_names(cls, arg_names):
        # arg_names is a list like ['x', 'y', 'z']
        args_str = ", ".join(arg_names)

        func_name = "dynamic_func_" + "".join(
            random.choices(string.ascii_lowercase, k=5)
        )

        source = f"""
def {func_name}(f):
    def func({args_str}):
        return f({args_str})
    return func
    """
        namespace = {}
        exec(source, globals(), namespace)
        return namespace[func_name]

    def __call__(self, gm: torch.fx.GraphModule):
        # This performs the actual match-and-replace
        matches = replace_pattern(gm, self.pattern, self.replacement)

        # Determine if the graph actually changed
        modified = len(matches) > 0

        if modified:
            gm.recompile()
            print(f"Applied {len(matches)} replacements.")

        # Return the PassResult object
        return PassResult(gm, modified)


def create_pass(pass_name, pass_rule):
    gm_pass = PatternReplacementPass(pass_rule)

    def func(gm):
        return gm_pass(gm)

    func.__name__ = pass_name
    func.__qualname__ = pass_name
    return func


def load_py_module(path, name="unamed"):
    spec = imp.spec_from_file_location(name, path)
    module = imp.module_from_spec(spec)
    module.__file__ = path
    spec.loader.exec_module(module)
    return module
