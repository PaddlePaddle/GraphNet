from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from pathlib import Path
import os
import shutil
import tempfile
import ast
import inspect
import torch
from graph_net.imp_util import load_module
from graph_net.tensor_meta import TensorMeta
from graph_net.hash_util import get_sha256_hash


class AstGraphVariableRenamer(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config):
        super().__init__(config)
        self.data_input_predicator = self._make_data_input_predicator(self.config)
        self.model_runnable_predicator = self._make_model_runnable_predicator(
            self.config
        )

    def _make_data_input_predicator(self, config):
        module = load_module(config["data_input_predicator_filepath"])
        cls = getattr(module, config["data_input_predicator_class_name"])
        return cls(config["data_input_predicator_config"])

    def _make_model_runnable_predicator(self, config):
        module = load_module(config["model_runnable_predicator_filepath"])
        cls = getattr(module, config["model_runnable_predicator_class_name"])
        return cls(config["model_runnable_predicator_config"])

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        device: str,
        resume: bool = False,
        limits_handled_models: int = None,
        data_input_predicator_filepath: str = None,
        data_input_predicator_class_name: str = None,
        data_input_predicator_config: dict = None,
        model_runnable_predicator_filepath: str = None,
        model_runnable_predicator_class_name: str = None,
        model_runnable_predicator_config: dict = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        return self.naive_sample_handled(rel_model_path, search_file_name="model.py")

    def resume(self, rel_model_path: str):
        torch.cuda.empty_cache()
        dst_model_path = os.path.realpath(
            os.path.join(self.config["output_dir"], rel_model_path)
        )
        src_model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        graph_module_class = load_class_from_file(
            os.path.join(src_model_path, "model.py"), class_name="GraphModule"
        )
        input_arg_names, weight_arg_names = self._get_input_and_weight_arg_names(
            graph_module_class, src_model_path
        )
        rename_map = self._create_rename_map(input_arg_names, weight_arg_names)
        with tempfile.TemporaryDirectory(prefix="graph_variable_renamer_") as temp_dir:
            temp_model_path = os.path.join(temp_dir, os.path.basename(dst_model_path))
            shutil.copytree(src_model_path, temp_model_path, dirs_exist_ok=True)
            self._update_model_py_file(
                temp_model_path, rename_map, input_arg_names, weight_arg_names
            )
            self._update_meta_file(temp_model_path, "weight_meta.py", rename_map)
            self._update_meta_file(temp_model_path, "input_meta.py", rename_map)
            self._try_run(temp_model_path)
            shutil.copytree(temp_model_path, dst_model_path, dirs_exist_ok=True)

    def _get_input_and_weight_arg_names(self, graph_module, model_path):
        input_arg_names = []
        weight_arg_names = []
        sig = inspect.signature(graph_module.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            is_not_data_input = not self.data_input_predicator(model_path, name)
            if is_not_data_input:
                weight_arg_names.append(name)
            else:
                input_arg_names.append(name)
        return input_arg_names, weight_arg_names

    def _create_rename_map(self, input_arg_names, weight_arg_names):
        rename_map = {}
        for idx, name in enumerate(input_arg_names):
            rename_map[name] = f"in_{idx}"
        for idx, name in enumerate(weight_arg_names):
            rename_map[name] = f"w_{idx}"
        return rename_map

    def _update_model_py_file(
        self, model_path, rename_map, input_arg_names, weight_arg_names
    ):
        model_file = Path(model_path) / "model.py"
        source = model_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = self._get_graph_module_ast(tree)
        graph_renamer = AstGraphRenamer(rename_map, input_arg_names, weight_arg_names)
        graph_renamer.visit(node)
        py_code = ast.unparse(tree)
        model_file.write_text(py_code, encoding="utf-8")
        file_hash = get_sha256_hash(py_code)
        (Path(model_path) / "graph_hash.txt").write_text(file_hash)

    def _get_graph_module_ast(self, tree):
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "GraphModule":
                return node
        return None

    def _update_meta_file(self, model_path, meta_filename, rename_map):
        meta_file = Path(model_path) / meta_filename
        tensor_metas = TensorMeta.unserialize_from_py_file(str(meta_file))
        for meta in tensor_metas:
            assert (
                meta.name in rename_map
            ), f"[Warning] {meta.name} in {meta_filename} not found in rename_map."
            if meta.original_name is None:
                meta.original_name = meta.name
            meta.name = rename_map[meta.name]

        py_code = "\n\n".join([meta.serialize_to_py_str() for meta in tensor_metas])
        meta_file.write_text(py_code)

    def _try_run(self, model_path):
        (f"[AstGraphVariableRenamer] Try to run {model_path}")
        assert self.model_runnable_predicator(
            model_path
        ), f"{model_path} is not a runnable model"


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = load_module(file_path, "unnamed_graph_module")
    model_class = getattr(module, class_name, None)
    return model_class


class AstGraphRenamer(ast.NodeTransformer):
    def __init__(self, rename_map, input_arg_names, weight_arg_names):
        self.rename_map = rename_map
        self.input_and_weight_arg_names = set(input_arg_names) | set(weight_arg_names)
        self.counters = {"tmp": 0}
        self.in_forward = False

    def visit_FunctionDef(self, node):
        if node.name != "forward":
            return node
        self.in_forward = True
        node.args.args = self._rename_function_args(node.args.args)
        node.body = self._rename_function_body(node.body)
        self.in_forward = False
        return node

    def _rename_function_args(self, args):
        new_function_args = []
        for arg in args:
            if arg.arg == "self":
                new_function_args.append(arg)
            else:
                new_function_args.append(self._create_renamed_arg(arg))
        return new_function_args

    def _create_renamed_arg(self, arg):
        if arg.arg in self.rename_map:
            return ast.arg(arg=self.rename_map[arg.arg], annotation=arg.annotation)
        return arg

    def _rename_function_body(self, body):
        new_function_body = []
        for stmt in body:
            stmt = self._remove_clear_stmt_of_args(stmt)
            if stmt:
                stmt = self.visit(stmt)
                new_function_body.append(stmt)
        return new_function_body

    def _remove_clear_stmt_of_args(self, stmt):
        # remove stmt like w_0 = None
        if self._is_assign_none(stmt):
            return self._clean_assign_none(stmt)
        # remove stmt like del w_0
        elif isinstance(stmt, ast.Delete):
            return self._clean_delete(stmt)
        else:
            pass
        return stmt

    def _is_assign_none(self, stmt):
        return (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None
        )

    def _clean_assign_none(self, stmt):
        new_targets = [t for t in stmt.targets if not self._is_input_or_weight_var(t)]
        if not new_targets:
            return None
        stmt.targets = new_targets
        return stmt

    def _is_input_or_weight_var(self, target):
        return (
            isinstance(target, ast.Name)
            and target.id in self.input_and_weight_arg_names
        )

    def _clean_delete(self, stmt):
        new_targets = []
        for target in stmt.targets:
            kept = self._filter_delete_target(target)
            if kept:
                new_targets.append(kept)

        if not new_targets:
            return None
        stmt.targets = new_targets
        return stmt

    def _filter_delete_target(self, target):
        if isinstance(target, ast.Tuple):  # del (a, b)
            kept_elts = [e for e in target.elts if not self._is_protected_var(e)]
            return ast.Tuple(elts=kept_elts, ctx=ast.Del()) if kept_elts else None
        elif not self._is_protected_var(target):  # del a
            return target
        else:
            pass
        return None

    def visit_Assign(self, node):
        if not self.in_forward:
            return node
        self._register_new_local_variables(node.targets)
        self.generic_visit(node)
        return node

    def _register_new_local_variables(self, targets):
        for target in targets:
            for name in self._flatten_assignment_target(target):
                self._register_if_unknown(name)

    def _flatten_assignment_target(self, target):
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from self._flatten_assignment_target(elt)
        else:
            pass

    def _register_if_unknown(self, name):
        if name not in self.rename_map:
            new_name = f"tmp_{self.counters['tmp']}"
            self.counters["tmp"] += 1
            self.rename_map[name] = new_name

    def visit_Name(self, node):
        if not self.in_forward:
            return node
        if node.id in self.rename_map:
            return ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
        return node
