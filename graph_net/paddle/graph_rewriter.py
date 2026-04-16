import ast
import json


def _match_paddle_c_ops(node, op_name=None):
    """Match a call to paddle._C_ops.<op_name>. Return the op name or None."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    func = node.func
    if not (
        isinstance(func.value, ast.Attribute)
        and func.value.attr == "_C_ops"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "paddle"
    ):
        return None
    matched = func.attr
    if op_name is not None and matched != op_name:
        return None
    return matched


class GraphRewriter:
    name = ""
    description = ""

    # All registered rewriter subclasses, populated by __init_subclass__
    _registry = {}
    _last_stats = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            GraphRewriter._registry[cls.name] = cls

    def __init__(self):
        self.op_counts = {}

    def rewrite(self, model_code):
        print(f"[GraphRewriter] Running: {self.name} - {self.description}")
        tree = ast.parse(model_code)
        transformer = self._create_transformer()
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)
        self.op_counts = getattr(transformer, "op_counts", {})
        return ast.unparse(modified_tree)

    def _create_transformer(self):
        raise NotImplementedError

    @classmethod
    def get_registry(cls):
        return dict(cls._registry)

    @classmethod
    def available_names(cls):
        return list(cls._registry.keys())

    @staticmethod
    def run_pipeline(model_code, rewriter_names=None):
        if rewriter_names is None:
            rewriter_names = list(GraphRewriter._registry.keys())
        GraphRewriter._last_stats = {}
        modified_code = model_code
        for name in rewriter_names:
            rewriter_cls = GraphRewriter._registry.get(name)
            if rewriter_cls is None:
                raise ValueError(
                    f"Unknown rewriter '{name}'. "
                    f"Available: {GraphRewriter.available_names()}"
                )
            rewriter = rewriter_cls()
            modified_code = rewriter.rewrite(modified_code)
            GraphRewriter._last_stats[name] = rewriter.op_counts
        GraphRewriter.print_stats()
        return modified_code

    @staticmethod
    def print_stats():
        stats = GraphRewriter._last_stats
        total = sum(c for ops in stats.values() for c in ops.values())
        output = {"rewriters": stats, "total": total}
        print(f"[GraphRewriter] Stats: {json.dumps(output)}")


class ModeRewriter(GraphRewriter):
    name = "mode"
    description = "Rewrite dropout/batch_norm is_test arg to use self.training"

    POSITIONAL_IS_TEST_INDEX = {
        "dropout": 3,
        "batch_norm": 5,
    }

    def _create_transformer(self):
        target_ops = self.POSITIONAL_IS_TEST_INDEX

        class _IsTestTransformer(ast.NodeTransformer):
            def __init__(self):
                super().__init__()
                self.op_counts = {}

            def visit_Call(self, node: ast.Call):
                self.generic_visit(node)

                op_name = _match_paddle_c_ops(node)
                if op_name is None or op_name not in target_ops:
                    return node

                idx = target_ops[op_name]
                if idx >= len(node.args):
                    return node

                node.args[idx] = ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="training",
                        ctx=ast.Load(),
                    ),
                )
                self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
                return node

        return _IsTestTransformer()


class RandomnessRemovalRewriter(GraphRewriter):
    name = "randomness_removal"
    description = "Redirect paddle._C_ops.uniform to CPUPlace for deterministic results"

    def _create_transformer(self):
        class _RandomnessRemovalTransformer(ast.NodeTransformer):
            def __init__(self):
                super().__init__()
                self.op_counts = {}

            def _is_uniform_assign(self, node):
                if not isinstance(node.value, ast.Call):
                    return False
                if _match_paddle_c_ops(node.value, "uniform") is None:
                    return False
                return isinstance(node.targets[0], ast.Name)

            def _build_cpu_redirect(self, call, original_name):
                cpu_name = original_name + "_cpu"
                original_place = call.args[-1]

                call.args[-1] = ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="paddle", ctx=ast.Load()),
                            attr="core",
                            ctx=ast.Load(),
                        ),
                        attr="CPUPlace",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )

                assign_cpu = ast.Assign(
                    targets=[ast.Name(id=cpu_name, ctx=ast.Store())],
                    value=call,
                )
                assign_gpu = ast.Assign(
                    targets=[ast.Name(id=original_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=cpu_name, ctx=ast.Load()),
                            attr="_copy_to",
                            ctx=ast.Load(),
                        ),
                        args=[original_place, ast.Constant(value=False)],
                        keywords=[],
                    ),
                )
                return [assign_cpu, assign_gpu]

            def visit_Assign(self, node):
                if not self._is_uniform_assign(node):
                    return self.generic_visit(node)
                self.op_counts["uniform"] = self.op_counts.get("uniform", 0) + 1
                return self._build_cpu_redirect(node.value, node.targets[0].id)

        return _RandomnessRemovalTransformer()


class UseGpudnnRewriter(GraphRewriter):
    name = "use_gpudnn"
    description = "Insert _use_gpudnn(False) before avg_pool2d calls"

    def _create_transformer(self):
        class _UseGpudnnTransformer(ast.NodeTransformer):
            def __init__(self):
                super().__init__()
                self.op_counts = {}

            def _is_assign_of_avg_pool(self, node):
                if not isinstance(node.value, ast.Call):
                    return False
                if _match_paddle_c_ops(node.value, "pool2d") is None:
                    return False
                return any(
                    isinstance(arg, ast.Constant) and arg.value == "avg"
                    for arg in node.value.args
                )

            def _create_in_dynamic_mode_call(self):
                # Create paddle.framework.in_dynamic_mode() call node
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="paddle", ctx=ast.Load()),
                            attr="framework",
                            ctx=ast.Load(),
                        ),
                        attr="in_dynamic_mode",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )

            def visit_Assign(self, node):
                # match assign of paddle._C_ops.pool2d
                if not self._is_assign_of_avg_pool(node):
                    return self.generic_visit(node)

                call = node.value
                dynamic_mode_check = ast.If(
                    test=self._create_in_dynamic_mode_call(),
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=call.args[0],
                                    attr="_use_gpudnn",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Constant(value=False)],
                                keywords=[],
                            )
                        )
                    ],
                    orelse=[],
                )

                # insert dynamic_mode_check before pool2d
                self.op_counts["pool2d"] = self.op_counts.get("pool2d", 0) + 1
                return [dynamic_mode_check, node]

        return _UseGpudnnTransformer()
