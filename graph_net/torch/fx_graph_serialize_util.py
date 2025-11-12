import re
import torch.fx


# def apply_ast_based_linear_replacement(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
#     """
#     Apply AST-based replacement of torch._C._nn.linear to torch.nn.functional.linear.
#
#     This function uses AST parsing and transformation to replace torch._C._nn.linear
#     calls with torch.nn.functional.linear in the GraphModule's code.
#
#     Note: This function is currently commented out as the replacement is now handled
#     by simple string replacement in serialize_graph_module_to_str.
#
#     Args:
#         gm: The GraphModule to modify.
#
#     Returns:
#         Modified GraphModule with torch._C._nn.linear replaced by torch.nn.functional.linear.
#     """
#     import ast
#     import torch
#     import types
#
#     # First recompile to generate code
#     gm.recompile()
#
#     # Use AST to modify the generated code, replacing torch._C._nn.linear with torch.nn.functional.linear
#     code_str = gm.code
#
#     # Parse AST
#     tree = ast.parse(code_str)
#
#     class LinearReplacer(ast.NodeTransformer):
#         def visit_Call(self, node):
#             # Check if it's a torch._C._nn.linear call
#             # Structure: torch._C._nn.linear(...)
#             filtered_nodes = [
#                 node
#                 for node in [node]
#                 if isinstance(node.func, ast.Attribute)
#                 if node.func.attr == "linear"
#                 if isinstance(node.func.value, ast.Attribute)
#                 if node.func.value.attr == "_nn"
#                 if isinstance(node.func.value.value, ast.Attribute)
#                 if node.func.value.value.attr == "_C"
#                 if isinstance(node.func.value.value.value, ast.Name)
#                 if node.func.value.value.value.id == "torch"
#             ]
#             if filtered_nodes:
#                 # Found torch._C._nn.linear, replace with torch.nn.functional.linear
#                 new_func = ast.Attribute(
#                     value=ast.Attribute(
#                         value=ast.Attribute(
#                             value=ast.Name(
#                                 id="torch",
#                                 ctx=ast.Load(),
#                             ),
#                             attr="nn",
#                             ctx=ast.Load(),
#                         ),
#                         attr="functional",
#                         ctx=ast.Load(),
#                     ),
#                     attr="linear",
#                     ctx=ast.Load(),
#                 )
#                 node.func = new_func
#             return self.generic_visit(node)
#
#     transformer = LinearReplacer()
#     modified_tree = transformer.visit(tree)
#     ast.fix_missing_locations(modified_tree)
#
#     # Convert the modified AST back to code string
#     new_code = ast.unparse(modified_tree)
#
#     # Recompile the modified code
#     # Need to import device, inf and other modules that may be used
#     namespace = {
#         "torch": torch,
#     }
#     # Try to import device (if used in code)
#     try:
#         from torch import device
#
#         namespace["device"] = device
#     except ImportError:
#         pass
#     # Try to import inf (if used in code)
#     try:
#         from torch import inf
#
#         namespace["inf"] = inf
#     except ImportError:
#         # If torch doesn't have inf, use math.inf
#         try:
#             import math
#
#             namespace["inf"] = math.inf
#         except:
#             pass
#
#     exec(compile(modified_tree, filename="<ast>", mode="exec"), namespace)
#
#     # Update GraphModule's forward method
#     forward_func = namespace.get("forward")
#     if forward_func:
#         gm.forward = types.MethodType(forward_func, gm)
#
#     # Use serialize_graph_module_to_str to get the serialized code
#     # This ensures the code is properly serialized with unstable API replacements
#     serialized_code = serialize_graph_module_to_str(gm)
#     gm._code = serialized_code
#
#     return gm


def serialize_graph_module_to_str(gm: torch.fx.GraphModule) -> str:
    """
    Serialize a GraphModule to a string representation, replacing unstable APIs
    with their stable counterparts.

    This function is used to normalize the code representation of GraphModule
    for consistency checks and code generation.

    Args:
        gm: The GraphModule to serialize.

    Returns:
        A string representation of the GraphModule code with unstable APIs
        replaced by stable ones.
    """
    code = gm.code
    # Replace torch._C._nn.avg_pool2d with torch.nn.functional.avg_pool2d
    replacements = [
        (r"torch\._C\._nn\.avg_pool2d\(", "torch.nn.functional.avg_pool2d("),
        (r"torch\._C\._fft\.fft_irfft\(", "torch.fft.irfft("),
        (r"torch\._C\._fft\.fft_rfft\(", "torch.fft.rfft("),
        (r"torch\._C\._fft\.fft_fftn\(", "torch.fft.fftn("),
        (r"torch\._C\._special\.special_logit\(", "torch.special.logit("),
        (r"torch\._C\._linalg\.linalg_vector_norm\(", "torch.linalg.vector_norm("),
        (r"torch\._C\._linalg\.linalg_norm\(", "torch.linalg.norm("),
        (r"torch\._C\._nn\.softplus\(", "torch.nn.functional.softplus("),
        (r"torch\._C\._nn\.one_hot\(", "torch.nn.functional.one_hot("),
        (r"torch\._C\._set_grad_enabled\(", "torch.set_grad_enabled("),
        (r"torch\._C\.set_grad_enabled\(", "torch.set_grad_enabled("),
        # replace this line with modification code for task 122 (torch._C._log_api_usage_once)
        (r"torch\._C\._nn\.pad\(", "torch.nn.functional.pad("),
        # replace this line with modification code for task 125 (torch._C._nn.gelu)
        # replace this line with modification code for task 126 (torch._C._nn.scaled_dot_product_attention)
        (r"torch\._C\._nn\.linear\(", "torch.nn.functional.linear("),
    ]
    for pattern, repl in replacements:
        code = re.sub(pattern, repl, code)
    return code
