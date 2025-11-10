import re
import torch.fx


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
        # replace this line with modification code for task 116 (torch._C._linalg.linalg_vector_norm)
        # replace this line with modification code for task 117 (torch._C._linalg.linalg_norm)
        # replace this line with modification code for task 118 (torch._C._nn.softplus)
        # replace this line with modification code for task 119 (torch._C._nn.one_hot)
        # replace this line with modification code for task 121 (torch._C._set_grad_enabled)
        # replace this line with modification code for task 122 (torch._C._log_api_usage_once)
        # replace this line with modification code for task 123 (torch._C._nn.pad)
        # replace this line with modification code for task 125 (torch._C._nn.gelu)
        # replace this line with modification code for task 126 (torch._C._nn.scaled_dot_product_attention)
        # replace this line with modification code for task 127 (torch._C._nn.linear)
    ]
    for pattern, repl in replacements:
        code = re.sub(pattern, repl, code)
    return code
