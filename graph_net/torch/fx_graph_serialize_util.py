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
        # Add new rules to this list as needed
    ]
    for pattern, repl in replacements:
        code = re.sub(pattern, repl, code)
    return code
