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
    code = re.sub(
        r"torch\._C\._nn\.avg_pool2d\(",
        "torch.nn.functional.avg_pool2d(",
        code,
    )
    return code
