import torch


def parse_sole_graph_module(module, inputs):
    traced_module = None
    traced_sample_inputs = None

    def my_backend(gm, sample_inputs):
        nonlocal traced_module
        traced_module = gm
        nonlocal traced_sample_inputs
        traced_sample_inputs = sample_inputs
        return gm.forward

    torch.compile(module, backend=my_backend)(*inputs)
    assert traced_module is not None
    assert all(id(a) == id(b) for a, b in zip(inputs, traced_sample_inputs))
    for node in traced_module.graph.nodes:
        if node.op != "placeholder":
            continue
        assert node.target[:2] == "L_" or node.target[:2] == "l_", f"{node.target=}"
        node.target = node.target[2:]
        assert node.name[:2] == "L_" or node.name[:2] == "l_", f"{node.name=}"
        node.name = node.name[2:]
    return traced_module
