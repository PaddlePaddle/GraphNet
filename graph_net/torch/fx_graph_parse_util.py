import torch
import inspect


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
    for node in traced_module.graph.nodes:
        if node.op != "placeholder":
            continue
        assert node.target[:2] == "L_" or node.target[:2] == "l_", f"{node.target=}"
        node.target = node.target[2:]
        if node.target[0] == "l":
            node.target = "L" + node.target[1:]
        assert node.name[:2] == "L_" or node.name[:2] == "l_", f"{node.name=}"
        node.name = node.name[2:]
        if node.name[0] == "l":
            node.name = "L" + node.name[1:]

    def get_input_names_from_signature():
        return inspect.signature(module.forward).parameters

    def get_input_names_from_placeholder():
        return [
            node.name for node in traced_module.graph.nodes if node.op == "placeholder"
        ]

    def get_diff_input_names():
        placeholder_names = set(get_input_names_from_placeholder())
        return [
            (i, name)
            for i, name in enumerate(get_input_names_from_signature())
            if name not in placeholder_names
        ]

    if len(inputs) == len(traced_sample_inputs) + 1:
        diff_input_names = get_diff_input_names()
        assert len(diff_input_names) == 1, f"{diff_input_names=}"
        pos, name = diff_input_names[0]
        for i, node in enumerate(traced_module.graph.nodes):
            if i < pos:
                assert node.op == "placeholder"
            elif i == pos:
                with traced_module.graph.inserting_before(node):
                    traced_module.graph.placeholder(name)
            else:
                break
        traced_module.recompile()
    assert len(get_diff_input_names()) == 0, f"{get_diff_input_names()=}"
    return traced_module
