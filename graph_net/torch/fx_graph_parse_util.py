import torch
import inspect


def _rename_placeholder(name):
    assert name[:2] == "L_" or name[:2] == "l_", f"{name=}"
    name = name[2:]
    if name[0] == "l":
        name = "L" + name[1:]
    name = name.replace(
        "modules_layer_norm_parameters",
        "modules_LayerNorm_parameters",
    )
    return name


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
        node.target = _rename_placeholder(node.target)
        node.name = _rename_placeholder(node.name)

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

    def get_zip_filter_names():
        names_from_signature = get_input_names_from_signature()
        names_from_placeholder = get_input_names_from_placeholder()
        return list(
            (i, name_from_signature, name_from_placeholder)
            for i, name_from_signature, name_from_placeholder in zip(
                range(len(names_from_signature)),
                names_from_signature,
                names_from_placeholder,
            )
            if name_from_signature != name_from_placeholder
        )

    if len(get_zip_filter_names()) > 0 and set(get_input_names_from_signature()) == set(
        get_input_names_from_placeholder()
    ):
        traced_module = _reorder_placeholders(
            traced_module, get_input_names_from_signature()
        )

    zip_filter_names = get_zip_filter_names()

    def zip_filter_names_str():
        for triple in zip_filter_names:
            print(triple)
        return "<printed before>"

    from pathlib import Path

    Path("/tmp/a.py").write_text(traced_module.code)
    assert len(zip_filter_names) == 0, f"{zip_filter_names_str()=}"
    return traced_module


def _reorder_placeholders(gm, sorted_names):
    sorted_names = list(sorted_names)
    name2placeholder = {
        node.name: node for node in gm.graph.nodes if node.op == "placeholder"
    }
    for i, current_placeholder_name in enumerate(sorted_names):
        if i == 0:
            continue
        prev_node = name2placeholder[sorted_names[i - 1]]
        current_node = name2placeholder[current_placeholder_name]
        with gm.graph.inserting_after(prev_node):
            new_node = gm.graph.placeholder(current_node.name)
            # force rename
            new_node.name = current_node.name
            new_node.target = current_node.target
            current_node.replace_all_uses_with(new_node)
            name2placeholder[current_placeholder_name] = new_node
            gm.graph.erase_node(current_node)

    gm.recompile()
    return gm
