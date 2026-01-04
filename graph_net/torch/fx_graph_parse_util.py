import torch
import inspect


class NamePatternMismatchDetector:
    def __init__(self, names_from_signature, names_from_placeholder):
        self.names_from_signature = names_from_signature
        self.names_from_placeholder = names_from_placeholder

    def __call__(self):
        mut_pattern2replacement = {}
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_LayerNorm",
            pattern_in_placeholder="modules_layer_norm",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_layer_norm",
            pattern_in_placeholder="modules_LayerNorm",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_meta4D_layers",
            pattern_in_placeholder="modules_meta4d_layers",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_meta4d_layers",
            pattern_in_placeholder="modules_meta4D_layers",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_SelfAttention_modules",
            pattern_in_placeholder="modules_self_attention_modules",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_self_attention_modules",
            pattern_in_placeholder="modules_SelfAttention_modules",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_meta3D_layers",
            pattern_in_placeholder="modules_meta3d_layers",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_meta3d_layers",
            pattern_in_placeholder="modules_meta3D_layers",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_DenseReluDense_modules",
            pattern_in_placeholder="modules_dense_relu_dense_modules",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_dense_relu_dense_modules",
            pattern_in_placeholder="modules_DenseReluDense_modules",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_EncDecAttention_modules",
            pattern_in_placeholder="modules_enc_dec_attention_modules",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_HashBucketCodepointEmbedder",
            pattern_in_placeholder="modules_hash_bucket_codepoint_embedder",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="modules_MBconv",
            pattern_in_placeholder="modules_mbconv",
        )
        self._detect_and_collect(
            mut_pattern2replacement,
            pattern_in_signature="_L_",
            pattern_in_placeholder="_l_",
        )
        return mut_pattern2replacement

    def _detect_and_collect(
        self, mut_pattern2replacement, pattern_in_signature, pattern_in_placeholder
    ):
        if not self._detect(pattern_in_signature, pattern_in_placeholder):
            return
        mut_pattern2replacement[pattern_in_placeholder] = pattern_in_signature

    def _detect(self, pattern_in_signature, pattern_in_placeholder):
        return self._check_pattern(
            self.names_from_signature,
            include_pattern=pattern_in_signature,
            exclude_pattern=pattern_in_placeholder,
        ) and self._check_pattern(
            self.names_from_placeholder,
            include_pattern=pattern_in_placeholder,
            exclude_pattern=pattern_in_signature,
        )

    def _check_pattern(self, names, include_pattern, exclude_pattern):
        return any(include_pattern in name for name in names) and all(
            exclude_pattern not in name for name in names
        )


def _get_name_pattern2replacement(names_from_signature, names_from_placeholder):
    dectector = NamePatternMismatchDetector(
        names_from_signature, names_from_placeholder
    )
    return dectector()


def _rename_placeholder(name, pattern2replacement):
    if not (name[:2] == "L_" or name[:2] == "l_"):
        return name
    name = name[2:]
    if name[:2] == "l_":
        name = "L_" + name[2:]
    for pattern, replacement in pattern2replacement.items():
        name = name.replace(pattern, replacement)
    return name


def parse_sole_graph_module_without_varify(module, inputs):
    traced_module = None
    traced_sample_inputs = None

    def my_backend(gm, sample_inputs):
        nonlocal traced_module
        nonlocal traced_sample_inputs
        traced_module = gm
        traced_sample_inputs = sample_inputs
        return gm.forward

    torch.compile(module, backend=my_backend)(*inputs)
    assert traced_module is not None
    return traced_module, traced_sample_inputs


def parse_sole_graph_module(module, inputs):
    traced_module, traced_sample_inputs = parse_sole_graph_module_without_varify(
        module, inputs
    )

    def get_input_names_from_signature():
        return inspect.signature(module.forward).parameters

    def get_input_names_from_placeholder():
        return [
            node.name for node in traced_module.graph.nodes if node.op == "placeholder"
        ]

    pattern2replacement = _get_name_pattern2replacement(
        names_from_signature=get_input_names_from_signature(),
        names_from_placeholder=get_input_names_from_placeholder(),
    )

    def handle_placeholder_name(pattern2replacement):
        for node in traced_module.graph.nodes:
            if node.op != "placeholder":
                continue
            node.target = _rename_placeholder(node.target, pattern2replacement)
            node.name = node.target

    handle_placeholder_name(pattern2replacement)

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

    def handle_underscore_suffix_difference():
        ph_nodes = {
            node.name: node
            for node in traced_module.graph.nodes
            if node.op == "placeholder"
        }
        sig_names = get_input_names_from_signature()
        sig_names_set = set(sig_names)
        for name in sig_names:
            target_ph_name = f"{name}_"
            if name in ph_nodes or target_ph_name not in ph_nodes:
                continue
            if target_ph_name in sig_names_set:
                continue
            node = ph_nodes[target_ph_name]
            node.target = node.name = name
        traced_module.recompile()

    handle_underscore_suffix_difference()

    def get_diff_input_names():
        placeholder_names = set(get_input_names_from_placeholder())
        return [
            (i, name)
            for i, name in enumerate(get_input_names_from_signature())
            if name not in placeholder_names
        ]

    if len(inputs) > len(traced_sample_inputs):
        diff_input_names = get_diff_input_names()
        first_node = next(iter(traced_module.graph.nodes))
        for _, name in diff_input_names:
            if name.startswith("l_"):
                name = "L_" + name[2:]
            with traced_module.graph.inserting_before(first_node):
                new_node = traced_module.graph.placeholder(name)
                new_node.name = name
                new_node.target = name
        traced_module.recompile()

    if len(get_zip_filter_names()) > 0 and set(get_input_names_from_signature()) == set(
        get_input_names_from_placeholder()
    ):
        traced_module = _reorder_placeholders(
            traced_module, get_input_names_from_signature()
        )

    zip_filter_names = get_zip_filter_names()

    def get_error_model_path():
        for triple in zip_filter_names:
            print(triple)
        return module.__graph_net_file_path__

    # from pathlib import Path
    # Path("/tmp/a.py").write_text(traced_module.code)
    assert len(zip_filter_names) == 0, f"{get_error_model_path()=}"
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
