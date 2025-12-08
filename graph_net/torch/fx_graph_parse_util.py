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
    assert name[:2] == "L_" or name[:2] == "l_", f"{name=}"
    name = name[2:]
    if name[0] == "l":
        name = "L" + name[1:]
    for pattern, replacement in pattern2replacement.items():
        name = name.replace(pattern, replacement)
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

    for node in traced_module.graph.nodes:
        if node.op != "placeholder":
            continue
        node.target = _rename_placeholder(node.target, pattern2replacement)
        node.name = _rename_placeholder(node.name, pattern2replacement)

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

    def handle_underscore_suffix_difference():
        zip_filter_names = get_zip_filter_names()
        if not (len(zip_filter_names) > 0):
            return
        if not all((a == b or f"{a}_" == b) for _, a, b in zip_filter_names):
            return
        names = set(
            name_in_placeholder
            for _0, name_in_signature, name_in_placeholder in zip_filter_names
            if f"{name_in_signature}_" == name_in_placeholder
        )
        for node in traced_module.graph.nodes:
            if not (node.op == "placeholder"):
                continue
            if node.target not in names:
                continue
            node.target = node.target[:-1]
            node.name = node.name[:-1]
        traced_module.recompile()

    handle_underscore_suffix_difference()

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
