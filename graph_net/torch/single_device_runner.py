from graph_net.torch import utils
import argparse
import importlib.util
import inspect
import torch
import logging
from pathlib import Path
from typing import Type, Any
import sys
from graph_net.torch.extractor import extract
import hashlib
from contextlib import contextmanager
import json
import record_util
import imp_util
import os


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def _get_sha_hash(content):
    m = hashlib.sha256()
    m.update(content.encode())
    return m.hexdigest()


def _save_to_model_path(dump_dir, hash_text):
    file_path = f"{dump_dir}/graph_hash.txt"
    with open(file_path, "w") as f:
        f.write(hash_text)


@contextmanager
def _dump_graph_hash_key_ctx(cmd_args):
    if not cmd_args.dump_graph_hash_key:
        yield {}
        return
    mut_graph_codes = []
    extractor_kwarg = {
        "placeholder_auto_rename": True,
        "mut_graph_codes": mut_graph_codes,
    }
    yield extractor_kwarg
    if len(mut_graph_codes) > 0:
        assert len(mut_graph_codes) == 1, f"{len(mut_graph_codes)=}"
        _save_to_model_path(cmd_args.model_path, _get_sha_hash(mut_graph_codes[0]))


def main(args):
    with _dump_graph_hash_key_ctx(args) as dump_graph_options:
        model_path = args.model_path
        model_class = load_class_from_file(
            f"{model_path}/model.py", class_name="GraphModule"
        )
        assert model_class is not None
        model = model_class()
        print(f"{model_path=}")
        if args.enable_extract:
            assert args.extract_name is not None
            kwargs = dict(name=args.extract_name, dynamic=False, **dump_graph_options)
            model = extract(**kwargs)(model)

        inputs_params = utils.make_input_and_param_tensors_from_model_path(
            f"{model_path}"
        )
        params = inputs_params["weight_info"]
        shape_modifier = _get_shape_modifier(args)
        state_dict = {
            k: utils.replay_tensor(v, shape_modifier) for k, v in params.items()
        }

        explain = torch._dynamo.explain(model)(**state_dict)
        if explain.graph_count != 1 or len(explain.break_reasons) != 0:
            logging.error(
                f"Failed to generate a complete graph. The extraction process resulted in an incomplete graph, which was broken into {explain.graph_count} subgraphs."
            )
            logging.error(f"Reason(s): {explain.break_reasons}.")
            raise ValueError(
                f"Graph extraction failed. The resulting graph is incomplete, broken into {explain.graph_count} subgraphs."
            )

        model(**state_dict)


def _get_shape_modifier(cli_args):
    """
    yield shape modifier from shape_modifiers.json in directory cli_args.model_path
    """
    if not cli_args.enable_shape_patch:
        return lambda name, shape: shape
    shape_patch_file_path = f"{cli_args.model_path}/shape_patch.py"
    if not os.path.exists(shape_patch_file_path):
        return lambda name, shape: shape
    shape_modifier_data = [
        attrs
        for name, cls in imp_util.load_name_and_classes_from_file(shape_patch_file_path)
        for attrs in [record_util.make_attrs_from_class(cls)]
    ]
    assert isinstance(shape_modifier_data, list)
    return _make_shape_modifier_impl(shape_modifier_data)


def _make_shape_modifier_impl(shape_modifier_data):
    name2new_shape = {attrs["name"]: attrs["shape"] for attrs in shape_modifier_data}
    print(f"{name2new_shape=}")
    return lambda name, shape: name2new_shape[name] if name in name2new_shape else shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--dump-graph-hash-key",
        action="store_true",
        default=False,
        help="Dump graph hash key",
    )
    parser.add_argument(
        "--enable-extract",
        type=bool,
        required=False,
        default=False,
        help="Enable extract",
    )
    parser.add_argument(
        "--extract-name",
        type=str,
        required=False,
        default=None,
        help="Extracted graph's name",
    )
    parser.add_argument(
        "--enable-shape-patch",
        type=bool,
        required=False,
        default=False,
        help="Enable extra inputs",
    )
    args = parser.parse_args()
    main(args=args)
