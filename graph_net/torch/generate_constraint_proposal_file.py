from graph_net.torch import utils
import argparse
import torch
import logging
from pathlib import Path
from typing import Type, Any
import sys
from graph_net.torch.imp_util import load_class_from_file
import hashlib
from contextlib import contextmanager
import json
import inspect
import imp_util
import record_util
import copy


def main(args):
    model_path = args.model_path
    name2input_param_attrs = _get_name2input_param_attrs(model_path)
    name_and_annotation_types = _get_name_and_annotation_types(model_path)
    input_name_and_meta_attrs = _get_input_name_and_meta_attrs(
        name2input_param_attrs, name_and_annotation_types
    )
    input_name_and_constraint_attrs = _get_input_name_and_constraint_attrs(
        input_name_and_meta_attrs
    )
    _dump_input_name_and_constraint_attrs(
        input_name_and_constraint_attrs, args.output_path
    )


def _dump_input_name_and_constraint_attrs(input_name_and_constraint_attrs, output_path):
    py_code = record_util.serialize_to_py_code(
        [attr for _, attr in input_name_and_constraint_attrs],
        class_prefix="ProgramInputConstraint",
    )
    print(f"{output_path=}")
    with open(output_path, "w") as f:
        f.write(py_code)


def _get_input_name_and_constraint_attrs(input_name_and_meta_attrs):
    seq_no = 0
    dim2seq = {}

    def find_or_new_seq(dim):
        nonlocal seq_no
        nonlocal dim2seq
        if dim in dim2seq:
            return dim2seq[dim]
        ret = seq_no
        dim2seq[dim] = ret
        seq_no += 1
        return ret

    def make_symoblic_shape(shape):
        return type(shape)(
            [
                symbolic_dim_desc
                for dim in shape
                for dim_seq_no in [find_or_new_seq(dim)]
                for symbolic_dim_desc in [
                    {"symbol_name": f"s{dim_seq_no}", "example_value": dim}
                ]
            ]
        )

    def make_constraint_attrs(attrs):
        attrs = copy.deepcopy(attrs)
        attrs["shape"] = make_symoblic_shape(attrs["shape"])
        return attrs

    return [
        (name, symbolic_attrs)
        for name, attrs in input_name_and_meta_attrs
        for symbolic_attrs in [make_constraint_attrs(attrs)]
    ]


def _get_input_name_and_meta_attrs(name2input_param_attrs, name_and_annotation_types):
    def constructed_from_self(name):
        return name.find("self_") != -1

    def is_tensor_type(annotation_type):
        return annotation_type is torch.Tensor

    ret = [
        (name, meta_attr)
        for name, annotation_type in name_and_annotation_types
        if is_tensor_type(annotation_type)
        if not constructed_from_self(name)
        for meta_attr in [name2input_param_attrs[name]]
    ]
    assert len(ret) > 0
    return ret


def _get_name_and_annotation_types(model_path):
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    annotations = inspect.getfullargspec(model_class.forward).annotations
    return [(k, v) for k, v in annotations.items()]


def _get_name2input_param_attrs(model_path):
    def get_classes():
        input_meta_file = f"{model_path}/input_meta.py"
        for _, cls in imp_util.load_name_and_classes_from_file(input_meta_file):
            yield cls

        weight_meta_file = f"{model_path}/weight_meta.py"
        for _, cls in imp_util.load_name_and_classes_from_file(weight_meta_file):
            yield cls

    return {
        name: attr
        for cls in get_classes()
        for attr in [record_util.make_attrs_from_class(cls)]
        for name in [attr["name"]]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate constraint proposal file")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output file path",
    )
    args = parser.parse_args()
    main(args=args)
