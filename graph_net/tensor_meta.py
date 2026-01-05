import ast
import importlib.util as imp
import inspect
from dataclasses import dataclass
import math
import functools
from pathlib import Path


@dataclass
class TensorMeta:
    record_class_name: str
    name: str
    original_name: str | None
    shape: list[int]
    dtype: str
    device: str | None
    mean: float
    std: float
    data: list[int | float] | None
    max_val: int | None
    min_val: int | None

    @classmethod
    def unserialize_from_py_file(cls, file_path: str) -> list["TensorMeta"]:
        return [
            TensorMeta(
                record_class_name=attrs.get("record_class_name"),
                name=attrs.get("name"),
                original_name=attrs.get("original_name", None),
                shape=attrs.get("shape", []),
                dtype=attrs.get("dtype"),
                device=attrs.get("device", None),
                mean=attrs.get("mean", None),
                std=attrs.get("std", None),
                data=attrs.get("data", None),
                max_val=attrs.get("max_val", None),
                min_val=attrs.get("min_val", None),
            )
            for name, tensor_meta_cls in cls._get_classes(file_path)
            for attrs in [cls._convert_cls_to_attrs(tensor_meta_cls)]
        ]

    @classmethod
    def unserialize_from_py_file_order_preserved(cls, file_path) -> list["TensorMeta"]:
        return [
            TensorMeta(
                record_class_name=attrs.get("record_class_name"),
                name=attrs.get("name"),
                original_name=attrs.get("original_name", None),
                shape=attrs.get("shape", []),
                dtype=attrs.get("dtype"),
                device=attrs.get("device", None),
                mean=attrs.get("mean", None),
                std=attrs.get("std", None),
                data=attrs.get("data", None),
                max_val=attrs.get("max_val", None),
                min_val=attrs.get("min_val", None),
            )
            for name, tensor_meta_cls in cls._get_classes_order_preserved(file_path)
            for attrs in [cls._convert_cls_to_attrs(tensor_meta_cls)]
        ]

    @classmethod
    def _convert_cls_to_attrs(cls, tensor_meta_cls):
        attrs = {
            k: v
            for k, v in tensor_meta_cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        attrs["record_class_name"] = tensor_meta_cls.__name__
        return attrs

    @classmethod
    def _get_classes(cls, file_path, name="unnamed"):
        spec = imp.spec_from_file_location(name, file_path)
        unnamed = imp.module_from_spec(spec)
        spec.loader.exec_module(unnamed)
        yield from inspect.getmembers(unnamed, inspect.isclass)

    @classmethod
    def _get_classes_order_preserved(cls, file_path, name="unnamed"):
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)

        class_names = [
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        ]

        spec = imp.spec_from_file_location(name, file_path)
        unnamed = imp.module_from_spec(spec)
        spec.loader.exec_module(unnamed)
        yield from [(name, getattr(unnamed, name)) for name in class_names]

    def update_shape_safely(self, shape):
        self.shape = shape
        if self.data is None:
            return
        assert isinstance(self.data, (list, tuple))
        size = functools.reduce(lambda a, b: a * b, self.shape, 1)
        extended_tensor_data = list(self.data)
        while len(extended_tensor_data) < size:
            extended_tensor_data.extend(extended_tensor_data)
        self.data = extended_tensor_data[:size]

    @classmethod
    def save_tensor_metas(cls, file_path: str, tensor_metas: list):
        py_code = "\n\n".join(
            tensor_meta.serialize_to_py_str() for tensor_meta in tensor_metas
        )
        Path(file_path).write_text(py_code)

    def serialize_to_py_str(self) -> str:
        lines = [
            (f"class {self.record_class_name}:"),
            (f'\tname = "{self.name}"'),
            *(
                [f'\toriginal_name = "{self.original_name}"']
                if self.original_name is not None
                else []
            ),
            (f"\tshape = {self.shape}"),
            (f'\tdtype = "{self.dtype}"'),
            (f'\tdevice = "{self.device}"'),
            (f"\tmean = {self._get_limited_precision_float_str(self.mean)}"),
            (f"\tstd = {self._get_limited_precision_float_str(self.std)}"),
            *(
                [f"\tdata = {self._format_data(self.data)}"]
                if self.data is not None
                else []
            ),
            *([f"\tmax_val = {self.max_val}"] if self.max_val is not None else []),
            *([f"\tmin_val = {self.min_val}"] if self.min_val is not None else []),
            (""),
        ]
        py_str = "\n".join(lines)
        return py_str

    def _get_limited_precision_float_str(self, value):
        if not isinstance(value, float):
            return value
        if math.isnan(value) or math.isinf(value):
            return f'float("{value}")'
        return f"{value:.3f}"

    def _format_data(self, data):
        if data is None:
            return "None"
        elif isinstance(data, list):
            return "[{}]".format(
                ", ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in data)
            )
        else:
            return repr(data)


if __name__ == "__main__":
    tensor_meta_code = """
class Program_weight_tensor_meta_L_self_modules_conv1_parameters_weight_:
    name = "L_self_modules_conv1_parameters_weight_"
    shape = [64, 3, 7, 7]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = -0.001
    std = 0.235
    data = None

"""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", encoding="utf-8") as tmp:
        tmp.write(tensor_meta_code)
        tmp.flush()
        tensor_metas = TensorMeta.unserialize_from_py_file(tmp.name)
        print(tensor_metas)
        for tensor_meta in tensor_metas:
            print(tensor_meta.serialize_to_py_str())

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", encoding="utf-8") as tmp:
        tmp.write(tensor_meta.serialize_to_py_str())
        tmp.flush()
        tensor_metas = TensorMeta.unserialize_from_py_file(tmp.name)
        print(tensor_metas)
        for tensor_meta in tensor_metas:
            print(tensor_meta.serialize_to_py_str())
