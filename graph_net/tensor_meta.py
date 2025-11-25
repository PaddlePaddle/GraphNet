import importlib.util as imp
import inspect
from dataclasses import dataclass


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
    def unserialize_from_py_file(cls, file_path) -> list["TensorMeta"]:
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
    def _convert_cls_to_attrs(cls, tensor_meta_cls):
        attrs = {
            k: v
            for k, v in tensor_meta_cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        attrs["record_class_name"] = tensor_meta_cls.__name__
        return attrs

    @classmethod
    def _get_classes(cls, file_path, name="unamed"):
        spec = imp.spec_from_file_location("unnamed", file_path)
        unnamed = imp.module_from_spec(spec)
        spec.loader.exec_module(unnamed)
        yield from inspect.getmembers(unnamed, inspect.isclass)

    def serialize_to_py_str(self) -> str:
        lines = [
            (f"class {self.record_class_name}:"),
            (f'\tname = "{self.name}"'),
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
