import os
from graph_net import path_utils
from graph_net.paddle import utils


class GraphMetaRestorer:
    def __init__(self, config, parent_model_path):
        self.config = config
        self.parent_model_path = parent_model_path
        print(f"parent_model_path: {self.parent_model_path}")

        assert path_utils.is_single_model_dir(
            parent_model_path
        ), f"{parent_model_path=} is not a graphnet sample."
        (
            parent_weight_meta_classes,
            parent_input_meta_classes,
        ) = self._load_weight_and_input_meta_classes(parent_model_path)
        self.original_name2parent_weight_meta_class = self._convert_to_dict(
            parent_weight_meta_classes
        )
        self.original_name2parent_input_meta_class = self._convert_to_dict(
            parent_input_meta_classes
        )

    def __call__(self, model_path):
        assert path_utils.is_single_model_dir(
            model_path
        ), f"{model_path=} is not a graphnet sample."
        (
            weight_meta_classes,
            input_meta_classes,
        ) = self._load_weight_and_input_meta_classes(model_path)

        assert self.config["update_inplace"]
        is_weight_meta_fully_updated = self._update_by_original_name(
            weight_meta_classes, self.original_name2parent_weight_meta_class
        )
        if is_weight_meta_fully_updated:
            new_weight_meta_codes = []
            for meta_class in weight_meta_classes:
                new_weight_meta_codes.append(
                    self._generate_py_code_from_meta_class(meta_class)
                )

            weight_meta_file_path = os.path.join(model_path, "weight_meta.py")
            if self.config["update_inplace"]:
                print(f"[GraphMetaRestorer] Update {weight_meta_file_path}")
                with open(weight_meta_file_path, "w") as f:
                    f.write("\n\n".join(new_weight_meta_codes))

        is_input_meta_fully_updated = self._update_by_tensor_spec(
            input_meta_classes, self.original_name2parent_input_meta_class
        )
        if is_input_meta_fully_updated:
            new_input_meta_codes = []
            for meta_class in input_meta_classes:
                new_input_meta_codes.append(
                    self._generate_py_code_from_meta_class(meta_class)
                )

            input_meta_file_path = os.path.join(model_path, "input_meta.py")
            if self.config["update_inplace"]:
                print(f"[GraphMetaRestorer] Update {input_meta_file_path}")
                with open(input_meta_file_path, "w") as f:
                    f.write("\n\n".join(new_input_meta_codes))

    def _load_weight_and_input_meta_classes(self, model_path):
        weight_meta_file_path = os.path.join(model_path, "weight_meta.py")
        weight_meta_classes = [
            meta_class
            for (name, meta_class) in utils.get_meta_classes(weight_meta_file_path)
        ]

        input_meta_file_path = os.path.join(model_path, "input_meta.py")
        input_meta_classes = [
            meta_class
            for (name, meta_class) in utils.get_meta_classes(input_meta_file_path)
        ]

        return weight_meta_classes, input_meta_classes

    def _convert_to_dict(self, meta_classes):
        original_name2meta_class = {}
        for meta_class in meta_classes:
            assert meta_class.original_name not in original_name2meta_class.keys()
            original_name2meta_class[meta_class.original_name] = meta_class
        return original_name2meta_class

    def _update_tensor_meta(self, meta_class, parent_meta_class):
        if (
            parent_meta_class
            and meta_class.dtype == parent_meta_class.dtype
            and meta_class.shape == parent_meta_class.shape
        ):
            for attr_name in ["max_val", "min_val", "mean", "std", "data"]:
                if hasattr(meta_class, attr_name) or hasattr(
                    parent_meta_class, attr_name
                ):
                    attr_value = getattr(parent_meta_class, attr_name, None)
                    setattr(meta_class, attr_name, attr_value)
            return True
        return False

    def _update_by_original_name(self, meta_classes, original_name2parent_meta_class):
        updated_class_names = set()
        for meta_class in meta_classes:
            if not meta_class.original_name:
                continue

            parent_meta_class = original_name2parent_meta_class.get(
                meta_class.original_name, None
            )
            if self._update_tensor_meta(meta_class, parent_meta_class):
                updated_class_names.add(meta_class.name)

        print(
            f"[GraphMetaRestorer] {len(updated_class_names)}/{len(meta_classes)} classes are updated."
        )
        return len(meta_classes) == len(updated_class_names)

    def _update_by_tensor_spec(self, meta_classes, original_name2parent_meta_class):
        updated_class_names = set()
        for meta_class in meta_classes:
            matched_parent_meta_class = [
                parent_meta_class
                for parent_meta_class in original_name2parent_meta_class.values()
                if meta_class.dtype == parent_meta_class.dtype
                and meta_class.shape == parent_meta_class.shape
            ]
            if len(matched_parent_meta_class) == 1:
                self._update_tensor_meta(meta_class, matched_parent_meta_class[0])
                updated_class_names.add(meta_class.name)

        print(
            f"[GraphMetaRestorer] {len(updated_class_names)}/{len(meta_classes)} classes are updated."
        )
        return len(meta_classes) == len(updated_class_names)

    def _generate_py_code_from_meta_class(self, meta_class):
        lines = [f"class {meta_class.__name__}:"]
        members = vars(meta_class)
        members = {k: v for k, v in members.items() if not k.startswith("__")}

        if not members:
            return lines[0] + "\n    pass"

        for name, value in members.items():
            value_str = (
                f"float('{repr(value)}')" if isinstance(value, float) else repr(value)
            )
            lines.append(f"    {name} = {value_str}")
        return "\n".join(lines)
