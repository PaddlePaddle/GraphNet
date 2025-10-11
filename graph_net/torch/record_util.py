import hash_util


def make_attrs_from_class(cls):
    return {
        k: v
        for k, v in cls.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }


def serialize_to_py_code(attrs, class_prefix):
    assert isinstance(attrs, (tuple, list))

    ret = "\n".join(
        _serialize_one_attr_to_py_code(attr, class_prefix) for attr in attrs
    )
    return ret


def _serialize_one_attr_to_py_code(attr, class_prefix):
    hash_str = hash_util.get_sha_hash(str(attr))
    hash_str = hash_str[:32]
    indent = " " * 4
    ret = "\n".join(
        [
            f"class {class_prefix}{hash_str}:",
            *[f"{indent}{name} = {repr(value)}" for name, value in attr.items()],
        ]
    )
    return f"{ret}\n\n"
