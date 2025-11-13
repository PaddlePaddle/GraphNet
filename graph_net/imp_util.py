import importlib.util as imp


def load_module(path, name="unamed"):
    spec = imp.spec_from_file_location(name, path)
    module = imp.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
