import os
import importlib.util as imp


def load_module(path, name="unnamed"):
    spec = imp.spec_from_file_location(name, path)
    module = imp.module_from_spec(spec)
    module.__file__ = path
    spec.loader.exec_module(module)
    module.__graph_net_file_path__ = os.path.normpath(path)
    return module
