import graph_net
import os


def get_default_samples_directory():
    return f"{os.path.dirname(graph_net.__file__)}/../samples"
