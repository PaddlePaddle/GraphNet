import graph_net
import os


def get_default_samples_directory():
    graph_net_root = os.path.dirname(os.path.dirname(graph_net.__file__))
    return f"{graph_net_root}/paddle_samples"
