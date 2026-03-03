import os
import graph_net


def get_graphnet_root():
    return os.path.dirname(os.path.dirname(graph_net.__file__))
