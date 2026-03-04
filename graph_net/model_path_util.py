from graph_net.graph_net_root import get_graphnet_root
from graph_net_bench import test_compiler_util


def get_allow_samples(allow_list):
    graphnet_root = get_graphnet_root()
    return test_compiler_util.get_allow_samples(allow_list, graphnet_root)
