# 统计 samples 目录下， graph_net.json 文件的数量
import os

graph_net_count = 0
for root, dirs, files in os.walk("../samples"):
    for file in files:
        if file == "graph_net.json":
            graph_net_count += 1
print(f"Number of graph_net.json files: {graph_net_count}")
