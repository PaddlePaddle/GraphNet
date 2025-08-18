# 统计 samples 目录下， graph_net.json 文件的数量
import os


filename = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(filename))
framework2dirname = {
    "torch": "samples",
    "paddle": "paddle_samples",
}

for framework in ["torch", "paddle"]:
    samples_dir = os.path.join(root_dir, framework2dirname[framework])
    model_categories = os.listdir(samples_dir)

    graph_net_count = 0
    graph_net_dict = {}
    for category in model_categories:
        category_dir = os.path.join(samples_dir, category)
        if os.path.isdir(category_dir):
            graph_net_dict[category] = 0
            for root, dirs, files in os.walk(category_dir):
                if "graph_net.json" in files:
                    graph_net_count += 1
                    graph_net_dict[category] += 1

    print(f"Number of {framework} samples: {graph_net_count}")
    for name, number in graph_net_dict.items():
        print(f"- {name:24}: {number}")
    print()
