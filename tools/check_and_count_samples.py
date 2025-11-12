import os
import json


def check_completeness(samples_dir):
    samples_missing_hash = []
    samples_missing_json = []
    samples_missing_meta = []
    samples_missing_model = []
    for root, dirs, files in os.walk(samples_dir):
        model_path = root
        if "shape_patches_" not in root and "model.py" in files:
            if not os.path.exists(os.path.join(model_path, "graph_hash.txt")):
                samples_missing_hash.append(model_path)
            if not os.path.exists(os.path.join(model_path, "graph_net.json")):
                samples_missing_json.append(model_path)
            if not os.path.exists(
                os.path.join(model_path, "input_meta.py")
            ) or not os.path.exists(os.path.join(model_path, "weight_meta.py")):
                samples_missing_meta.append(model_path)
        if "graph_net.json" in files and "model.py" not in files:
            samples_missing_model.append(model_path)

    all_samples_complete = (
        len(samples_missing_hash) == 0
        and len(samples_missing_json) == 0
        and len(samples_missing_meta) == 0
        and len(samples_missing_model) == 0
    )

    if not all_samples_complete:
        print(f"Check completeness result for {samples_dir}:")
        print(f"1. {len(samples_missing_hash)} samples missing graph_hash.txt")
        for model_path in samples_missing_hash:
            print(f"  - {model_path}")

        print(f"2. {len(samples_missing_json)} samples missing graph_net.json")
        for model_path in samples_missing_json:
            print(f"  - {model_path}")

        print(
            f"3. {len(samples_missing_meta)} samples missing input_meta.py or weight_meta.py"
        )
        for model_path in samples_missing_meta:
            print(f"  - {model_path}")

        print(f"4. {len(samples_missing_model)} samples missing model.py")
        for model_path in samples_missing_model:
            print(f"  - {model_path}")

        print()

    return all_samples_complete


def check_redandancy(samples_dir):
    graph_hash2model_paths = {}
    for root, dirs, files in os.walk(samples_dir):
        if "graph_hash.txt" in files:
            model_path = root
            graph_hash_path = os.path.join(model_path, "graph_hash.txt")
            graph_hash = open(graph_hash_path).read()
            if graph_hash not in graph_hash2model_paths.keys():
                graph_hash2model_paths[graph_hash] = [model_path]
            else:
                graph_hash2model_paths[graph_hash].append(model_path)

    has_duplicates = False
    print(f"Totally {len(graph_hash2model_paths)} unique graphs under {samples_dir}.")
    for graph_hash, model_paths in graph_hash2model_paths.items():
        graph_hash2model_paths[graph_hash] = sorted(model_paths)
        if len(model_paths) > 1:
            has_duplicates = True
            print(f"Redundant models detected for grap_hash {graph_hash}:")
            for model_path in model_paths:
                print(f"    {model_path}")
    return has_duplicates, graph_hash2model_paths


def count_samples(samples_dir, framework):
    model_sources = os.listdir(samples_dir)

    graph_net_count = 0
    graph_net_dict = {}
    model_names_set = set()
    for source in model_sources:
        source_dir = os.path.join(samples_dir, source)
        if os.path.isdir(source_dir):
            graph_net_dict[source] = 0
            for root, dirs, files in os.walk(source_dir):
                if "graph_net.json" in files:
                    with open(os.path.join(root, "graph_net.json"), "r") as f:
                        data = json.load(f)
                        model_name = data.get("model_name", None)
                    if model_name is not None and model_name != "NO_VALID_MATCH_FOUND":
                        if model_name not in model_names_set:
                            model_names_set.add(model_name)
                            graph_net_count += 1
                            graph_net_dict[source] += 1
                    else:
                        graph_net_count += 1
                        graph_net_dict[source] += 1

    print(f"Number of {framework} samples: {graph_net_count}")
    for name, number in graph_net_dict.items():
        print(f"- {name:24}: {number}")
    print()


def main():
    filename = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(filename))

    framework2dirname = {
        "torch": "samples",
        "paddle": "paddle_samples",
    }

    all_samples_complete = True
    for samples_dirname in framework2dirname.values():
        samples_dir = os.path.join(root_dir, samples_dirname)
        all_samples_complete = all_samples_complete and check_completeness(samples_dir)
    assert all_samples_complete, "Please fix the incompleted samples!"

    all_samples_has_duplicates = False
    for samples_dirname in framework2dirname.values():
        samples_dir = os.path.join(root_dir, samples_dirname)
        has_duplicates, graph_hash2model_paths = check_redandancy(samples_dir)
        all_samples_has_duplicates = all_samples_has_duplicates or has_duplicates
    print()
    assert not all_samples_has_duplicates, "Please remove the redundant samples!"

    for framework in framework2dirname.keys():
        samples_dir = os.path.join(root_dir, framework2dirname[framework])
        count_samples(samples_dir, framework)


if __name__ == "__main__":
    main()
