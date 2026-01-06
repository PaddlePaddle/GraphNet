import os
import pickle
import numpy as np
import random
import re
import paddle

from graph_net.paddle import samples_util


def set_seed(random_seed):
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def _extract_model_name_for_original_sample(model_path):
    fields = model_path.rstrip("/").split(os.sep)
    pattern = r"^subgraph(_\d+)?$"
    model_name = (
        f"{fields[-2]}_{fields[-1]}" if re.match(pattern, fields[-1]) else fields[-1]
    )
    return model_name


def _extract_model_name_for_decomposed_subgraph(model_path):
    # Parse model name and subgraph index
    model_name_with_subgraph_idx = model_path.rstrip("/").split(os.sep)[-1]
    model_name = "_".join(model_name_with_subgraph_idx.split("_")[:-1])
    return model_name


def _generate_random_state_filename(model_path):
    samples_dir = samples_util.get_default_samples_directory()
    if os.path.abspath(model_path).startswith(samples_dir):
        model_name = _extract_model_name_for_original_sample(model_path)
    else:
        model_name = _extract_model_name_for_decomposed_subgraph(model_path)
    return f"{model_name}.random_states.pkl"


def save_random_states(model_path, output_dir, random_state_dict):
    filepath = os.path.join(output_dir, _generate_random_state_filename(model_path))
    print(f"Write to {filepath}.", flush=True)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(random_state_dict, f)
    except Exception:
        print(f"Fail to open {filepath}.")


def load_random_states(model_path, output_dir):
    filepath = os.path.join(output_dir, _generate_random_state_filename(model_path))
    print(f"Read from {filepath}.", flush=True)
    random_states = None
    try:
        with open(filepath, "rb") as f:
            random_states = pickle.load(f)
    except Exception:
        print(f"Fail to open {filepath}.")
    return random_states
