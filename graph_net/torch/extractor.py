import os
import torch
import json
from typing import Union, Callable
from . import utils

def extract(name, dynamic=True):
    """
    A decorator for PyTorch functions to capture the computation graph.

    Args:
        name (str): The name of the model, used as the directory name for saving.
        dynamic (bool): Enable dynamic shape support in torch.compile.
    """
    def wrapper(model: torch.nn.Module):
        def extractor(gm: torch.fx.GraphModule, sample_inputs):
            # 1. Get workspace path
            workspace_path = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
            if not workspace_path:
                raise EnvironmentError("Environment variable 'GRAPH_NET_EXTRACT_WORKSPACE' is not set.")
            model_path = os.path.join(workspace_path, name)
            os.makedirs(model_path, exist_ok=True)

            # 2. Get full params
            params = {}
            input_idx = 0
            for node in gm.graph.nodes:
                if node.op == 'placeholder':
                    input = sample_inputs[input_idx]
                    if isinstance(input, torch.SymInt):
                        input = torch.tensor(0)
                    params[node.target] = input
                    input_idx += 1
            assert input_idx == len(sample_inputs)
            # 3. Generate and save model code
            base_code = gm.code
            write_code = utils.apply_templates(base_code)
            with open(os.path.join(model_path, 'model.py'), 'w') as fp:
                fp.write(write_code)

            # 4. Save metadata
            metadata = {
                "framework": "torch",
                "num_devices_required": 1,
                "num_nodes_required": 1
            }
            with open(os.path.join(model_path, 'graph_net.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

            # 5. Save tensor metadata
            # Adapt to different input structures (e.g., single tensor vs. dict/tuple of tensors)
            converted = utils.convert_state_and_inputs(params, [])
            utils.save_converted_to_text(
                converted,
                file_path=model_path
            )
            utils.save_constraints_text(
                converted,
                file_path=os.path.join(model_path, 'input_tensor_constraints.py')
            )

            print(f"Graph and tensors for '{name}' extracted successfully to: {model_path}")

            return gm.forward

        # return torch.compile(backend=extractor, dynamic=dynamic)
        compiled_model = torch.compile(model, backend=extractor, dynamic=dynamic)
        
        return compiled_model

    return wrapper
