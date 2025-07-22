import os
import torch
import json
from graph_net.torch._extract import utils
from torch.export import export
from torch.fx import symbolic_trace


def get_param_or_buffer(module, dyanmo_source):
    if type(dyanmo_source).__name__ == 'TensorPropertySource':
        return None
    if type(dyanmo_source).__name__ == 'UnspecializedBuiltinNNModuleSource':
        return get_param_or_buffer(module, dyanmo_source.base)
    if type(dyanmo_source).__name__ == 'UnspecializedNNModuleSource':
        return get_param_or_buffer(module, dyanmo_source.base)
    if hasattr(dyanmo_source, 'member'):
        sub_module = get_param_or_buffer(module, dyanmo_source.base)
        return getattr(sub_module, dyanmo_source.member)
    if hasattr(dyanmo_source, 'index'):
        sub_module = get_param_or_buffer(module, dyanmo_source.base)
        return sub_module[dyanmo_source.index]
    assert type(dyanmo_source).__name__ == 'LocalSource'
    return module if dyanmo_source.local_name == 'self' else None

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
            ctrls = {}
            for node in gm.graph.nodes:
                if node.op == 'placeholder':
                    param = get_param_or_buffer(model, node._dynamo_source)
                    if param is not None:
                        params[node.target] = param
                    elif node.target == 's1':
                        ctrls[node.target] = torch.tensor(0)

            # 3. Generate and save model code
            base_code = gm.__str__()
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
            inputs_for_conversion = sample_inputs[0]
            if isinstance(inputs_for_conversion, tuple) and len(inputs_for_conversion) == 1 and isinstance(inputs_for_conversion[0], torch.Tensor):
                inputs_for_conversion = inputs_for_conversion[0]

            converted = utils.convert_state_and_inputs_and_ctrls(params, inputs_for_conversion, ctrls)
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