import graph_net.torch.runner.utils as utils
import argparse
import importlib.util
import torch
from pathlib import Path
from typing import Type, Any
import sys

def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    file = Path(file_path).resolve()
    module_name = file.stem

    with open(file_path, 'r', encoding='utf-8') as f:
        original_code = f.read()
    import_stmt= "import torch"
    modified_code = f"{import_stmt}\n{original_code}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    compiled_code = compile(modified_code, filename=file, mode='exec')
    exec(compiled_code, module.__dict__)

    model_class = getattr(module, class_name, None)
    return model_class

def main(model_path: str):
    model_class = load_class_from_file(f"{model_path}/model.py", class_name="GraphModule")
    model = model_class()

    inputs_params = utils.load_converted_from_text(f'{model_path}/source_tensor_meta.py')
    inputs = inputs_params["input_info"]
    inputs = [utils.replay_tensor(i) for i in inputs]
    params = inputs_params["weight_info"]

    state_dict = {}
    for k, v in params.items():
        k = utils.convert_param_name(k)
        v = utils.replay_tensor(v)
        state_dict[k] = v

    y = model(x=inputs[0], **state_dict)[0]
    print(torch.argmin(y), torch.argmax(y))
    print(y.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型文件夹的路径，如'../../samples/torch/resnet18'")
    args = parser.parse_args()
    main(model_path=args.model_path)
