from . import utils
import argparse
import importlib.util
import inspect
import torch
from pathlib import Path
from typing import Type, Any
import sys
from graph_net.torch.extractor import extract


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

def main(args):
    model_path = args.model_path
    model_class = load_class_from_file(f"{model_path}/model.py", class_name="GraphModule")
    model = model_class()
    print(f'{model_path=}')
    if args.enable_extract:
        assert args.extract_name is not None
        model = extract(name=args.extract_name)(model)

    inputs_params = utils.load_converted_from_text(f'{model_path}')
    params = inputs_params["weight_info"]
    state_dict = {
        k: utils.replay_tensor(v) for k, v in params.items()
    }
    
    y = model(**state_dict)[0]

    print(torch.argmin(y), torch.argmax(y))
    print(y.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to folder e.g '../../samples/torch/resnet18'")
    parser.add_argument("--enable-extract", type=bool, required=False, default=False, help="Enable extract")
    parser.add_argument("--extract-name", type=str, required=False, default=None, help="Extracted graph's name")
    args = parser.parse_args()
    main(args=args)
