import paddle
import os
import json
import numpy as np
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ExampleInputsGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.set_seed(123)  # 固定随机种子保证一致性

    def set_seed(self, random_seed: int):
        paddle.seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def load_model(self):
        """动态加载模型"""
        from importlib.util import spec_from_loader, module_from_spec
        import sys
        
        model_file = f"{self.model_path}/model.py"
        with open(model_file, "r") as f:
            code = f.read()
        
        module_name = Path(model_file).stem
        spec = spec_from_loader(module_name, loader=None)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        exec(compile(f"import paddle\n{code}", model_file, "exec"), module.__dict__)
        
        return module.GraphModule()

    def get_input_dict(self) -> Dict:
        """获取输入字典"""
        from graph_net.paddle import utils
        inputs_params = utils.load_converted_from_text(self.model_path)
        params = inputs_params["weight_info"]
        inputs = inputs_params["input_info"]
        params.update(inputs)
        return {k: utils.replay_tensor(v) for k, v in params.items()}

    def generate(self, result_dir: str) -> Dict:
        """生成示例输入并保存到指定目录"""
        os.makedirs(result_dir, exist_ok=True)
        
        # 复制模型文件到目标目录
        for file in ['model.py', 'graph_net.json']:
            src = os.path.join(self.model_path, file)
            dst = os.path.join(result_dir, file)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {file} to benchmark directory")
        
        model = self.load_model()
        model.eval()
        input_dict = self.get_input_dict()
        
        # 分离输入和权重
        inputs = {}
        weights = {}
        
        # 从graph_net.json读取权重信息
        graph_net_file = os.path.join(self.model_path, 'graph_net.json')
        if os.path.exists(graph_net_file):
            with open(graph_net_file, 'r') as f:
                graph_net_data = json.load(f)
                weight_info = graph_net_data.get('weight_info', {})
                for key in weight_info.keys():
                    if key in input_dict:
                        weights[key] = input_dict.pop(key)
        
        # 剩下的都是输入
        inputs = input_dict
        
        # 获取静态模型运行一次以获取参考输出
        static_model = paddle.jit.to_static(
            model,
            input_spec=self.get_input_spec(),
            full_graph=True,
            backend=None,
        )
        static_model.eval()
        
        with paddle.no_grad():
            output = static_model(**input_dict)
        
        # 保存输入数据
        inputs_file = f"{result_dir}/example_inputs.json"
        with open(inputs_file, "w") as f:
            json.dump(self._serialize_tensors(inputs), f, indent=2)
        
        # 保存权重数据
        weights_file = f"{result_dir}/example_weights.json"
        with open(weights_file, "w") as f:
            json.dump(self._serialize_tensors(weights), f, indent=2)
        
        print(f"Generated example inputs saved to: {inputs_file}")
        print(f"Generated example weights saved to: {weights_file}")
        
        return {
            "inputs": self._serialize_tensors(inputs),
            "weights": self._serialize_tensors(weights),
            "outputs": self._serialize_tensor(output)
        }

    def get_input_spec(self):
        """获取输入规范"""
        from graph_net.paddle import utils
        inputs_params = utils.load_converted_list_from_text(self.model_path)
        return [
            paddle.static.InputSpec(shape=v["info"]["shape"], dtype=v["info"]["dtype"])
            for v in inputs_params
        ]

    def _serialize_tensors(self, tensor_dict: Dict) -> Dict:
        """序列化张量字典"""
        return {
            k: {
                "data": t.numpy().tolist(),
                "dtype": str(t.dtype).replace("paddle.", ""),
                "shape": list(t.shape)
            } for k, t in tensor_dict.items()
        }

    def _serialize_tensor(self, tensor) -> List:
        """序列化单个张量或张量列表"""
        if isinstance(tensor, paddle.Tensor):
            tensor = [tensor]
        return [{
            "data": t.numpy().tolist(),
            "dtype": str(t.dtype).replace("paddle.", ""),
            "shape": list(t.shape)
        } for t in tensor]


def generate_example_inputs(args):
    """兼容原有函数接口"""
    generator = ExampleInputsGenerator(args.model_path)
    return generator.generate(args.result_dir)


def set_seed(random_seed):
    """兼容原有函数"""
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def generate_for_benchmark(model_path: str, benchmark_path: str) -> bool:
    """为benchmark目录生成输入文件"""
    try:
        # 构建输出目录结构（保持与模型相同的相对路径）
        model_rel_path = os.path.relpath(model_path, os.getcwd())
        output_dir = os.path.join(benchmark_path, model_rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating inputs for: {model_path}")
        print(f"Output directory: {output_dir}")
        
        generator = ExampleInputsGenerator(model_path)
        generator.generate(output_dir)
        return True
    except Exception as e:
        print(f"Failed to generate inputs: {str(e)}")
        return False


def main():
    """命令行接口 - 支持单模型和批量生成"""
    parser = argparse.ArgumentParser(description="Generate example inputs for models")
    parser.add_argument(
        '--model-path',
        required=True,
        help='Path to model directory or samples directory'
    )
    parser.add_argument(
        '--benchmark-path',
        required=True,
        help='Base directory to store generated inputs'
    )
    
    args = parser.parse_args()
    
    # 确保benchmark目录存在
    os.makedirs(args.benchmark_path, exist_ok=True)
    
    # 检查是单模型还是目录
    if os.path.isfile(os.path.join(args.model_path, 'model.py')):
        # 单模型
        success = generate_for_benchmark(args.model_path, args.benchmark_path)
        exit(0 if success else 1)
    else:
        # 目录批量生成
        success_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(args.model_path):
            if 'model.py' in files:
                total_count += 1
                if generate_for_benchmark(root, args.benchmark_path):
                    success_count += 1
        
        print(f"\nSummary: {success_count}/{total_count} models generated successfully")
        exit(0 if success_count == total_count else 1)


if __name__ == "__main__":
    main()