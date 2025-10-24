import paddle
import os
import json
import time
import numpy as np
import random
import platform
import traceback
import argparse
import sys
from pathlib import Path
from typing import Dict, List

from graph_net.paddle import utils
from graph_net import test_compiler_util


class FixedRandomSeedDeviceRunner:
    def __init__(self, model_path: str, compiler: str = "cinn", device: str = "cuda", 
                 warmup: int = 5, trials: int = 10):
        self.model_path = model_path
        self.compiler = compiler
        self.device = device
        self.warmup = warmup
        self.trials = trials
        self.set_seed(123)  # 固定随机种子

    def set_seed(self, random_seed: int):
        """设置随机种子"""
        paddle.seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def get_hardware_name(self) -> str:
        """获取硬件名称"""
        if self.device == "cuda":
            return paddle.device.cuda.get_device_name(0)
        elif self.device == "cpu":
            return platform.processor()
        return "unknown"

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

    def get_synchronizer_func(self):
        """获取同步函数"""
        return paddle.device.synchronize
    def generate_inputs_from_seed(self) -> Dict:
        """通过固定种子生成输入张量"""
        # 从元数据文件获取张量形状和类型
        input_meta_file = os.path.join(self.model_path, "input_meta.py")
        weight_meta_file = os.path.join(self.model_path, "weight_meta.py")
        
        all_tensors = {}
        
        # 动态导入input_meta模块
        if os.path.exists(input_meta_file):
            import importlib.util
            spec = importlib.util.spec_from_file_location("input_meta", input_meta_file)
            input_meta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(input_meta)
            
            # 收集输入张量信息
            inputs = {}
            for attr_name in dir(input_meta):
                if attr_name.startswith("Program_weight_tensor_"):
                    cls = getattr(input_meta, attr_name)
                    if hasattr(cls, "name") and hasattr(cls, "shape") and hasattr(cls, "dtype"):
                        inputs[cls.name] = {"shape": cls.shape, "dtype": cls.dtype}
            
            # 为每个输入张量分配独立的种子
            for i, key in enumerate(inputs.keys()):
                seed = 123 + i
                tensor_info = inputs[key]
                shape = tensor_info["shape"]
                dtype = tensor_info["dtype"]
                # 使用numpy生成随机数据
                np.random.seed(seed)
                if dtype in ['float32', 'float64']:
                    data = np.random.rand(*shape).astype(dtype)
                elif dtype in ['int32', 'int64']:
                    data = np.random.randint(0, 100, size=shape).astype(dtype)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
                all_tensors[key] = paddle.to_tensor(data, dtype=dtype)
        
        # 动态导入weight_meta模块  
        if os.path.exists(weight_meta_file):
            import importlib.util
            spec = importlib.util.spec_from_file_location("weight_meta", weight_meta_file)
            weight_meta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weight_meta)
            
            # 收集权重张量信息
            weights = {}
            for attr_name in dir(weight_meta):
                if attr_name.startswith("Program_weight_tensor_"):
                    cls = getattr(weight_meta, attr_name)
                    if hasattr(cls, "name") and hasattr(cls, "shape") and hasattr(cls, "dtype"):
                        weights[cls.name] = {"shape": cls.shape, "dtype": cls.dtype}
            
            # 为每个权重张量分配独立的种子
            for i, key in enumerate(weights.keys()):
                seed = 123 + i + 1000
                tensor_info = weights[key]
                shape = tensor_info["shape"]
                dtype = tensor_info["dtype"]
                # 使用numpy生成随机数据
                np.random.seed(seed)
                if dtype in ['float32', 'float64']:
                    data = np.random.rand(*shape).astype(dtype)
                elif dtype in ['int32', 'int64']:
                    data = np.random.randint(0, 100, size=shape).astype(dtype)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
                all_tensors[key] = paddle.to_tensor(data, dtype=dtype)
        
        return all_tensors

    def get_input_spec(self):
        """获取输入规范"""
        try:
            # 从元数据文件获取张量信息
            input_meta_file = os.path.join(self.model_path, "input_meta.py")
            weight_meta_file = os.path.join(self.model_path, "weight_meta.py")
            
            if not os.path.exists(input_meta_file) or not os.path.exists(weight_meta_file):
                return []
            
            # 动态导入元数据模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("input_meta", input_meta_file)
            input_meta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(input_meta)
            
            spec = importlib.util.spec_from_file_location("weight_meta", weight_meta_file)
            weight_meta = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(weight_meta)
            
            input_specs = []
            
            # 从input_meta.py获取输入张量规范
            for attr_name in dir(input_meta):
                if attr_name.startswith("Program_weight_tensor_"):
                    cls = getattr(input_meta, attr_name)
                    if hasattr(cls, "shape") and hasattr(cls, "dtype"):
                        shape = cls.shape
                        dtype = cls.dtype
                        paddle_dtype = getattr(paddle, dtype)
                        input_specs.append(paddle.static.InputSpec(shape=shape, dtype=paddle_dtype))
            
            # 从weight_meta.py获取权重张量规范
            for attr_name in dir(weight_meta):
                if attr_name.startswith("Program_weight_tensor_"):
                    cls = getattr(weight_meta, attr_name)
                    if hasattr(cls, "shape") and hasattr(cls, "dtype"):
                        shape = cls.shape
                        dtype = cls.dtype
                        paddle_dtype = getattr(paddle, dtype)
                        input_specs.append(paddle.static.InputSpec(shape=shape, dtype=paddle_dtype))
            
            return input_specs
        except Exception as e:
            print(f"Warning: Failed to get input spec from metadata: {e}")
            return []

    def get_compiled_model(self, model):
        """获取编译后的模型"""
        if self.compiler == "nope":
            return model
        input_spec = self.get_input_spec()
        build_strategy = paddle.static.BuildStrategy()
        compiled_model = paddle.jit.to_static(
            model,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )
        compiled_model.eval()
        return compiled_model

    def measure_performance(self, model_call, synchronizer_func):
        """测量性能"""
        # Warmup runs
        for _ in range(self.warmup):
            model_call()
        synchronizer_func()

        # 性能测试
        e2e_times = []
        for i in range(self.trials):
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, synchronizer_func):
                model_call()
            e2e_times.append(duration_box.value)
            print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms")

        stats = test_compiler_util.get_timing_stats(e2e_times)
        return stats

    def run(self) -> bool:
        """运行测试"""
        synchronizer_func = self.get_synchronizer_func()
        
        # 生成输入张量
        input_dict = self.generate_inputs_from_seed()
        if not input_dict:
            print("Error: Failed to generate inputs from seed")
            return False
        
        model = self.load_model()
        model.eval()

        # 打印基本配置信息
        print(f"[Config] device: {self.device}")
        print(f"[Config] compiler: {self.compiler}")
        print(f"[Config] hardware: {self.get_hardware_name()}")
        print(f"[Config] framework_version: {paddle.__version__}")
        print(f"[Config] warmup: {self.warmup}")
        print(f"[Config] trials: {self.trials}")

        # 运行测试
        success = False
        try:
            print(f"Run model with compiler: {self.compiler}")
            if self.compiler == "nope":
                compiled_model = model
            else:
                compiled_model = self.get_compiled_model(model)
            
            # 测量性能
            time_stats = self.measure_performance(
                lambda: compiled_model(**input_dict), synchronizer_func
            )
            success = True
            
            # 打印结果（不保存到文件，直接输出到控制台）
            print(f"[Result] model_path: {self.model_path}")
            print(f"[Result] compiler: {self.compiler}")
            print(f"[Result] device: {self.device}")
            print(f"[Result] e2e_mean: {time_stats['mean']:.5f}")
            print(f"[Result] e2e_std: {time_stats['std']:.5f}")
            
        except Exception as e:
            print(f"Run model failed: {str(e)}")
            print(traceback.format_exc())

        return success


def main():
    """命令行接口 - 简化的单模型测试"""
    parser = argparse.ArgumentParser(description="Test device performance with fixed random seeds")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing (e.g., 'cpu', 'cuda', or 'dcu')"
    )
    parser.add_argument(
        "--compiler",
        type=str,
        required=False,
        default="cinn",
        help="Compiler backend to use (cinn or nope)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        required=False,
        default=5,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=False,
        default=10,
        help="Number of timing trials"
    )
    
    args = parser.parse_args()
    
    # 运行测试
    runner = FixedRandomSeedDeviceRunner(
        model_path=args.model_path,
        compiler=args.compiler,
        device=args.device,
        warmup=args.warmup,
        trials=args.trials
    )
    
    success = runner.run()
    return 0 if success else 1


if __name__ == "__main__":
    main()