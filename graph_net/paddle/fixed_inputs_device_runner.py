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
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr

from graph_net.paddle import utils
from graph_net import path_utils
from graph_net import test_compiler_util
from graph_net.paddle.device_utils import DeviceEnvironment, create_device_spec_file


class FixedInputsDeviceRunner:
    def __init__(self, model_path: str, compiler: str = "cinn", device: str = "cuda", 
                 warmup: int = 5, trials: int = 10, log_prompt: str = "graph-net-fixed-inputs-runner"):
        self.model_path = model_path
        self.compiler = compiler
        self.device = device
        self.warmup = warmup
        self.trials = trials
        self.log_prompt = log_prompt
        self.set_seed(123)

    def set_seed(self, random_seed: int):
        paddle.seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def get_hardware_name(self) -> str:
        if self.device == "cuda":
            return paddle.device.cuda.get_device_name(0)
        elif self.device == "cpu":
            return platform.processor()
        return "unknown"

    def get_compile_framework_version(self) -> str:
        if self.compiler == "cinn":
            return paddle.__version__
        if self.compiler == "nope":
            return "nope-baseline"
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
        return paddle.device.synchronize

    def get_input_dict(self) -> Dict:
        """获取输入字典"""
        inputs_params = utils.load_converted_from_text(self.model_path)
        params = inputs_params["weight_info"]
        inputs = inputs_params["input_info"]
        params.update(inputs)
        return {k: utils.replay_tensor(v) for k, v in params.items()}

    def get_input_spec(self):
        """获取输入规范"""
        # 首先尝试从固定的输入数据中获取输入规范
        try:
            # 从模型目录读取example_inputs.json
            inputs_file = os.path.join(self.model_path, "example_inputs.json")
            if os.path.exists(inputs_file):
                with open(inputs_file, "r") as f:
                    inputs_data = json.load(f)
                
                # 从固定输入数据构建输入规范
                input_specs = []
                for key, tensor_info in inputs_data.items():
                    shape = tensor_info["shape"]
                    dtype = tensor_info["dtype"]
                    # 将字符串dtype转换为paddle dtype
                    paddle_dtype = getattr(paddle, dtype)
                    input_specs.append(paddle.static.InputSpec(shape=shape, dtype=paddle_dtype))
                
                return input_specs
        except Exception as e:
            print(f"Warning: Failed to get input spec from fixed inputs: {e}")
        
        # 如果固定输入方法失败，回退到原来的方法
        try:
            inputs_params = utils.load_converted_list_from_text(self.model_path)
            return [
                paddle.static.InputSpec(shape=v["info"]["shape"], dtype=v["info"]["dtype"])
                for v in inputs_params
            ]
        except Exception as e:
            print(f"Error: Failed to get input spec: {e}")
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

    def load_fixed_inputs(self, result_dir: str) -> Dict:
        """从结果目录加载预先生成的固定输入"""
        model_dir = os.path.dirname(result_dir)
        inputs_file = os.path.join(model_dir, "example_inputs.json")
        
        if not os.path.exists(inputs_file):
            print(f"Example inputs file not found: {inputs_file}")
            return None
        
        try:
            with open(inputs_file, "r") as f:
                inputs_data = json.load(f)
            
            input_dict = {}
            for key, tensor_info in inputs_data.items():
                data = np.array(tensor_info["data"])
                dtype = tensor_info["dtype"]
                input_dict[key] = paddle.to_tensor(data, dtype=dtype)
            
            return input_dict
        except Exception as e:
            print(f"Error loading fixed inputs: {e}")
            return None

    def load_fixed_weights(self, result_dir: str) -> Dict:
        """从结果目录加载预先生成的固定权重"""
        model_dir = os.path.dirname(result_dir)
        weights_file = os.path.join(model_dir, "example_weights.json")
        
        if not os.path.exists(weights_file):
            print(f"Example weights file not found: {weights_file}")
            return None
        
        try:
            with open(weights_file, "r") as f:
                weights_data = json.load(f)
            
            weights_dict = {}
            for key, tensor_info in weights_data.items():
                data = np.array(tensor_info["data"])
                dtype = tensor_info["dtype"]
                weights_dict[key] = paddle.to_tensor(data, dtype=dtype)
            
            return weights_dict
        except Exception as e:
            print(f"Error loading fixed weights: {e}")
            return None

    def measure_performance(self, model_call, synchronizer_func, profile=False):
        """测量性能"""
        runtime_seed = 1024
        stats = {}

        paddle.seed(runtime_seed)
        outs = model_call()

        # Warmup runs
        for _ in range(self.warmup):
            model_call()
        synchronizer_func()

        hardware_name = self.get_hardware_name()
        print(
            f"[Profiling] Using device: {self.device} {hardware_name}, warm up {self.warmup}, trials {self.trials}",
            flush=True,
        )

        if "cuda" in self.device:
            e2e_times = []
            gpu_times = []

            if profile:
                paddle.base.core.nvprof_start()
            for i in range(self.trials):
                duration_box = test_compiler_util.DurationBox(-1)
                with test_compiler_util.naive_timer(duration_box, synchronizer_func):
                    start_event = paddle.device.Event(enable_timing=True)
                    end_event = paddle.device.Event(enable_timing=True)

                    start_event.record()
                    model_call()
                    end_event.record()

                gpu_time_ms = start_event.elapsed_time(end_event)
                e2e_times.append(duration_box.value)
                gpu_times.append(gpu_time_ms)
                print(
                    f"Trial {i + 1}: e2e={duration_box.value:.5f} ms, gpu={gpu_time_ms:.5f} ms",
                    flush=True,
                )
            if profile:
                paddle.base.core.nvprof_stop()

            stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)
            stats["gpu"] = test_compiler_util.get_timing_stats(gpu_times)
        else:  # CPU or other devices
            e2e_times = []
            for i in range(self.trials):
                duration_box = test_compiler_util.DurationBox(-1)
                with test_compiler_util.naive_timer(duration_box, synchronizer_func):
                    model_call()
                print(
                    f"Trial {i + 1}: e2e={duration_box.value:.4f} ms",
                    flush=True,
                )
                e2e_times.append(duration_box.value)
            stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)

        return outs, stats

    def check_outputs(self, expected_out, compiled_out):
        """检查输出一致性"""
        if isinstance(expected_out, paddle.Tensor):
            expected_out = [expected_out]
        if isinstance(compiled_out, paddle.Tensor):
            compiled_out = [compiled_out]

        eager_dtypes = [None] * len(expected_out)
        for i, tensor in enumerate(expected_out):
            eager_dtypes[i] = (
                str(tensor.dtype).replace("paddle.", "") if tensor is not None else "none"
            )

        compiled_dtypes = [None] * len(compiled_out)
        for i, tensor in enumerate(compiled_out):
            compiled_dtypes[i] = (
                str(tensor.dtype).replace("paddle.", "") if tensor is not None else "none"
            )

        type_match = test_compiler_util.check_output_datatype(
            self, eager_dtypes, compiled_dtypes
        )

        def transfer_to_float(origin_outputs):
            outputs = []
            for item in origin_outputs:
                if (
                    item is not None
                    and isinstance(item, paddle.Tensor)
                    and item.dtype not in [paddle.float32, paddle.float64]
                ):
                    item = item.astype("float32")
                outputs.append(item)
            return outputs

        if type_match:
            test_compiler_util.check_equal(
                self,
                expected_out,
                compiled_out,
                cmp_equal_func=self.get_cmp_equal,
            )

            expected_out_fp32 = transfer_to_float(expected_out)
            compiled_out_fp32 = transfer_to_float(compiled_out)
            test_compiler_util.check_allclose(
                self,
                expected_out_fp32,
                compiled_out_fp32,
                cmp_all_close_func=self.get_cmp_all_close,
                cmp_max_diff_func=self.get_cmp_max_diff,
                cmp_mean_diff_func=self.get_cmp_mean_diff,
                cmp_max_relative_diff_func=self.get_cmp_max_relative_diff,
                cmp_mean_relative_diff_func=self.get_cmp_mean_relative_diff,
            )

    def run(self, result_dir: str) -> bool:
        """运行测试"""
        synchronizer_func = self.get_synchronizer_func()
        
        # 分别加载固定输入和权重
        input_dict = self.load_fixed_inputs(result_dir)
        weights_dict = self.load_fixed_weights(result_dir)
        
        # 合并输入和权重
        if input_dict is not None and weights_dict is not None:
            input_dict = {**input_dict, **weights_dict}
        elif input_dict is None:
            print("No pre-generated example inputs found, generating dynamically...")
            input_dict = self.get_input_dict()
        
        model = self.load_model()
        model.eval()

        test_compiler_util.print_basic_config(
            self, self.get_hardware_name(), self.get_compile_framework_version()
        )

        # 只运行一次（根据compiler参数选择backend）
        success = False
        try:
            print(f"Run model with compiler: {self.compiler}", flush=True)
            if self.compiler == "nope":
                compiled_model = model
            else:
                compiled_model = self.get_compiled_model(model)
            
            compiled_out, time_stats = self.measure_performance(
                lambda: compiled_model(**input_dict), synchronizer_func, profile=False
            )
            success = True
            
            # 记录结果到日志（使用类似配置信息的格式）
            print(f"{self.log_prompt} [Result] model_path: {self.model_path}")
            print(f"{self.log_prompt} [Result] compiler: {self.compiler}")
            print(f"{self.log_prompt} [Result] device: {self.device}")
            print(f"{self.log_prompt} [Result] timestamp: {time.time()}")
            
            for metric, values in time_stats.items():
                print(f"{self.log_prompt} [Result] {metric}_mean: {values['mean']:.5f}")
                print(f"{self.log_prompt} [Result] {metric}_std: {values['std']:.5f}")
            
            if isinstance(compiled_out, paddle.Tensor):
                print(f"{self.log_prompt} [Result] output: {compiled_out.numpy().tolist()}")
            else:
                for i, out in enumerate(compiled_out):
                    print(f"{self.log_prompt} [Result] output_{i}: {out.numpy().tolist()}")
            
        except Exception as e:
            print(
                f"Run model failed: {str(e)}\n{traceback.format_exc()}",
                flush=True,
            )

        return success

    def get_cmp_equal(self, expected_out, compiled_out):
        def convert(x):
            if x.dtype in [paddle.float16, paddle.bfloat16]:
                return x.astype("float32")
            elif x.dtype in [paddle.uint8, paddle.int8, paddle.int16]:
                return x.astype("int32")
            return x

        return " ".join(
            str(int(paddle.equal_all(convert(a), convert(b))))
            for a, b in zip(expected_out, compiled_out)
        )

    def get_cmp_all_close(self, expected_out, compiled_out, atol, rtol):
        return " ".join(
            str(int(paddle.allclose(a, b, atol=atol, rtol=rtol)))
            for a, b in zip(expected_out, compiled_out)
        )

    def get_format_str(self, f):
        if (abs(f) > 1e5 or abs(f) < 1e-5) and abs(f) != 0.0:
            return str(f"{f:.5E}")
        else:
            return str(f"{f:.5f}")

    def get_cmp_max_diff(self, expected_out, compiled_out):
        return " ".join(
            self.get_format_str(paddle.max(paddle.abs(a - b)).item())
            for a, b in zip(expected_out, compiled_out)
        )

    def get_cmp_mean_diff(self, expected_out, compiled_out):
        return " ".join(
            self.get_format_str(paddle.mean(paddle.abs(a - b)).item())
            for a, b in zip(expected_out, compiled_out)
        )

    def get_cmp_max_relative_diff(self, expected_out, compiled_out):
        epsilon = 1e-8
        return " ".join(
            self.get_format_str(paddle.max(paddle.abs(a - b) / (paddle.abs(a) + epsilon)).item())
            for a, b in zip(expected_out, compiled_out)
        )

    def get_cmp_mean_relative_diff(self, expected_out, compiled_out):
        epsilon = 1e-8
        return " ".join(
            self.get_format_str(
                paddle.mean(paddle.abs(a - b) / (paddle.abs(a) + epsilon)).item()
            )
            for a, b in zip(expected_out, compiled_out)
        )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_model_paths(benchmark_path: str):
    """在benchmark目录下查找所有包含example_inputs.json的模型目录"""
    model_paths = []
    for root, dirs, files in os.walk(benchmark_path):
        if 'example_inputs.json' in files:
            # 获取相对于benchmark_path的路径
            rel_path = os.path.relpath(root, benchmark_path)
            # 如果相对路径是当前目录，则使用空字符串
            if rel_path == '.':
                rel_path = ''
            model_paths.append(rel_path)
    return model_paths


def run_single_device_test(model_rel_path, benchmark_path, compiler, device, warmup, trials):
    """在单个设备上运行测试"""
    print(f"Running test on device: {device}, compiler: {compiler}")
    
    # 构建完整路径
    model_full_path = os.path.join(benchmark_path, model_rel_path)
    log_file = f"{model_full_path}/run_{device}_{compiler}.log"
    
    # 重定向输出到字符串，以便保存日志
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # 创建设备环境并记录设备信息
        device_env = DeviceEnvironment(device, compiler)
        device_env.setup_environment()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # 记录设备规格信息
            print(f"test-device-{compiler}-{device} [Configurations] device: {device}")
            print(f"test-device-{compiler}-{device} [Configurations] compiler: {compiler}")
            print(f"test-device-{compiler}-{device} [Configurations] hardware: {device_env.hardware_info.get('device_name', 'unknown')}")
            print(f"test-device-{compiler}-{device} [Configurations] framework_version: {paddle.__version__}")
            
            runner = FixedInputsDeviceRunner(
                model_path=model_full_path,
                compiler=compiler,
                device=device,
                warmup=warmup,
                trials=trials,
                log_prompt=f"test-device-{compiler}-{device}"
            )
            success = runner.run(model_full_path)
        
        # 保存日志到结果目录
        with open(log_file, "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout_capture.getvalue())
            f.write("\n=== STDERR ===\n")
            f.write(stderr_capture.getvalue())
        
        return success
        
    except Exception as e:
        # 保存错误日志
        with open(log_file, "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout_capture.getvalue())
            f.write("\n=== STDERR ===\n")
            f.write(stderr_capture.getvalue() + f"\nException: {str(e)}")
        
        return False


def main():
    """命令行接口 - 支持单模型和批量测试"""
    parser = argparse.ArgumentParser(description="Test device performance with fixed inputs")
    
    # 原有接口参数（向后兼容）
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to model directory (legacy mode)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=False,
        help="Result directory (legacy mode)"
    )
    
    # 新接口参数
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=False,
        help="Benchmark directory containing generated inputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
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
    
    # 判断运行模式
    if args.benchmark_path and args.device:
        # 新接口模式 - 批量测试
        ensure_dir(args.benchmark_path)
        
        # 查找所有需要测试的模型
        model_paths = find_model_paths(args.benchmark_path)
        
        if not model_paths:
            print(f"Error: No models found in benchmark path: {args.benchmark_path}")
            print("Please run example_inputs_generator first to generate input files.")
            return 1
        
        print(f"Found {len(model_paths)} models to test:")
        for model_path in model_paths:
            print(f"  - {model_path}")
        
        # 对每个模型运行测试
        success_count = 0
        for model_rel_path in model_paths:
            print(f"\nTesting model: {model_rel_path}")
            
            success = run_single_device_test(
                model_rel_path,
                args.benchmark_path,
                args.compiler,
                args.device,
                args.warmup,
                args.trials,
            )
            
            if success:
                success_count += 1
                print(f"✓ Test completed successfully for {model_rel_path}")
            else:
                print(f"✗ Test failed for {model_rel_path}")
        
        print(f"\nSummary: {success_count}/{len(model_paths)} models completed successfully")
        return 0 if success_count == len(model_paths) else 1
        
    elif args.model_path and args.result_dir:
        # 原有接口模式 - 单模型测试（向后兼容）
        runner = FixedInputsDeviceRunner(
            model_path=args.model_path,
            compiler=args.compiler or "cinn",
            device=args.device or "cuda",
            warmup=args.warmup or 5,
            trials=args.trials or 10
        )
        success = runner.run(args.result_dir)
        return 0 if success else 1
        
    else:
        parser.error("Either use --benchmark-path and --device for batch testing, or --model-path and --result-dir for single model testing")


if __name__ == "__main__":
    main()