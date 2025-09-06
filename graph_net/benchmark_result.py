import os
import sys
import json
import re


class BenchmarkResult:
    def __init__(self, args, framework, hardware, compile_framework_version):
        self.configuration = {
            "model_name": self.get_model_name(args),
            "subgraph_tag": self.get_subgraph_tag(args),
            "model_path": args.model_path,
            "device": args.device,
            "hardware": hardware,
            "framework": framework,
            "compiler": args.compiler,
            "compile_framework_version": compile_framework_version,
            "warmup": args.warmup,
            "trials": args.trials,
        }
        self.model_info = {
            "num_ops": -1,
            "input_dtypes": None,
            "param_dtypes": None,
        }
        self.correctness = {}
        self.performance = {
            "eager": None,
            "compiled": None,
            "speedup": {},
        }

        self.device = args.device

        self.eager_e2e_time_ms = -1
        self.compiled_e2e_time_ms = -1
        self.e2e_speedup = -1

        self.eager_gpu_time_ms = -1
        self.compiled_gpu_time_ms = -1
        self.gpu_speedup = -1

    def get_model_name(self, args):
        model_name = None
        with open(os.path.join(args.model_path, "graph_net.json"), "r") as f:
            data = json.load(f)
            model_name = data.get("model_name", None)

        if model_name is not None:
            fields = args.model_path.split(os.sep)
            pattern = rf"^subgraph(_\d+)?$"
            model_name = fields[-2] if re.match(pattern, fields[-1]) else fields[-1]
        return model_name

    def get_subgraph_tag(self, args):
        fields = args.model_path.split(os.sep)
        pattern = rf"^subgraph(_\d+)?$"
        return fields[-1] if re.match(pattern, fields[-1]) else ""

    def update_model_info(self, num_ops, input_dtypes, param_dtypes):
        self.model_info["num_ops"] = num_ops
        self.model_info["input_dtypes"] = input_dtypes
        self.model_info["param_dtypes"] = param_dtypes

    def update_corrrectness(self, key, cmp_ret):
        self.correctness[key] = cmp_ret

    def update_performance(self, eager_stats, compiled_stats):
        self.performance["eager"] = eager_stats
        self.performance["compiled"] = compiled_stats

        self.eager_e2e_time_ms = eager_stats.get("e2e", {}).get("mean", -1)
        self.compiled_e2e_time_ms = compiled_stats.get("e2e", {}).get("mean", -1)
        if self.eager_e2e_time_ms > 0 and self.compiled_e2e_time_ms > 0:
            self.e2e_speedup = self.eager_e2e_time_ms / self.compiled_e2e_time_ms
            self.performance["speedup"]["e2e"] = float(f"{self.e2e_speedup:.6g}")

        if "cuda" in self.device:
            self.eager_gpu_time_ms = eager_stats.get("gpu", {}).get("mean", -1)
            self.compiled_gpu_time_ms = compiled_stats.get("gpu", {}).get("mean", -1)
            if self.eager_gpu_time_ms > 0 and self.compiled_gpu_time_ms > 0:
                self.gpu_speedup = self.eager_gpu_time_ms / self.compiled_gpu_time_ms
            self.performance["speedup"]["gpu"] = float(f"{self.gpu_speedup:.6g}")

    def write_to_json(self, output_dir):
        assert output_dir is not None
        os.makedirs(output_dir, exist_ok=True)
        result_data = {
            "configuration": self.configuration,
            "model_info": self.model_info,
            "correctness": self.correctness,
            "performance": self.performance,
        }
        model_name = self.configuration["model_name"]
        subgraph_tag = self.configuration["subgraph_tag"]
        compiler_name = self.configuration["compiler"]
        file_path = os.path.join(
            output_dir, f"{model_name}_{subgraph_tag}_{compiler_name}.json"
        )
        with open(file_path, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"Result saved to {file_path}", file=sys.stderr)
        print(result_data)
