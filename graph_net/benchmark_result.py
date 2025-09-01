import os
import sys
import json
import re


class BenchmarkResult:
    def __init__(self, args, hardware, compile_framework_version):
        self.configuration = {
            "model_name": self.get_model_name(args),
            "device": args.device,
            "hardware": hardware,
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
            "speedup": None,
        }

    def get_model_name(self, args):
        fields = args.model_path.split(os.sep)

        pattern = rf"^subgraph(_\d+)?$"
        if re.match(pattern, fields[-1]):
            model_name = f"{fields[-2]}_{fields[-1]}"
        else:
            model_name = fields[-1]
        return model_name

    def update_model_info(self, num_ops, input_dtypes, param_dtypes):
        self.model_info["num_ops"] = num_ops
        self.model_info["input_dtypes"] = input_dtypes
        self.model_info["param_dtypes"] = param_dtypes

    def update_corrrectness(self, key, cmp_ret):
        self.correctness[key] = cmp_ret

    def update_performance(self, eager_time_ms, compiled_time_ms):
        self.performance["eager"] = eager_time_ms
        self.performance["compiled"] = compiled_time_ms
        if eager_time_ms > 0 and compiled_time_ms > 0:
            self.performance["speedup"] = eager_time_ms / compiled_time_ms
        return self.performance["speedup"]

    def write_to_json(self, output_dir):
        assert output_dir is not None
        os.makedirs(output_dir, exist_ok=True)
        result_data = {
            "configuration": self.configuration,
            "model_info": self.model_info,
            "correctness": self.correctness,
            "performance": {
                k: float(f"{v:.6f}") if isinstance(v, float) else v
                for k, v in self.performance.items()
            },
        }
        model_name = self.configuration["model_name"]
        compiler_name = self.configuration["compiler"]
        file_path = os.path.join(output_dir, f"{model_name}_{compiler_name}.json")
        with open(file_path, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"Result saved to {file_path}", file=sys.stderr)
        print(result_data)
