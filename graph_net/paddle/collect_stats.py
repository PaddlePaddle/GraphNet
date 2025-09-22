import argparse
import os
import re
import sys
import math
import subprocess
from datetime import datetime

import paddle
from graph_net import collect_stats_util
from graph_net.paddle import utils


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def read_graph_source_and_tag(model_path):
    try:
        with open(os.path.join(model_path, "graph_net.json"), "r") as f:
            data = json.load(f)
            return data["source"], data["heuristic_tag"]
    except Exception:
        if "PaddleX" in model_path:
            return "PaddleX", "computer_vision"
        elif "PaddleNLP" in model_path:
            return "PaddleNLP", "nlp"
        elif "PaddleScience" in model_path:
            return "PaddleScience", "scientific_computing"
        else:
            return "unknown", "unknown"


def get_input_spec(model_path):
    inputs_params_list = utils.load_converted_list_from_text(f"{model_path}")
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


class ProgramAnalyzer:
    def __init__(self):
        self.op_stats = {}
        self.input_dict = {}
        self.num_ops = 0
        self.num_ops_misses_dtypes = 0
        self.is_complete = True

    def update_op_stats(self, op_name, op_dtype):
        if op_name is not None:
            dtype_str = str(op_dtype).replace("paddle.", "")
            if self.op_stats.get(op_name, None) is None:
                self.op_stats[op_name] = collect_stats_util.OpStat(
                    op_name, {dtype_str: 1}, 1
                )
            else:
                self.op_stats[op_name].op_dtypes[dtype_str] = (
                    self.op_stats[op_name].op_dtypes.get(dtype_str, 0) + 1
                )
                self.op_stats[op_name].count += 1

    def parse_pir_value_dtypes(self, type_str):
        short_form2dtype = {
            "f32": "float32",
            "f16": "float16",
            "bf16": "bfloat16",
            "i64": "int64",
        }
        # type_str: "vec[tensor<1x18x13x9xf32>,tensor<1x9x13x9xf32>]"
        matches = re.findall(r"tensor<([^>]+)>", type_str)
        dtype_strs = []
        for s in matches:
            parts = s.split("x")
            assert len(parts) > 0

            dtype = parts[-1].lower()
            dtype_strs.append(short_form2dtype[dtype])
        return dtype_strs

    def __call__(self, program):
        assert isinstance(program, paddle.base.libpaddle.pir.Program)

        self.op_stats = {}
        self.num_ops_misses_dtypes = 0
        self.num_ops = 0
        for block in program.blocks:
            for op in block.ops:
                op_name = None
                op_dtype = None
                if op.name() == "pd_op.data":
                    op_name = "data"
                    op_attrs = op.attrs()
                    op_dtype = op_attrs["dtype"]
                    self.input_dict[op_attrs["name"]] = {
                        "dtype": str(op_dtype).replace("paddle.", ""),
                        "shape": op_attrs["shape"],
                    }
                elif op.name().startswith("pd_op."):
                    self.num_ops += 1
                    op_name = op.name().replace("pd_op.", "")
                    try:
                        if len(op.results()) > 0:
                            out = op.results()[0]
                            if out.is_dense_tensor_type():
                                op_dtype = out.dtype
                            else:
                                # for paddle.base.libpaddle.pir.VectorType, but cannot be accurately determined
                                if op_name in [
                                    "split",
                                    "split_with_num",
                                    "meshgrid",
                                    "distribute_fpn_proposals",
                                ]:
                                    op_dtype = self.parse_pir_value_dtypes(
                                        str(out.type())
                                    )[0]
                                else:
                                    assert False, f"Unsupport op: {op}"
                    except Exception:
                        if self.num_ops_misses_dtypes == 0:
                            print(f"dtype inference failed for {op_name}")
                    if op_dtype is not None:
                        self.update_op_stats(op_name, op_dtype)
                    else:
                        self.num_ops_misses_dtypes += 1
                elif not op.name().startswith("builtin."):
                    assert False, f"Unrecognized op: {op}"

        if self.num_ops_misses_dtypes > 0:
            self.is_complete = False

    def summary(self):
        print(
            f"Totally {self.num_ops} operators, and {self.num_ops_misses_dtypes} operators failed to inference dtypes."
        )


def collect_op_stats(model, model_path):
    assert isinstance(model, paddle.nn.Layer), f"{type(model)=}"
    try:
        static_model = paddle.jit.to_static(
            model,
            input_spec=get_input_spec(model_path),
            full_graph=True,
            backend=None,
        )
        static_model.eval()
        program = static_model.forward.concrete_program.main_program

        program_analyzer = ProgramAnalyzer()
        program_analyzer(program)
        program_analyzer.summary()
        return program_analyzer
    except Exception:
        print("Failed with to_static")
        return None


def collect_model_stats(model_path, log_prompt):
    file_path = os.path.join(model_path, "model.py")
    model_class = collect_stats_util.load_class_from_file(file_path, "GraphModule")
    model = model_class()

    model_size = 0
    input_dtypes = {}
    param_dtypes = {}
    ops_count_dict = {}
    op_dtypes = {}

    program_analyzer = collect_op_stats(model, model_path)
    if program_analyzer is not None:
        for op_name, stat in sorted(program_analyzer.op_stats.items()):
            ops_count_dict[op_name] = stat.count
            for dtype_str, num in stat.op_dtypes.items():
                if dtype_str is not None and dtype_str != "None":
                    op_dtypes[dtype_str] = op_dtypes.get(dtype_str, 0) + num

        inputs_params = utils.load_converted_from_text(f"{model_path}")
        params = inputs_params["weight_info"]
        inputs = inputs_params["input_info"]

        for name, value in program_analyzer.input_dict.items():
            dtype_str = value["dtype"]
            if name in params.keys():
                param_numel = math.prod(value["shape"])
                model_size += param_numel
                param_dtypes[dtype_str] = param_dtypes.get(dtype_str, 0) + 1
            elif name in inputs.keys():
                input_dtypes[dtype_str] = input_dtypes.get(dtype_str, 0) + 1

    num_outputs = collect_stats_util.get_number_of_returns(
        file_path, "GraphModule", "forward"
    )
    num_ops = program_analyzer.num_ops if program_analyzer is not None else 0
    source, heuristic_tag = read_graph_source_and_tag(model_path)
    is_complete = (
        program_analyzer.is_complete if program_analyzer is not None else False
    )
    print(
        f"model_stats collection information: model_path={model_path}, method=to_static, is_ops_complete={is_complete}"
    )

    stats = collect_stats_util.ModelStats(
        model_path=model_path,
        num_inputs=sum(input_dtypes.values()),
        num_params=sum(param_dtypes.values()),
        num_outputs=num_outputs,
        num_ops=num_ops,
        model_size_in_billion=model_size / 1e9,
        input_dtypes=input_dtypes,
        param_dtypes=param_dtypes,
        op_dtypes=op_dtypes,
        ops=ops_count_dict,
        source=source,
        heuristic_tag=heuristic_tag,
    )
    collect_stats_util.print_model_stats(stats, log_prompt)


def main(args):
    if args.model_path is not None:
        assert os.path.isdir(args.model_path)
        assert is_single_model_dir(args.model_path)
        timestamp_sec = datetime.now().timestamp()
        dt = datetime.fromtimestamp(timestamp_sec)
        formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{formatted_dt}] Collect information for {args.model_path}")
        collect_model_stats(args.model_path, args.log_prompt)
    else:
        graph_net_samples_path = (
            (graph_net.paddle.samples_util.get_default_samples_directory())
            if args.graph_net_samples_path is None
            else args.graph_net_samples_path
        )

        i = 0
        for root, dirs, files in os.walk(graph_net_samples_path):
            if is_single_model_dir(root):
                print(f"[{i}] Collect information for {root}")
                cmd = [
                    "python",
                    "-m",
                    "graph_net.paddle.collect_stats",
                    f"--device={args.device}",
                    f"--model-path={root}",
                    f"--log-prompt={args.log_prompt}",
                ]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                )
                print(result.stdout)
                if result.returncode != 0:
                    print(result.stderr)
                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect stats for computation graph samples. return 0 if success"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing the compiler (e.g., 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Computation graph sample directory. e.g '../../paddle_samples/PaddleX/ResNet18'",
    )
    parser.add_argument(
        "--graph-net-samples-path",
        type=str,
        required=False,
        default=None,
        help="GraphNet samples directory. e.g '../../paddle_samples'",
    )
    parser.add_argument(
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-collect-stats-log",
        help="Log prompt for stats log filtering.",
    )
    args = parser.parse_args()
    main(args=args)
