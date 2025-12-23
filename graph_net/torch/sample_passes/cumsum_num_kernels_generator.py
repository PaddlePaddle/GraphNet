from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.optional import Optional
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
from graph_net.torch.decompose_util import convert_to_submodules_graph
from graph_net.torch.count_kernels_util import CountNumKernelsNNModule
from graph_net.torch.fx_graph_module_util import (
    get_fx_graph_num_ops,
    get_torch_module_and_inputs,
)
from pathlib import Path
import json
import torch


class CumSumNumKernelsGenerator(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        resume: bool = False,
        start_offset_in_original_graph: int = 0,
        limits_handled_models: int = None,
        output_json_file_name: str = "cumsum_num_kernels.json",
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        file_name = self.config["output_json_file_name"]
        return self.naive_sample_handled(rel_model_path, search_file_name=file_name)

    def resume(self, rel_model_path: str):
        model_path = Path(self.config["model_path_prefix"]) / rel_model_path
        start_offset_in_original_graph = self.config["start_offset_in_original_graph"]
        analyzer = CumsumNumKernelsAnalyzer(model_path, start_offset_in_original_graph)
        cumsum_num_kernels = analyzer.analyze()
        cumsum_num_kernels_json = json.dumps(cumsum_num_kernels, indent=4)
        output_dir_path = Path(self.config["output_dir"]) / rel_model_path
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / self.config["output_json_file_name"]
        output_file_path.write_text(cumsum_num_kernels_json)


class CumsumNumKernelsAnalyzer:
    def __init__(self, model_path: Path, start_offset_in_original_graph: int):
        self.model_path = model_path
        self.start_offset_in_original_graph = start_offset_in_original_graph

    def analyze(self):
        triples = list(self._get_cumsum_num_kernels())
        data = {
            "start_offset_in_original_graph": self.start_offset_in_original_graph,
            "num_subgraph_kernels": [
                num_kernels for start, end, num_kernels in triples
            ],
            "num_subgraph_ops": [end for start, end, num_kernels in triples],
        }
        return data

    def _get_cumsum_num_kernels(self):
        model_path = str(self.model_path)
        module, inputs = get_torch_module_and_inputs(model_path, use_dummy_inputs=False)
        gm = parse_immutable_model_path_into_sole_graph_module(model_path)
        for start, end in self._get_ranges(gm):
            assert start == 0
            num_kernels = self._get_num_kernels_if_submodule_compiled(
                graph_module=gm,
                nn_module=module,
                inputs=inputs,
                submodule_start=start,
                submodule_end=end,
            )
            print(f"subgraph_range=[{start}, {end})\t{num_kernels=}")
            yield start, end, num_kernels

    def _get_num_kernels_if_submodule_compiled(
        self, graph_module, nn_module, inputs, submodule_start, submodule_end
    ):
        mut_opt_num_kernels = Optional(None)

        def compile_and_count_num_kernels(m, seq_no):
            return CountNumKernelsNNModule(m, mut_opt_num_kernels)

        rewrited_gm: torch.fx.GraphModule = convert_to_submodules_graph(
            graph_module,
            submodule_hook=compile_and_count_num_kernels,
            split_positions=[submodule_start, submodule_end],
            subgraph_ranges=[(submodule_start, submodule_end)],
            group_head_and_tail=False,
            chain_style=False,
        )
        rewrited_gm(*inputs)
        assert mut_opt_num_kernels.is_some()
        return mut_opt_num_kernels.unwrap()

    def _get_ranges(self, gm):
        num_ops = get_fx_graph_num_ops(gm)
        for i in range(num_ops):
            cum_num_ops = i + 1
            yield 0, cum_num_ops
