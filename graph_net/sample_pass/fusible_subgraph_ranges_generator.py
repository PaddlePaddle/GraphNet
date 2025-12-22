from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from pathlib import Path
import json
from itertools import groupby


class FusibleSubgraphRangesGenerator(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        input_json_file_name: str,
        resume: bool = False,
        limits_handled_models: int = None,
        output_json_file_name: str = "fusible_subgraph_ranges.json",
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        file_name = self.config["output_json_file_name"]
        return self.naive_sample_handled(rel_model_path, search_file_name=file_name)

    def resume(self, rel_model_path: str):
        analyzer = self._make_analyzer(rel_model_path)
        output_obj = analyzer.analyze()
        self._save_output(rel_model_path, output_obj)

    def _save_output(self, rel_model_path, output_obj):
        output_json = json.dumps(output_obj, indent=4)
        output_dir_path = Path(self.config["output_dir"]) / rel_model_path
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / self.config["output_json_file_name"]
        output_file_path.write_text(output_json)

    def _make_analyzer(self, rel_model_path: str):
        model_path = (
            Path(self.config["model_path_prefix"])
            / rel_model_path
            / self.config["input_json_file_name"]
        )
        json_ctx = self._make_json_ctx(model_path)
        return FusibleSubgraphRangesAnalyzer(
            num_subgraph_kernels_list=self._get_num_subgraph_kernels_list(json_ctx),
            num_subgraph_ops_list=self._get_num_subgraph_ops_list(json_ctx),
            start_offset_in_original_graph=self._get_start_offset_in_original_graph(
                json_ctx
            ),
        )

    def _get_start_offset_in_original_graph(self, json_ctx):
        return json_ctx["start_offset_in_original_graph"]

    def _get_num_subgraph_kernels_list(self, json_ctx):
        return json_ctx["num_subgraph_kernels"]

    def _get_num_subgraph_ops_list(self, json_ctx):
        return json_ctx["num_subgraph_ops"]

    def _make_json_ctx(self, model_path: Path):
        obj = json.loads(model_path.read_text())
        assert len(obj["num_subgraph_kernels"]) == len(obj["num_subgraph_ops"])
        return obj


class FusibleSubgraphRangesAnalyzer:
    def __init__(
        self,
        num_subgraph_kernels_list: list[int],
        num_subgraph_ops_list: list[int],
        start_offset_in_original_graph: int,
    ):
        assert len(num_subgraph_kernels_list) == len(num_subgraph_ops_list)
        self.num_subgraph_kernels_list = num_subgraph_kernels_list
        self.num_subgraph_ops_list = num_subgraph_ops_list
        self.start_offset_in_original_graph = start_offset_in_original_graph

    def analyze(self):
        num_kernels_and_num_ops_list: list[
            (int, list[int])
        ] = self._make_num_kernels_and_num_ops_list()
        num_kernels_and_num_ops_list = sorted(
            num_kernels_and_num_ops_list, key=lambda pair: pair[0]
        )
        num_ops_lists = [
            sorted(num_ops_list)
            for _, num_ops_list in num_kernels_and_num_ops_list
            if len(set(num_ops_list)) > 1
        ]
        fusible_subgraph_ranges = [
            (start, end)
            for num_ops_list in num_ops_lists
            for start in [num_ops_list[0] - 1]
            for end in [num_ops_list[-1]]
        ]
        # sorted by `start`
        fusible_subgraph_ranges = sorted(
            fusible_subgraph_ranges, key=lambda pair: pair[0]
        )
        # remove shadowed
        fusible_subgraph_ranges = [
            fusible_subgraph_ranges[i]
            for i in range(len(fusible_subgraph_ranges))
            if i == 0
            or (fusible_subgraph_ranges[i][0] >= fusible_subgraph_ranges[i - 1][1])
        ]
        return fusible_subgraph_ranges

    def _make_num_kernels_and_num_ops_list(self):
        num_kernels_and_num_ops = zip(
            self.num_subgraph_kernels_list,
            self.num_subgraph_ops_list,
        )

        def get_num_kernels(pair):
            return pair[0]

        num_kernels_and_num_ops = sorted(num_kernels_and_num_ops, key=get_num_kernels)
        grouped_num_kernels_and_num_ops = groupby(
            num_kernels_and_num_ops, key=get_num_kernels
        )
        num_kernels_and_num_ops_list = [
            (num_kernels, [num_ops for _, num_ops in group])
            for num_kernels, group in grouped_num_kernels_and_num_ops
        ]
        return num_kernels_and_num_ops_list
