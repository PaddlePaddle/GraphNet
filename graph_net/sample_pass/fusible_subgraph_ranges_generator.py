from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from pathlib import Path
import json
from itertools import groupby
from dataclasses import dataclass


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
        output_obj = {
            "subgraph_ranges": analyzer.analyze(),
        }
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
        analysis_ctx = self._make_analysis_ctx()
        num_kernels_and_num_ops_list = analysis_ctx.num_kernels_and_num_ops_list
        # The tail num_kernels equals the head num_kernels for each num_ops_list
        naive_proposal_fused_num_ops_lists = [
            sorted(set(num_ops_list))
            for _, num_ops_list in num_kernels_and_num_ops_list
            if len(set(num_ops_list)) > 1
        ]
        proposal_fused_num_ops_lists = self._merge_all_decreasing_num_ops_lists(
            analysis_ctx, naive_proposal_fused_num_ops_lists
        )
        return self._create_subgraph_ranges_from_proposal(
            analysis_ctx,
            proposal_fused_num_ops_lists,
        )

    def _merge_all_decreasing_num_ops_lists(self, analysis_ctx, num_ops_lists):
        dead_loop_detect_cnt = 0
        kLimit = 99999
        while True:
            last_len_num_ops_lists = len(num_ops_lists)
            num_ops_lists = self._merge_one_decreasing_num_ops_lists(
                analysis_ctx, num_ops_lists
            )
            assert last_len_num_ops_lists >= len(num_ops_lists)
            if last_len_num_ops_lists == len(num_ops_lists):
                break
            dead_loop_detect_cnt += 1
            assert dead_loop_detect_cnt < kLimit, f"{dead_loop_detect_cnt=}"
        return num_ops_lists

    def _merge_one_decreasing_num_ops_lists(self, analysis_ctx, num_ops_lists):
        merge_pos = self._detect_mergable_decreasing_position(
            analysis_ctx, num_ops_lists
        )
        if merge_pos is None:
            return num_ops_lists
        assert merge_pos >= 0
        assert merge_pos < len(num_ops_lists) - 1
        return [
            *num_ops_lists[:merge_pos],
            [*num_ops_lists[merge_pos], *num_ops_lists[merge_pos + 1]],
            *num_ops_lists[merge_pos + 2 :],
        ]

    def _detect_mergable_decreasing_position(self, analysis_ctx, num_ops_lists):
        def get_cur_tail_num_kernels(i):
            return analysis_ctx.num_kernels4num_ops(num_ops_lists[i][-1])

        def get_next_head_num_kernels(i):
            return analysis_ctx.num_kernels4num_ops(num_ops_lists[i + 1][0])

        for i in range(len(num_ops_lists) - 1):
            assert len(num_ops_lists[i]) > 1
            if get_cur_tail_num_kernels(i) >= get_next_head_num_kernels(i):
                return i
        return None

    def _create_subgraph_ranges_from_proposal(
        self, analysis_ctx, proposal_fused_num_ops_lists
    ):
        # filter valid num_ops_list

        def is_a_range(int_list):
            assert len(int_list) > 1
            return (int_list[-1] + 1) - int_list[0] == len(int_list)

        def have_any_increasing(num_ops_list: list[int]):
            for i, cur_num_ops in enumerate(num_ops_list):
                if i == 0:
                    continue
                cur_num_kernels = analysis_ctx.num_kernels4num_ops(cur_num_ops)
                last_num_kernels = analysis_ctx.num_kernels4num_ops(num_ops_list[i - 1])
                if cur_num_kernels > last_num_kernels:
                    return True
            return False

        def head_eq_tail(num_ops_list: list[int]):
            return analysis_ctx.num_kernels4num_ops(
                num_ops_list[0]
            ) == analysis_ctx.num_kernels4num_ops(num_ops_list[-1])

        def head_gt_tail(num_ops_list: list[int]):
            return analysis_ctx.num_kernels4num_ops(
                num_ops_list[0]
            ) > analysis_ctx.num_kernels4num_ops(num_ops_list[-1])

        def valid_fused_ops(num_ops_list: list[int]):
            if head_gt_tail(num_ops_list):
                return True
            if head_eq_tail(num_ops_list):
                return not have_any_increasing(num_ops_list)
            return False

        proposal_fused_num_ops_lists = [
            sorted(set(num_ops_list)) for num_ops_list in proposal_fused_num_ops_lists
        ]
        num_ops_lists = [
            num_ops_list
            for num_ops_list in proposal_fused_num_ops_lists
            if len(num_ops_list) > 1
            if is_a_range(num_ops_list)
            if valid_fused_ops(num_ops_list)
        ]
        offset = self.start_offset_in_original_graph
        fusible_subgraph_ranges = [
            (start, end)
            for num_ops_list in num_ops_lists
            for start in [num_ops_list[0] - 1 + offset]
            for end in [num_ops_list[-1] + offset]
        ]

        # sorted by `start`
        def range_sort_key(pair):
            start, end = pair
            # smaller `start` first
            # bigger `end` first
            return (start, -end)

        fusible_subgraph_ranges = sorted(fusible_subgraph_ranges, key=range_sort_key)
        # remove shadowed
        fusible_subgraph_ranges = [
            fusible_subgraph_ranges[i]
            for i in range(len(fusible_subgraph_ranges))
            if i == 0
            or (fusible_subgraph_ranges[i][0] >= fusible_subgraph_ranges[i - 1][1])
        ]
        return fusible_subgraph_ranges

    def _make_analysis_ctx(self):
        return AnalysisContext(
            num_kernels_and_num_ops_list=self._make_num_kernels_and_num_ops_list(),
            num_ops2num_kernels=self._make_num_ops2num_kernels(),
        )

    def _make_num_ops2num_kernels(self):
        return dict(zip(self.num_subgraph_ops_list, self.num_subgraph_kernels_list))

    def _make_num_kernels_and_num_ops_list(self):
        num_kernels_and_num_ops = zip(
            self.num_subgraph_kernels_list,
            self.num_subgraph_ops_list,
        )

        def get_num_kernels(pair):
            return pair[0]

        def get_num_ops(pair):
            return pair[1]

        num_kernels_and_num_ops = sorted(num_kernels_and_num_ops, key=get_num_ops)
        grouped_num_kernels_and_num_ops = groupby(
            num_kernels_and_num_ops, key=get_num_kernels
        )
        num_kernels_and_num_ops_list = [
            (num_kernels, [num_ops for _, num_ops in group])
            for num_kernels, group in grouped_num_kernels_and_num_ops
        ]
        return num_kernels_and_num_ops_list


@dataclass
class AnalysisContext:
    num_kernels_and_num_ops_list: list[(int, list[int])]
    num_ops2num_kernels: dict[int, int]

    def num_kernels4num_ops(self, num_ops: int):
        return self.num_ops2num_kernels[num_ops]
