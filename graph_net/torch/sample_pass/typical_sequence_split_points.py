import json
import os
from pathlib import Path
from typing import Dict, List

from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser
from graph_net.torch.rp_expr.rp_expr_util import (
    MakeNestedIndexRangeFromLetsListTokenRpExpr,
)
from graph_net.sample_pass.sample_pass import SamplePass


class TypicalSequenceSplitPointsGenerator(SamplePass):
    def __init__(self, config=None):
        super().__init__(config)
        self.inputs_seqs = []
        self.valid_models = []

    def declare_config(
        self,
        model_path_prefix: str,
        op_names_path_prefix: str,
        output_dir: str,
        resume: bool = False,
        window_size: int = 10,
        fold_policy: str = "default",
        fold_times: int = 0,
        min_seq_ops: int = 2,
        max_seq_ops: int = 64,
        subgraph_ranges_file_name: str = "subgraph_ranges.json",
        subgraph_ranges_key_name: str = "subgraph_ranges",
        subgraph_ranges_json: str = "subgraph_ranges.json",
        output_json: str = "split_results.json",
        device: str = "cpu",
    ):
        pass

    def BEGIN(self, rel_model_paths: List[str]):
        self._print_configuration()

    def __call__(self, rel_model_path: str):
        assert os.path.realpath(self.config["model_path_prefix"]) != os.path.realpath(
            self.config["output_dir"]
        )
        if self.config["resume"] and self.sample_handled(rel_model_path):
            return

        txt_path = (
            Path(self.config["op_names_path_prefix"]) / rel_model_path / "op_names.txt"
        )
        if not txt_path.exists():
            print(f"[Warning] File not found: {txt_path}")
            return

        print(f"[TypicalSequenceSplitPointsGenerator] Processing {rel_model_path}")
        seq = self._load_op_names_from_file(txt_path)
        if not seq:
            print(f"[Warning] Empty sequence in: {txt_path}")
            return

        self.inputs_seqs.append(seq)
        self.valid_models.append((Path(rel_model_path).name, rel_model_path))
        self._save_placeholder_result(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        return self.naive_sample_handled(
            rel_model_path, search_file_name=self.config["subgraph_ranges_file_name"]
        )

    def naive_sample_handled(self, rel_model_path: str, search_file_name: str) -> bool:
        dst_model_path = Path(self.config["output_dir"]) / rel_model_path
        if not dst_model_path.exists():
            return False
        num_model_py_files = len(list(dst_model_path.rglob(search_file_name)))
        assert num_model_py_files <= 1
        return num_model_py_files == 1

    def END(self, rel_model_paths: list[str]):
        if not self.inputs_seqs:
            print(
                "[TypicalSequenceSplitPointsGenerator] No valid operator sequences collected"
            )
            return

        print(
            f"[TypicalSequenceSplitPointsGenerator] END: Starting global RP-Expr analysis for {len(self.inputs_seqs)} models"
        )

        try:
            (
                split_positions_json,
                subgraph_ranges_json,
            ) = self._perform_rp_expr_analysis()
            self._save_global_results(split_positions_json, subgraph_ranges_json)
            self._save_individual_results(subgraph_ranges_json)
            self._print_analysis_summary(subgraph_ranges_json)
            print(
                f"[TypicalSequenceSplitPointsGenerator] END: Successfully analyzed {len(self.valid_models)} models"
            )
        except Exception as e:
            print(f"[TypicalSequenceSplitPointsGenerator] Error during analysis: {e}")
            raise

    def _print_configuration(self):
        print("-" * 50)
        print("RP-Expr Analysis Configuration:")
        print(f"  Window Size: {self.config['window_size']}")
        print(f"  Fold Policy: {self.config['fold_policy']}")
        print(f"  Fold Times: {self.config['fold_times']}")
        print(f"  Min Sequence Ops: {self.config['min_seq_ops']}")
        print(f"  Max Sequence Ops: {self.config['max_seq_ops']}")
        print(f"  Output Dir: {self.config['output_dir']}")
        print(f"  Resume Mode: {self.config['resume']}")
        print("-" * 50)

    def _perform_rp_expr_analysis(self):
        inputs_seqs = self.inputs_seqs
        valid_models = self.valid_models
        if not inputs_seqs:
            return {}, {}

        assert len(inputs_seqs) > 0
        rp_parser = RpExprParser(
            window_size=self.config["window_size"],
            fold_policy=self.config["fold_policy"],
            fold_times=self.config["fold_times"],
        )
        rp_expr, token_id2primitive_id = rp_parser(inputs_seqs)
        trees = MakeNestedIndexRangeFromLetsListTokenRpExpr(rp_expr)
        num_primitives = len(token_id2primitive_id)
        symbol_map = dict(zip(rp_expr.symbol_token_ids, rp_expr.symbol_token_tensors))
        token2len = self._calculate_token_lengths(rp_expr, num_primitives, symbol_map)
        split_positions_json = {}
        subgraph_ranges_json = {}
        for i, (model_name, original_path) in enumerate(valid_models):
            if i >= len(rp_expr.body_rp_expr):
                break

            tree = trees[i]
            target_body_tensor = rp_expr.body_rp_expr[i]
            seq_tokens = target_body_tensor.tolist()
            full_model_ops = []
            for t in seq_tokens:
                full_model_ops.extend(
                    self._resolve_token_to_ops(
                        t, num_primitives, token_id2primitive_id, symbol_map
                    )
                )

            total_len = sum(token2len.get(t, 1) for t in seq_tokens)
            subgraph_ranges = list(
                tree.FilterSubTreeRangeBySize(
                    self.config["min_seq_ops"], self.config["max_seq_ops"]
                )
            )
            sorted_splits = sorted(
                set(
                    split_pos
                    for start, end in subgraph_ranges
                    for split_pos in (start, end)
                    if end - start > 1
                )
            )
            self._print_analysis(
                model_name, str(original_path), sorted_splits, total_len, full_model_ops
            )
            split_positions_json[str(original_path)] = {
                "model_name": model_name,
                "split_positions": sorted_splits,
                "total_length": total_len,
            }
            sorted_subgraph_ranges = sorted(
                set((start, end) for start, end in subgraph_ranges if end - start > 1)
            )
            sorted_subgraph_ranges = [
                sorted_subgraph_ranges[i]
                for i in range(len(sorted_subgraph_ranges))
                if i == 0
                or sorted_subgraph_ranges[i][0] >= sorted_subgraph_ranges[i - 1][1]
            ]
            subgraph_ranges_json[str(original_path)] = {
                "model_name": model_name,
                self.config["subgraph_ranges_key_name"]: sorted_subgraph_ranges,
                "total_length": total_len,
            }
        return split_positions_json, subgraph_ranges_json

    def _save_placeholder_result(self, rel_model_path: str):
        model_output_dir = Path(self.config["output_dir"]) / rel_model_path
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / self.config["subgraph_ranges_file_name"]
        placeholder_data = {
            "model_name": Path(rel_model_path).name,
            self.config["subgraph_ranges_key_name"]: [],
            "total_length": 0,
            "status": "pending_global_analysis",
        }
        with open(output_file, "w") as f:
            json.dump(placeholder_data, f, indent=4)

    def _save_global_results(
        self, split_positions_json: Dict, subgraph_ranges_json: Dict
    ):
        if self.config.get("output_json"):
            with open(self.config["output_json"], "w") as f:
                json.dump(split_positions_json, f, indent=4)
            print(
                f"[TypicalSequenceSplitPointsGenerator] Global split positions saved to {self.config['output_json']}"
            )

        if self.config.get("subgraph_ranges_json"):
            with open(self.config["subgraph_ranges_json"], "w") as f:
                json.dump(subgraph_ranges_json, f, indent=4)
            print(
                f"[TypicalSequenceSplitPointsGenerator] Global subgraph ranges saved to {self.config['subgraph_ranges_json']}"
            )

    def _save_individual_results(self, subgraph_ranges_json: Dict):
        success_count = 0
        for model_name, rel_model_path in self.valid_models:
            if str(rel_model_path) not in subgraph_ranges_json:
                continue

            model_data = subgraph_ranges_json[str(rel_model_path)]
            model_output_dir = Path(self.config["output_dir"]) / rel_model_path
            model_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = model_output_dir / self.config["subgraph_ranges_file_name"]
            try:
                with open(output_file, "w") as f:
                    result_data = {
                        self.config["subgraph_ranges_key_name"]: model_data[
                            "subgraph_ranges"
                        ]
                    }
                    json.dump(result_data, f, indent=4)
                success_count += 1
                print(
                    f"[TypicalSequenceSplitPointsGenerator] Individual results saved for {model_name}"
                )
            except Exception as e:
                print(
                    f"[TypicalSequenceSplitPointsGenerator] Failed to save results for {model_name}: {e}"
                )

        print(
            f"[TypicalSequenceSplitPointsGenerator] Successfully saved results for {success_count} models"
        )

    def _print_analysis_summary(self, subgraph_ranges_json: Dict):
        total_models = len(subgraph_ranges_json)
        total_subgraphs = sum(
            len(data[self.config["subgraph_ranges_key_name"]])
            for data in subgraph_ranges_json.values()
        )
        avg_subgraphs = total_subgraphs / total_models if total_models > 0 else 0
        print("=" * 60)
        print("Typical Sequence Split Analysis Summary")
        print(f"Total models analyzed: {total_models}")
        print(f"Total subgraphs generated: {total_subgraphs}")
        print(f"Average subgraphs per model: {avg_subgraphs:.1f}")
        print("=" * 60)

    def _print_analysis(self, name, path, splits, total_len, full_ops):
        print("=" * 60)
        print(f"Model: {name}")
        print(f"Path:  {path}")
        print(f"Splits: {splits}")
        print("-" * 60)
        last_split = 0
        for split in splits + [total_len]:
            segment_len = split - last_split
            start_safe = min(last_split, len(full_ops))
            end_safe = min(split, len(full_ops))
            segment_ops = full_ops[start_safe:end_safe]
            ops_display = str(segment_ops)
            if len(segment_ops) > 5:
                ops_display = f"[{segment_ops[0]}, ..., {segment_ops[-1]}]"
            print(
                f"Range [{last_split:3d}, {split:3d}), Len: {segment_len:3d} | Ops: {ops_display}"
            )
            last_split = split
        print("\n")

    def _resolve_token_to_ops(
        self, tid, num_primitives, token_id2primitive_id, symbol_map
    ) -> List[str]:
        if tid < num_primitives:
            return [token_id2primitive_id[tid]]
        if tid in symbol_map:
            sub_tokens = symbol_map[tid].tolist()
            ops = []
            for t in sub_tokens:
                ops.extend(
                    self._resolve_token_to_ops(
                        t, num_primitives, token_id2primitive_id, symbol_map
                    )
                )
            return ops
        return [f"Unknown({tid})"]

    def _load_op_names_from_file(self, txt_path: Path) -> List[str]:
        if not txt_path.exists():
            return []
        return [
            line.strip() for line in txt_path.read_text().split("\n") if line.strip()
        ]

    def _calculate_token_lengths(
        self, rp_expr, num_primitives, symbol_map
    ) -> Dict[int, int]:
        token2len = {}

        def get_len(tid):
            if tid in token2len:
                return token2len[tid]
            if tid < num_primitives:
                token2len[tid] = 1
                return 1
            if tid in symbol_map:
                sub_tokens = symbol_map[tid].tolist()
                length = sum(get_len(t) for t in sub_tokens)
                token2len[tid] = length
                return length
            token2len[tid] = 1
            return 1

        for sym_id in rp_expr.symbol_token_ids:
            get_len(sym_id)
        return token2len
