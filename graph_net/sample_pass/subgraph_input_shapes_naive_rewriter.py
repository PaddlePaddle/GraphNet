from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.tensor_meta import TensorMeta
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class SubgraphRelationship:
    original_graph_rel_model_path: str
    subgraph_rel_model_path: str
    range_start: int
    range_end: int


class SubgraphInputShapesNaiveRewriter(SamplePass, ResumableSamplePassMixin):
    """
    Rewrite shapes in weight_meta.py using sole relationship in subgraph_sources.json
    """

    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        shape_propagate_json_root: str,
        subgraph_input_producer_indexes_json_root: str,
        subgraph_sources_json_root: str,
        shape_propagate_json_file_name: str = "shape_prop.json",
        shape_propagate_json_key: str = "op_name_and_tensor_output_shape_list",
        subgraph_input_producer_indexes_json_file_name: str = "subgraph_input_producer_indexes.json",
        subgraph_input_producer_indexes_json_key: str = "subgraph_input_producer_indexes",
        subgraph_input_producer_indexes_json_rel_model_path_key: str = "subgraph_relative_model_paths",
        subgraph_sources_json_file_name: str = "subgraph_sources.json",
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        file_name = "weight_meta.py"
        return self.naive_sample_handled(rel_model_path, search_file_name=file_name)

    def resume(self, rel_model_path: str):
        mut_tensor_metas = self._get_old_tensor_metas(rel_model_path)
        infered_shapes = self._get_infered_shapes(rel_model_path)
        self._update_tensor_metas(mut_tensor_metas, infered_shapes)
        self._save_tensor_metas(rel_model_path, mut_tensor_metas)

    def _save_tensor_metas(self, rel_model_path: str, tensor_metas: list[TensorMeta]):
        model_path = self._get_dst_model_path(rel_model_path)
        data_tensor_metas = self._get_data_tensor_metas(model_path)
        assert len(data_tensor_metas) == 0
        weight_tensor_meta_file_path = str(model_path / "weight_meta.py")
        TensorMeta.save_tensor_metas(weight_tensor_meta_file_path, tensor_metas)

    def _update_tensor_metas(
        self, mut_tensor_metas: list[TensorMeta], infered_shapes: list[list[int]]
    ):
        assert len(mut_tensor_metas) == len(infered_shapes)
        for i in range(len(mut_tensor_metas)):
            self._update_tensor_shape(mut_tensor_metas[i], infered_shapes[i])

    def _update_tensor_shape(self, mut_tensor_meta, shape):
        mut_tensor_meta.update_shape_safely(shape)

    def _get_infered_shapes(self, rel_model_path: str):
        subgraph_relationship = self._get_subgraph_relationship(rel_model_path)
        original_graph_shape_propagation = self._get_original_graph_shape_propagation(
            subgraph_relationship
        )
        subgraph_input_producer_indexes = self._get_subgraph_input_producer_indexes(
            subgraph_relationship
        )
        return self._map_input_shapes(
            original_graph_shape_propagation, subgraph_input_producer_indexes
        )

    def _map_input_shapes(
        self,
        original_graph_shape_propagation: list[(str, list[int])],
        subgraph_input_producer_indexes: list[int],
    ) -> list[list[int]]:
        def get_shape(index):
            return original_graph_shape_propagation[index][1]

        return [get_shape(index) for index in subgraph_input_producer_indexes]

    def _get_subgraph_input_producer_indexes(
        self, subgraph_relationship: SubgraphRelationship
    ):
        json_obj = self._get_subgraph_input_producer_indexes_json_obj(
            subgraph_relationship.original_graph_rel_model_path
        )
        subgraph_input_producer_indexes = zip(
            json_obj[self.config["subgraph_input_producer_indexes_json_key"]],
            json_obj[
                self.config["subgraph_input_producer_indexes_json_rel_model_path_key"]
            ],
        )
        filtered = [
            range_and_indexes["input_producer_indexes"]
            for range_and_indexes, subgraph_rel_model_path in subgraph_input_producer_indexes
            if subgraph_rel_model_path == subgraph_relationship.subgraph_rel_model_path
            if range_and_indexes["range_start"] == subgraph_relationship.range_start
            if range_and_indexes["range_end"] == subgraph_relationship.range_end
        ]
        assert len(filtered) == 1, f"{filtered=}"
        return filtered[0]

    def _get_subgraph_input_producer_indexes_json_obj(
        self, original_graph_rel_model_path: str
    ):
        model_path = (
            Path(self.config["subgraph_input_producer_indexes_json_root"])
            / original_graph_rel_model_path
        )
        json_path = (
            model_path / self.config["subgraph_input_producer_indexes_json_file_name"]
        )
        return json.load(open(json_path))

    def _get_original_graph_shape_propagation(
        self, subgraph_relationship: SubgraphRelationship
    ):
        json_obj = self._get_shape_propagation_json_obj(
            subgraph_relationship.original_graph_rel_model_path
        )
        return json_obj[self._get_shape_propagation_json_key()]

    def _get_shape_propagation_json_key(self):
        return self.config["shape_propagate_json_key"]

    def _get_shape_propagation_json_obj(self, original_graph_rel_model_path):
        model_path = (
            Path(self.config["shape_propagate_json_root"])
            / original_graph_rel_model_path
        )
        json_file_path = model_path / self.config["shape_propagate_json_file_name"]
        return json.load(open(json_file_path))

    def _get_subgraph_relationship(self, rel_model_path: str):
        subgraph_sources = self._get_subgraph_sources(rel_model_path)
        sole_range_subgraph_source = self._get_sole_range_subgraph_source(
            subgraph_sources
        )
        return self._make_subgraph_relationship(
            sole_range_subgraph_source, rel_model_path
        )

    def _make_subgraph_relationship(self, sole_range_subgraph_source, rel_model_path):
        return SubgraphRelationship(
            original_graph_rel_model_path=sole_range_subgraph_source[0],
            subgraph_rel_model_path=rel_model_path,
            range_start=sole_range_subgraph_source[1][0],
            range_end=sole_range_subgraph_source[1][1],
        )

    def _get_sole_range_subgraph_source(
        self, subgraph_sources: dict[str, list[(int, int)]]
    ) -> (str, (int, int)):
        assert len(subgraph_sources) == 1
        rel_model_path, ranges = list(subgraph_sources.items())[0]
        assert len(ranges) == 1
        return rel_model_path, ranges[0]

    def _get_subgraph_sources(self, rel_model_path: str):
        model_path = Path(self.config["subgraph_sources_json_root"]) / rel_model_path
        json_file_path = model_path / self.config["subgraph_sources_json_file_name"]
        return json.load(open(json_file_path))

    def _get_old_tensor_metas(self, rel_model_path: str):
        model_path = self._get_src_model_path(rel_model_path)
        data_tensor_metas = self._get_data_tensor_metas(model_path)
        assert len(data_tensor_metas) == 0
        return self._get_weight_tensor_metas(model_path)

    def _get_weight_tensor_metas(self, model_path: Path):
        weight_tensor_meta_file_path = str(model_path / "weight_meta.py")
        return TensorMeta.unserialize_from_py_file(weight_tensor_meta_file_path)

    def _get_data_tensor_metas(self, model_path: Path):
        data_tensor_meta_file_path = model_path / "input_meta.py"
        if not data_tensor_meta_file_path.exists():
            return []
        return TensorMeta.unserialize_from_py_file(str(data_tensor_meta_file_path))

    def _get_src_model_path(self, rel_model_path: str):
        return Path(self.config["model_path_prefix"]) / rel_model_path

    def _get_dst_model_path(self, rel_model_path: str):
        model_path = Path(self.config["output_dir"]) / rel_model_path
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
