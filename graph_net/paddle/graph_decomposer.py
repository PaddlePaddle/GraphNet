import os
from typing import List
import paddle
from graph_net import imp_util
from graph_net.paddle.extractor import GraphExtractor as BuiltinGraphExtractor


class GraphExtractor:
    def __init__(
        self,
        config: dict,
        model,
        name,
        dynamic,
        input_spec=None,
    ):
        self.model = model
        self.name = name.replace("/", "_")
        self.dynamic = dynamic
        self.input_spec = input_spec
        self.config = self.make_config(**config)

    def make_config(
        self,
        split_positions=None,
        group_head_and_tail=False,
        chain_style=False,
        output_dir="./tmp/naive_decomposer_dir",
        post_extract_process_path=None,
        post_extract_process_class_name=None,
        post_extract_process_config=None,
    ):
        assert not chain_style, "chain_style=True is not supported now."
        if split_positions is not None:
            assert isinstance(
                split_positions, (tuple, list)
            ), f"split_positions is expected to be tuple or list, but recived {split_positions=}"
            for pos in split_positions:
                assert isinstance(
                    pos, int
                ), f"split_positions is expected to be tuple or list of int, but recived {split_positions=}"
        return {
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "output_dir": output_dir,
            "post_extract_process_path": post_extract_process_path,
            "post_extract_process_class_name": post_extract_process_class_name,
            "post_extract_process_config": post_extract_process_config,
        }

    def __call__(self, **input_dict):
        extracted_model = self.get_naive_decomposer_extractor()(**input_dict)
        return extracted_model

    def get_naive_decomposer_extractor(self):
        return NaiveDecomposerExtractor(
            config=self.config,
            parent_model=self.model,
            parent_model_name=self.name,
            parent_input_spec=self.input_spec,
        )


class NaiveDecomposerExtractor:
    def __init__(
        self,
        config: dict,
        parent_model: paddle.nn.Layer,
        parent_model_name: str,
        parent_input_spec: List[paddle.static.InputSpec],
    ):
        self.config = config
        self.extracted = False
        self.parent_model_path = os.path.dirname(parent_model.__graph_net_file_path__)
        self.builtin_extractor = BuiltinGraphExtractor(
            model=parent_model,
            name=parent_model_name,
            dynamic=False,
            input_spec=parent_input_spec,
            workspace_path=self.config["output_dir"],
        )
        self.split_positions = self.config["split_positions"]
        self.group_head_and_tail = self.config["group_head_and_tail"]
        self.post_extract_process = self.make_post_extract_process(self.config)

    def do_extract(self, **input_dict):
        # 1. Run the model to dump pir programs
        model_dump_path = os.path.join(
            self.builtin_extractor.dump_path, self.builtin_extractor.name
        )
        static_model = self.builtin_extractor.run_model_with_dump_enabled(
            model_dump_path, **input_dict
        )

        # 2. Convert pir programs to graphnet samples
        self.builtin_extractor.translate_pir_program_to_sample_codes(
            model_dump_path,
            split_positions=self.split_positions,
            group_head_and_tail=self.group_head_and_tail,
        )

        # 3. Save to model_path
        self.subgraph_path_list = []
        model_path = os.path.join(
            self.builtin_extractor.workspace_path, self.builtin_extractor.name
        )
        assert len(self.builtin_extractor.subgraph_idx2samples) == 1

        samples = self.builtin_extractor.subgraph_idx2samples[0]
        for seq_idx in range(len(samples)):
            subgraph_path = f"{model_path}_{seq_idx}"
            self.subgraph_path_list.append(subgraph_path)
            self.builtin_extractor.write_sample_to_file(subgraph_path, samples[seq_idx])
            print(f"Save to {subgraph_path}")
        return static_model

    def __call__(self, **input_dict):
        extracted_model = None
        if not self.extracted:
            extracted_model = self.do_extract(**input_dict)
            self.extracted = True

        for subgraph_path in self.subgraph_path_list:
            self._post_extract_process(subgraph_path)
        return extracted_model

    def _post_extract_process(self, subgraph_path):
        return self.post_extract_process(subgraph_path)

    def make_post_extract_process(self, config):
        if config.get("post_extract_process_path") is None:
            return lambda *args, **kwargs: None
        module = imp_util.load_module(config["post_extract_process_path"])
        cls = getattr(module, config["post_extract_process_class_name"])
        return cls(config["post_extract_process_config"], self.parent_model_path)
