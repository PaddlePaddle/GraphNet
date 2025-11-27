import os
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
        self.name = name
        self.dynamic = dynamic
        self.input_spec = input_spec
        self.config = self.make_config(**config)

    def make_config(
        self,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        output_dir="./tmp/naive_decomposer_dir",
    ):
        for pos in split_positions:
            assert isinstance(
                pos, int
            ), f"split_positions should be list of int, {split_positions=}"
        return {
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "output_dir": output_dir,
        }

    def __call__(self, **input_dict):
        extracted_model = self.get_naive_decomposer_extractor()(**input_dict)
        return extracted_model

    def get_naive_decomposer_extractor(self):
        return NaiveDecomposerExtractor(self)


class NaiveDecomposerExtractor:
    def __init__(self, parent_graph_extractor):
        super().__init__()
        self.parent_graph_extractor = parent_graph_extractor
        self.extracted = False
        self.builtin_extractor = BuiltinGraphExtractor(
            model=parent_graph_extractor.model,
            name=parent_graph_extractor.name,
            dynamic=parent_graph_extractor.dynamic,
            input_spec=parent_graph_extractor.input_spec,
            workspace_path=self.parent_graph_extractor.config["output_dir"],
        )
        self.split_positions = self.parent_graph_extractor.config["split_positions"]
        self.post_process = self.make_post_process(self.parent_graph_extractor.config)

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
            model_dump_path, split_positions=self.split_positions
        )

        # 3. Save to model_path
        self.subgraph_path_list = []
        model_path = os.path.join(
            self.builtin_extractor.workspace_path, self.builtin_extractor.name
        )
        for (
            subgraph_idx,
            samples,
        ) in self.builtin_extractor.subgraph_idx2samples.items():
            for seq_idx in range(len(samples)):
                if (
                    self.builtin_extractor.num_samples_of_all_subgraphs == 1
                    and len(samples) == 1
                ):
                    subgraph_path = model_path
                elif len(samples) == 1:
                    subgraph_path = os.path.join(model_path, f"subgraph_{subgraph_idx}")
                else:
                    subgraph_path = os.path.join(
                        model_path, f"subgraph_{subgraph_idx}_{seq_idx}"
                    )
                self.subgraph_path_list.append(subgraph_path)
                self.builtin_extractor.write_sample_to_file(
                    subgraph_path, samples[seq_idx]
                )
        print(
            f"Graph and tensors for '{self.builtin_extractor.name}' extracted successfully to: {model_path}"
        )
        return static_model

    def __call__(self, **input_dict):
        extracted_model = None
        if not self.extracted:
            extracted_model = self.do_extract(**input_dict)
            self.extracted = True
        # if self.extracted:
        #    for subgraph_path in self.subgraph_path_list:
        #        self.post_process(subgraph_path)
        return extracted_model

    def make_post_process(self, config):
        return None
        # if config["post_process_path"] is None:
        #    return None
        # module = imp_util.load_module(config["post_process_path"])
        # return module.PostExtractProcess(config["post_process_config"])
