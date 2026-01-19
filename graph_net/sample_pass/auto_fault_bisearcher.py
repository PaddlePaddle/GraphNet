import os
import graph_net
from pathlib import Path
from typing import List, Tuple
from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.imp_util import load_module
from graph_net.fault_locator.bi_search import bi_search

# Determine the base directory of the graph_net package
GN_PATH = os.path.dirname(graph_net.__file__)


class AutoFaultBisearcher(SamplePass):
    """
    SamplePass to perform fault localization via binary search.
    It records the search history [truncate_size, has_fault] into a specified file.
    """

    def __init__(self, config=None):
        # 1. Initialize configuration and the base class
        super().__init__(config=config)

        # 2. Instantiate all functional components (functors) during initialization
        # These instances are reused for every model in the list
        self.truncator = self._instantiate_component("truncator")
        self.evaluator = self._instantiate_component("evaluator")
        self.es_scores_calculator = self._instantiate_component("es_scores_calculator")
        self.predicator = self._instantiate_component("predicator")
        self.stoper = self._instantiate_component("stoper")

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        output_file_name: str = "truncate_size_has_fault.txt",
        # Configs for dynamic loading of Evaluator
        evaluator_file_path: str = None,
        evaluator_class_name: str = None,
        evaluator_config: dict = None,
        # Configs for dynamic loading of ESScoresCalculator
        es_scores_calculator_file_path: str = f"{GN_PATH}/fault_locator/calculate_es_scores.py",
        es_scores_calculator_class_name: str = "ESScoresCalculator",
        es_scores_calculator_config: dict = None,
        # Configs for dynamic loading of Truncator
        truncator_file_path: str = f"{GN_PATH}/fault_locator/graph_truncator.py",
        truncator_class_name: str = "GraphTruncator",
        truncator_config: dict = None,
        # Configs for dynamic loading of Predicator
        predicator_file_path: str = f"{GN_PATH}/fault_locator/fault_detector.py",
        predicator_class_name: str = "FaultDetector",
        predicator_config: dict = None,
        # Configs for dynamic loading of Stoper
        stoper_file_path: str = f"{GN_PATH}/fault_locator/terminator.py",
        stoper_class_name: str = "BiserachTerminator",
        stoper_config: dict = None,
        # Global search parameters
        tolerance: int = 0,
    ):
        """
        Defines the configuration schema for the binary search pass.
        """
        pass

    def _instantiate_component(self, key_prefix: str):
        """
        Helper method to dynamically load a class from a file and
        instantiate it with its specific configuration.
        """
        file_path = self.config[f"{key_prefix}_file_path"]
        class_name = self.config[f"{key_prefix}_class_name"]
        # Default to an empty dictionary if no config is provided for the component
        comp_config = self.config.get(f"{key_prefix}_config") or {}

        module = load_module(file_path, name=f"dynamic_{key_prefix}")
        cls = getattr(module, class_name)
        return cls(comp_config)

    def __call__(self, rel_model_path: str):
        """
        Executed for each relative model path provided in the model-path-list.
        """
        # 2. Invoke the core binary search algorithm
        # history type: list[tuple[int, bool]]
        history: List[Tuple[int, bool]] = bi_search(
            relative_model_path=rel_model_path,
            truncator=self.truncator,
            evaluator=self.evaluator,
            es_scores_calculator=self.es_scores_calculator,
            predicator=self.predicator,
            stoper=self.stoper,
            tolerance=self.config.get("tolerance", 0),
        )

        # 3. Handle result persistence
        # Target path format: {output_dir}/{rel_model_path}/{output_file_name}
        file_name = self.config.get("output_file_name", "truncate_size_has_fault.txt")
        output_base = Path(self.config["output_dir"]) / rel_model_path

        # Ensure the directory exists (creates parents recursively if needed)
        output_base.mkdir(parents=True, exist_ok=True)

        result_file = output_base / file_name

        # Write history entries in the format: {truncate_size} {has_fault}
        with result_file.open("w", encoding="utf-8") as f:
            for trunc_size, has_fault in history:
                f.write(f"{trunc_size} {has_fault}\n")

        print(
            f"[AutoFault] Search history for {rel_model_path} saved to: {result_file}"
        )
        return history
