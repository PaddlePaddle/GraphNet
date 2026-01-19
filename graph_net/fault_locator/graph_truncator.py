import os
import glob
import shutil
import sys
import json
import base64
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from graph_net.declare_config_mixin import DeclareConfigMixin


class GraphTruncator(DeclareConfigMixin):
    """
    Functor responsible for physical graph truncation by invoking SubgraphGenerator.
    It slices the model from operator 0 to truncate_size.
    """

    def __init__(self, config=None):
        self.init_config(config)

    def declare_config(
        self,
        output_dir: str,
        model_path_prefix: str,
        chain_style: bool = False,
        use_all_inputs: bool = False,
        device: str = "auto",
    ):
        """
        Declares user-facing parameters.
        Internal framework flags are hidden for a cleaner AFL interface.
        """
        pass

    def __call__(self, relative_model_path: str, truncate_size: int) -> tuple[str, str]:
        """
        Orchestrates the truncation process.
        Returns: (ret_model_path_prefix, relative_model_path)
        """
        output_prefix = str(Path(self.config["output_dir"]) / str(truncate_size))

        with TemporaryDirectory() as temp_root:
            # 1. Setup subgraph ranges and sample list inside the temp_root
            sample_list_path = os.path.join(temp_root, "sample_list.txt")

            self._prepare_subgraph_ranges(temp_root, relative_model_path, truncate_size)
            self._prepare_sample_list(sample_list_path, relative_model_path)

            # 2. Execute the physical slice
            config_b64 = self._build_encoded_config(temp_root, output_prefix)
            self._run_subgraph_generator(sample_list_path, config_b64)

            # Post-processing: Move data from _decomposed/... to the target directory
            self._restructure_output(output_prefix, relative_model_path)

        return output_prefix, relative_model_path

    def _restructure_output(self, output_prefix: str, rel_path: str):
        """
        Moves files from the internal '_decomposed' directory to the base relative path.
        From: {output_prefix}/{rel_path}/_decomposed/{subfolder}/
        To:   {output_prefix}/{rel_path}/
        """
        target_dir = os.path.join(output_prefix, rel_path)
        decomposed_root = os.path.join(target_dir, "_decomposed")

        if not os.path.exists(decomposed_root):
            return

        # Use glob to find the nested subfolder (e.g., resnet18_start0_end5_0)
        subfolders = glob.glob(os.path.join(decomposed_root, "*"))
        if not subfolders:
            return

        # Assuming SubgraphGenerator generates one primary subfolder for our range
        source_dir = subfolders[0]

        # Move all files from the subfolder up to the target_dir
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(target_dir, item)
            # Handle potential overwrites if any
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)

        # Cleanup: Remove the now-empty _decomposed directory tree
        shutil.rmtree(decomposed_root)

    def _prepare_subgraph_ranges(self, root: str, rel_path: str, size: int):
        """
        Creates the nested directory structure and range JSON file
        specifically required by the SubgraphGenerator's lookup logic.
        """
        # SubgraphGenerator expects root/{relative_model_path}/subgraph_ranges.json
        range_dir = os.path.join(root, rel_path)
        os.makedirs(range_dir, exist_ok=True)

        range_data = {"subgraph_ranges": [[0, size]]}
        json_path = os.path.join(range_dir, "subgraph_ranges.json")

        with open(json_path, "w") as f:
            json.dump(range_data, f)

    def _prepare_sample_list(self, path: str, content: str):
        """Writes the target model path to the temporary list file within temp_root."""
        with open(path, "w") as f:
            f.write(content)

    def _build_encoded_config(self, temp_root: str, output_prefix: str) -> str:
        """Assembles the internal SubgraphGenerator config and encodes it to Base64."""
        generator_config = self.config.copy()
        generator_config.update(
            {
                "output_dir": output_prefix,
                "subgraph_ranges_json_root": temp_root,
                "subgraph_ranges_json_file_name": "subgraph_ranges.json",
                "subgraph_ranges_json_key": "subgraph_ranges",
                # Enforced internal defaults
                "resume": False,
                "limits_handled_models": None,
                "group_head_and_tail": False,
            }
        )
        config_json = json.dumps(generator_config)
        return base64.b64encode(config_json.encode("utf-8")).decode("utf-8")

    def _run_subgraph_generator(self, sample_list: str, config_b64: str):
        """Invokes the external graph_net.apply_sample_pass module via subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "graph_net.apply_sample_pass",
            "--model-path-list",
            sample_list,
            "--sample-pass-file-path",
            self._get_script_path(),
            "--sample-pass-class-name",
            "SubgraphGenerator",
            "--sample-pass-config",
            config_b64,
        ]

        # Capture the output so we can see it during testing
        process = subprocess.run(cmd, capture_output=True, text=True)

        # ALWAYS print output for debugging until the pipeline is stable
        if process.stdout:
            print(f"\n[Subprocess STDOUT]\n{process.stdout}")
        if process.stderr:
            print(f"\n[Subprocess STDERR]\n{process.stderr}")

        if process.returncode != 0:
            raise RuntimeError(f"Truncation failed with exit code {process.returncode}")

    def _get_script_path(self) -> str:
        """Dynamically locates the SubgraphGenerator script path."""
        import graph_net

        root = os.path.dirname(os.path.dirname(graph_net.__file__))
        return os.path.join(root, "graph_net/torch/sample_pass/subgraph_generator.py")
