import os
import subprocess
import logging
import sys

class SubprocessExtractor:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.logger = logging.getLogger("SubprocessExtractor")

    def extract(self, script_path: str, model_id: str) -> str:
        """
        Run the extraction script in a subprocess.
        """
        # Define where the output is expected to be
        # The script template defines it as {model_dir}/extracted_sample
        model_dir = os.path.dirname(script_path)
        expected_output_dir = os.path.join(model_dir, "extracted_sample")
        
        cmd = [sys.executable, script_path]
        
        self.logger.info(f"Executing: {' '.join(cmd)}")
        
        # Determine cwd (should be GraphNet root to allow imports)
        # Assuming self.workspace/GraphNet/graph_net/agent/../../..
        # But here we assume user sets up PYTHONPATH or we run from root.
        # We try to find 'graph_net' package root.
        # graph_net is in D:\playground\task_10_agent\GraphNet
        # We need to add D:\playground\task_10_agent\GraphNet to PYTHONPATH
        
        # Heuristic: assume this file is in graph_net/agent/extractor.py
        # root is 3 levels up
        current_file = os.path.abspath(__file__)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + root_dir
        
        try:
            result = subprocess.run(
                cmd,
                cwd=root_dir, # Run from root
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info("Extraction script finished successfully.")
            self.logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error("Extraction script failed.")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Extraction failed for {model_id}")

        if not os.path.exists(expected_output_dir):
            raise FileNotFoundError(f"Expected output directory {expected_output_dir} was not created.")
            
        return expected_output_dir
