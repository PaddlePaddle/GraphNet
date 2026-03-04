import unittest
import os
import shutil
from graph_net.filter_fault_model_path import FaultModelPathFilter


class TestFaultModelPathFilter(unittest.TestCase):
    def setUp(self):
        # Locate the root of the project
        self.graph_net_root = "/workspace/GraphNet"

        # Paths based on your requirements
        self.log_file = os.path.join(
            self.graph_net_root,
            "graph_net/test/data_calculate_es_scores/evaluation.log",
        )
        self.output_txt = "/tmp/workspace_fault_filter/faulty_models.txt"

        # Clean up output directory
        if os.path.exists(os.path.dirname(self.output_txt)):
            shutil.rmtree(os.path.dirname(self.output_txt))
        os.makedirs(os.path.dirname(self.output_txt), exist_ok=True)

        # Initialize the filter with your requested config
        self.config = {
            "log_file_path": self.log_file,
            "output_txt_file_path": self.output_txt,
            "model_path_prefix": self.graph_net_root,
            "tolerance": 0,  # Default behavior
            "graph_net_log_prompt": "graph-net-test-compiler-log",  # Default behavior
        }
        self.filter_op = FaultModelPathFilter(self.config)

    def test_filter_execution(self):
        """
        Verify that the filter correctly parses the log and writes relative paths.
        """
        print(f"\n[Test] Filtering log: {self.log_file}")

        # Execute the filter
        self.filter_op()

        # 1. Verify the output file exists
        self.assertTrue(
            os.path.exists(self.output_txt), "Output text file was not created."
        )

        # 2. Read the results and verify content
        with open(self.output_txt, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[Debug] Found {len(lines)} faulty models in output.")
        for line in lines:
            print(f"  - {line}")

        # 3. Basic Validation
        # Check that paths are indeed relative (should not start with /workspace/GraphNet)
        for path in lines:
            self.assertFalse(
                path.startswith(self.graph_net_root),
                f"Path '{path}' should be relative to prefix.",
            )
            # Check for a sample expected path segment if you know one from evaluation.log
            # e.g., self.assertIn("samples/ultralytics/yolov3-tinyu", path)


if __name__ == "__main__":
    unittest.main()
