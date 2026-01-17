import unittest
import os
import shutil
import graph_net
from graph_net.fault_locator.graph_truncator import GraphTruncator


class TestGraphTruncator(unittest.TestCase):
    def setUp(self):
        self.graph_net_root = os.path.dirname(os.path.dirname(graph_net.__file__))
        self.output_base = "/tmp/workspace_graph_truncator"

        self.config = {
            "output_dir": self.output_base,
            "model_path_prefix": self.graph_net_root,
            "device": "cuda",
            "use_all_inputs": True,
        }

        if os.path.exists(self.output_base):
            shutil.rmtree(self.output_base)

        self.truncator = GraphTruncator(self.config)
        self.relative_model_path = "samples/timm/resnet18"
        self.truncate_size = 5

    def test_actual_run(self):
        print(
            f"\n[Test] Truncating {self.relative_model_path} to size {self.truncate_size}..."
        )

        # 1. Execute
        ret_prefix, ret_rel_path = self.truncator(
            self.relative_model_path, self.truncate_size
        )

        # 2. Debug: List what was actually created
        print(f"[Debug] Return Prefix: {ret_prefix}")
        if os.path.exists(self.output_base):
            print(f"[Debug] Directory tree at {self.output_base}:")
            for root, dirs, files in os.walk(self.output_base):
                for f in files:
                    print(f"  - {os.path.join(root, f)}")
        else:
            print(f"[Error] The output_base {self.output_base} does not exist at all!")

        # 3. Assertions
        expected_prefix = os.path.join(self.output_base, str(self.truncate_size))
        self.assertEqual(ret_prefix, expected_prefix)

        # Check for the expected model file/folder
        # Note: SubgraphGenerator might append suffixes or create folders
        output_path = os.path.join(ret_prefix, self.relative_model_path)
        self.assertTrue(
            os.path.exists(output_path), f"Expected output not found at: {output_path}"
        )


if __name__ == "__main__":
    unittest.main()
