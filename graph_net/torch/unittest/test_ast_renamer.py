import unittest
import shutil
from pathlib import Path

from graph_net.sample_pass.ast_graph_variable_renamer import AstGraphVariableRenamer


class TestAstGraphVariableRenamerProduction(unittest.TestCase):
    def test_end_to_end_renaming_logic(self):
        test_model_path_prefix = Path(__file__).parent / "workspace_test_ast_renamer"
        test_model_name = "test_sample_demo"
        test_ast_renamer_output_dir = test_model_path_prefix / "test_ast_renamer_output"
        expected_output_model_path = test_model_path_prefix / "expected_output"
        if test_ast_renamer_output_dir.exists():
            shutil.rmtree(test_ast_renamer_output_dir)
        test_ast_renamer_output_dir.mkdir(parents=True)

        import graph_net.torch.constraint_util as cu

        real_constraint_path = cu.__file__

        handler_config = {
            "device": "gpu",
            "resume": True,
            "try_run": False,
            "model_path_prefix": str(test_model_path_prefix) + "/",
            "output_dir": str(test_ast_renamer_output_dir),
            "data_input_predicator_filepath": real_constraint_path,
            "data_input_predicator_class_name": "NaiveDataInputPredicator",
            "data_input_predicator_config": {},
            "model_runnable_predicator_filepath": real_constraint_path,
            "model_runnable_predicator_class_name": "ModelRunnablePredicator",
            "model_runnable_predicator_config": {},
        }

        renamer = AstGraphVariableRenamer(handler_config)
        renamer(test_model_name)

        target_dir = test_ast_renamer_output_dir / test_model_name
        self.compare_output_and_expected(target_dir, expected_output_model_path)

    def compare_output_and_expected(self, target_dir, expected_output_model_path):
        self.assertTrue(target_dir.exists(), "failed to find any output")
        for expected_file in expected_output_model_path.glob("*.py"):
            output_file = target_dir / expected_file.name
            self.assertTrue(
                output_file.exists(), f"{output_file.name} was not generated!"
            )
            self.assertEqual(
                "".join(output_file.read_text().split()),
                "".join(expected_file.read_text().split()),
                f"Content mismatch in {output_file.name}",
            )


if __name__ == "__main__":
    unittest.main()
