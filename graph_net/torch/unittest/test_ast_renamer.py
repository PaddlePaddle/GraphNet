import unittest
import shutil
from pathlib import Path

from graph_net.sample_pass.ast_graph_variable_renamer import AstGraphVariableRenamer


class TestAstGraphVariableRenamerProduction(unittest.TestCase):
    def test_end_to_end_renaming_logic(self):
        test_model_path_prefix = Path(__file__).parent / "workspace_test_ast_renamer"
        test_model_name = "test_sample_demo"
        test_ast_renamer_output_dir = test_model_path_prefix / "test_ast_renamer_output"
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
        output_model_path = target_dir / "model.py"
        self.assertTrue(
            output_model_path.exists(),
            f"failed to find any output in {output_model_path}",
        )
        expected_output_model_path = (
            test_model_path_prefix / test_model_name / "expected_output_model.py"
        )
        self.assertEqual(
            "".join(output_model_path.read_text().split()),
            "".join(expected_output_model_path.read_text().split()),
        )


if __name__ == "__main__":
    unittest.main()
