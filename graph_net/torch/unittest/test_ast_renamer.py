import unittest
import shutil
from pathlib import Path

from graph_net.sample_pass.ast_graph_variable_renamer import AstGraphVariableRenamer
from graph_net.tensor_meta import TensorMeta


class TestAstGraphVariableRenamerProduction(unittest.TestCase):
    def setUp(self):
        self.test_model_path_prefix = (
            Path(__file__).parent / "workspace_test_ast_renamer"
        )
        self.rel_model_path = "test_sample_demo"
        self.test_ast_renamer_output = (
            self.test_model_path_prefix / "test_ast_renamer_output"
        )
        if self.test_ast_renamer_output.exists():
            shutil.rmtree(self.test_ast_renamer_output)
        self.test_ast_renamer_output.mkdir(parents=True)

        def make_meta(filename, var_name):
            full_model_dir = self.test_model_path_prefix / self.rel_model_path
            meta_file = full_model_dir / filename
            if meta_file.exists():
                meta_file.unlink()
            meta = TensorMeta(
                name=var_name,
                shape=[1, 10],
                dtype="torch.float32",
                record_class_name="TensorMeta",
                original_name=None,
                device="gpu",
                mean=0.0,
                std=1.0,
                data=None,
                max_val=1.0,
                min_val=-1.0,
            )
            meta_file.write_text(meta.serialize_to_py_str())

        make_meta("input_meta.py", "x_data")
        make_meta("weight_meta.py", "L__self___weight")

        import graph_net.torch.constraint_util as cu

        self.real_constraint_path = cu.__file__

    def test_end_to_end_renaming_logic(self):
        handler_config = {
            "device": "gpu",
            "resume": True,
            "try_run": False,
            "model_path_prefix": str(self.test_model_path_prefix) + "/",
            "output_dir": str(self.test_ast_renamer_output),
            "data_input_predicator_filepath": self.real_constraint_path,
            "data_input_predicator_class_name": "NaiveDataInputPredicator",
            "data_input_predicator_config": {},
            "model_runnable_predicator_filepath": self.real_constraint_path,
            "model_runnable_predicator_class_name": "ModelRunnablePredicator",
            "model_runnable_predicator_config": {},
        }

        renamer = AstGraphVariableRenamer(handler_config)
        renamer(self.rel_model_path)

        target_dir = self.test_ast_renamer_output / self.rel_model_path
        output_model_path = target_dir / "model.py"
        self.assertTrue(
            output_model_path.exists(),
            f"failed to find any output in {output_model_path}",
        )
        expected_output_model_path = (
            self.test_model_path_prefix
            / self.rel_model_path
            / "expected_output_model.py"
        )
        output_model_code = output_model_path.read_text()
        expected_output_model_code = expected_output_model_path.read_text()

        def normalize_code(model_code):
            lines = [line.rstrip() for line in model_code.splitlines() if line.strip()]
            return "\n".join(lines)

        self.assertMultiLineEqual(
            normalize_code(output_model_code),
            normalize_code(expected_output_model_code),
            "output of renamer differs from expected!",
        )


if __name__ == "__main__":
    unittest.main()
