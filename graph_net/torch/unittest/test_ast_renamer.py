import unittest
import shutil
import tempfile
import textwrap
from pathlib import Path

from graph_net.sample_pass.ast_graph_variable_renamer import AstGraphVariableRenamer
from graph_net.tensor_meta import TensorMeta


class TestAstGraphVariableRenamerProduction(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(tempfile.mkdtemp())
        self.src_repo = self.test_root / "src_repo"
        self.dst_workspace = self.test_root / "workspace"
        self.src_repo.mkdir()
        self.dst_workspace.mkdir()

        self.rel_model_path = "samples/demo_model"
        self.full_src_path = self.src_repo / self.rel_model_path
        self.full_src_path.mkdir(parents=True)

        (self.full_src_path / "model.py").write_text(
            textwrap.dedent(
                """
            import torch
            class GraphModule(torch.nn.Module):
                def forward(self, x_data, L__self___weight):
                    res = x_data + L__self___weight
                    L__self___weight = None
                    return res
        """
            )
        )

        def save_meta_file(filename, var_name):
            meta = TensorMeta(
                name=var_name,
                shape=[1, 10],
                dtype="torch.float32",
                record_class_name="TensorMeta",
                original_name=None,
                device="cpu",
                mean=0.0,
                std=1.0,
                data=None,
                max_val=1.0,
                min_val=-1.0,
            )
            (self.full_src_path / filename).write_text(meta.serialize_to_py_str())

        save_meta_file("input_meta.py", "x_data")
        save_meta_file("weight_meta.py", "L__self___weight")

        import graph_net.torch.constraint_util as cu

        self.real_constraint_path = cu.__file__

    def tearDown(self):
        shutil.rmtree(self.test_root)

    def test_end_to_end_renaming_logic(self):
        handler_config = {
            "device": "cpu",
            "resume": True,
            "try_run": False,
            "model_path_prefix": str(self.src_repo),
            "output_dir": str(self.dst_workspace),
            "data_input_predicator_filepath": self.real_constraint_path,
            "data_input_predicator_class_name": "NaiveDataInputPredicator",
            "data_input_predicator_config": {},
            "model_runnable_predicator_filepath": self.real_constraint_path,
            "model_runnable_predicator_class_name": "ModelRunnablePredicator",
            "model_runnable_predicator_config": {},
        }

        renamer = AstGraphVariableRenamer(handler_config)
        renamer(self.rel_model_path)

        target_dir = self.dst_workspace / self.rel_model_path
        new_code = (target_dir / "model.py").read_text()
        self.assertIn("in_0", new_code, "x_data 应该被识别为 in_0")
        self.assertIn("w_0", new_code, "L__self___weight 应该被识别为 w_0")
        self.assertIn("tmp_0", new_code, "中间变量 res 应该被重命名为 tmp_0")
        self.assertNotIn("None", new_code, "权重清理语句应被 AST 转换器删除")

        new_weight_metas = TensorMeta.unserialize_from_py_file(
            str(target_dir / "weight_meta.py")
        )
        self.assertEqual(new_weight_metas[0].name, "w_0")
        self.assertEqual(new_weight_metas[0].original_name, "L__self___weight")

        self.assertTrue((target_dir / "graph_hash.txt").exists())
        hash_val = (target_dir / "graph_hash.txt").read_text()
        self.assertEqual(len(hash_val), 64, "Hash 应为标准的 SHA256 长度")

    def test_predicator_classification_diagnostic(self):
        from graph_net.imp_util import load_module

        module = load_module(self.real_constraint_path)
        pred_cls = getattr(module, "NaiveDataInputPredicator")
        predicator = pred_cls({})

        self.assertFalse(predicator(None, "L__self___weight"))
        self.assertTrue(predicator(None, "random_var_name"))


if __name__ == "__main__":
    unittest.main()
