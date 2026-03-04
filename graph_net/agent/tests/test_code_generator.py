"""Tests for code generation"""

import tempfile
import unittest
from pathlib import Path

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata
from graph_net.agent.code_generator.template_generator import TemplateCodeGenerator


class TestTemplateCodeGenerator(unittest.TestCase):
    """Test TemplateCodeGenerator"""

    def setUp(self):
        """Set up test environment"""
        self.generator = TemplateCodeGenerator()
        self.temp_dir = tempfile.mkdtemp()

    def test_generate_code(self):
        """Test code generation"""
        model_dir = Path(self.temp_dir) / "model"
        model_dir.mkdir()

        meta = ModelMetadata(
            model_id="bert-base-uncased",
            input_shapes={"input_ids": [1, 128]},
            input_dtypes={"input_ids": "int64"},
            model_type="bert",
        )

        output_dir = Path(self.temp_dir) / "output"
        script_path = self.generator.generate(model_dir, meta, output_dir)

        self.assertTrue(script_path.exists())
        self.assertEqual(script_path.name, "run_model.py")

        # Check code content
        code = script_path.read_text()
        self.assertIn("bert-base-uncased", code)
        self.assertIn("input_ids", code)
        self.assertIn("graph_net.torch.extract", code)

    def test_generate_model_loader(self):
        """Test model loader generation"""
        model_dir = Path(self.temp_dir) / "model"
        model_dir.mkdir()

        # Test BERT model
        meta_bert = ModelMetadata(
            model_id="bert-base-uncased",
            input_shapes={"input_ids": [1, 128]},
            input_dtypes={"input_ids": "int64"},
            model_type="bert",
        )
        load_code = self.generator._generate_model_loader(model_dir, meta_bert)
        self.assertIn("AutoModel.from_pretrained", load_code)

    def test_generate_input_code(self):
        """Test input code generation"""
        meta = ModelMetadata(
            model_id="test",
            input_shapes={
                "input_ids": [1, 128],
                "attention_mask": [1, 128],
            },
            input_dtypes={
                "input_ids": "int64",
                "attention_mask": "int64",
            },
        )

        input_code = self.generator._generate_input_code(meta)
        self.assertIn("input_ids", input_code)
        self.assertIn("attention_mask", input_code)
        self.assertIn("torch.randn", input_code)


if __name__ == "__main__":
    unittest.main()
