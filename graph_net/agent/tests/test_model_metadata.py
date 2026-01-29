"""Tests for ModelMetadata"""

import unittest

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata


class TestModelMetadata(unittest.TestCase):
    """Test ModelMetadata data class"""

    def test_basic_creation(self):
        """Test basic metadata creation"""
        meta = ModelMetadata(
            model_id="bert-base-uncased",
            input_shapes={"input_ids": [1, 128]},
            input_dtypes={"input_ids": "int64"},
        )
        self.assertEqual(meta.model_id, "bert-base-uncased")
        self.assertEqual(meta.input_shapes["input_ids"], [1, 128])
        self.assertEqual(meta.input_dtypes["input_ids"], "int64")

    def test_multiple_inputs(self):
        """Test metadata with multiple inputs"""
        meta = ModelMetadata(
            model_id="test-model",
            input_shapes={
                "input_ids": [1, 128],
                "attention_mask": [1, 128],
            },
            input_dtypes={
                "input_ids": "int64",
                "attention_mask": "int64",
            },
        )
        self.assertEqual(len(meta.input_shapes), 2)
        self.assertEqual(len(meta.input_dtypes), 2)

    def test_model_type(self):
        """Test metadata with model type"""
        meta = ModelMetadata(
            model_id="bert-base-uncased",
            input_shapes={"input_ids": [1, 128]},
            input_dtypes={"input_ids": "int64"},
            model_type="bert",
        )
        self.assertEqual(meta.model_type, "bert")

    def test_empty_input_shapes_raises_error(self):
        """Test that empty input_shapes raises error"""
        with self.assertRaises(ValueError):
            ModelMetadata(
                model_id="test",
                input_shapes={},
                input_dtypes={"input_ids": "int64"},
            )

    def test_empty_input_dtypes_raises_error(self):
        """Test that empty input_dtypes raises error"""
        with self.assertRaises(ValueError):
            ModelMetadata(
                model_id="test",
                input_shapes={"input_ids": [1, 128]},
                input_dtypes={},
            )

    def test_mismatched_keys_raises_error(self):
        """Test that mismatched keys raise error"""
        with self.assertRaises(ValueError):
            ModelMetadata(
                model_id="test",
                input_shapes={"input_ids": [1, 128]},
                input_dtypes={"attention_mask": "int64"},
            )


if __name__ == "__main__":
    unittest.main()
