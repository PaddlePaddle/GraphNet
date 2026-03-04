"""Tests for utility modules"""

import tempfile
import unittest

from graph_net.agent.utils.exceptions import (
    AgentError,
    ModelFetchError,
    AnalysisError,
    CodeGenError,
    ExtractionError,
    VerificationError,
)
from graph_net.agent.utils.workspace_manager import WorkspaceManager


class TestExceptions(unittest.TestCase):
    """Test exception classes"""

    def test_exception_hierarchy(self):
        """Test exception inheritance"""
        self.assertTrue(issubclass(ModelFetchError, AgentError))
        self.assertTrue(issubclass(AnalysisError, AgentError))
        self.assertTrue(issubclass(CodeGenError, AgentError))
        self.assertTrue(issubclass(ExtractionError, AgentError))
        self.assertTrue(issubclass(VerificationError, AgentError))

    def test_exception_raising(self):
        """Test exception can be raised"""
        with self.assertRaises(ModelFetchError):
            raise ModelFetchError("Test error")


class TestWorkspaceManager(unittest.TestCase):
    """Test WorkspaceManager"""

    def setUp(self):
        """Set up test workspace"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = WorkspaceManager(self.temp_dir)

    def test_directory_creation(self):
        """Test workspace directories are created"""
        self.assertTrue(self.workspace.models_dir.exists())
        self.assertTrue(self.workspace.generated_dir.exists())
        self.assertTrue(self.workspace.samples_dir.exists())
        self.assertTrue(self.workspace.logs_dir.exists())

    def test_get_model_dir(self):
        """Test model directory path generation"""
        model_id = "bert-base-uncased"
        model_dir = self.workspace.get_model_dir(model_id)
        self.assertEqual(model_dir.name, "bert-base-uncased")
        self.assertEqual(model_dir.parent, self.workspace.models_dir)

    def test_get_generated_dir(self):
        """Test generated directory path generation"""
        model_id = "test/model"
        gen_dir = self.workspace.get_generated_dir(model_id)
        self.assertEqual(gen_dir.name, "test_model")
        self.assertEqual(gen_dir.parent, self.workspace.generated_dir)

    def test_get_sample_dir(self):
        """Test sample directory path generation"""
        model_id = "resnet50"
        sample_dir = self.workspace.get_sample_dir(model_id)
        self.assertEqual(sample_dir.name, "resnet50")
        self.assertEqual(sample_dir.parent, self.workspace.samples_dir)

    def test_get_log_path(self):
        """Test log path generation"""
        model_id = "test-model"
        log_path = self.workspace.get_log_path(model_id, "20240101_120000")
        self.assertIn("test-model", log_path.name)
        self.assertIn("20240101_120000", log_path.name)
        self.assertEqual(log_path.parent, self.workspace.logs_dir)


if __name__ == "__main__":
    unittest.main()
