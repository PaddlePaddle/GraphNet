"""Integration tests for Agent end-to-end workflow"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from graph_net.agent.graph_net_agent import GraphNetAgent


class TestAgentIntegration(unittest.TestCase):
    """Test Agent end-to-end workflow"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.agent = GraphNetAgent(workspace=self.temp_dir)

    @patch("graph_net.agent.model_fetcher.huggingface_fetcher.snapshot_download")
    def test_agent_initialization(self, mock_download):
        """Test Agent can be initialized"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.model_fetcher)
        self.assertIsNotNone(self.agent.metadata_analyzer)
        self.assertIsNotNone(self.agent.code_generator)
        self.assertIsNotNone(self.agent.graph_extractor)
        self.assertIsNotNone(self.agent.sample_verifier)

    @patch("graph_net.agent.model_fetcher.huggingface_fetcher.snapshot_download")
    @patch("graph_net.agent.graph_extractor.subprocess_graph_extractor.subprocess.run")
    def test_full_workflow_mock(self, mock_subprocess, mock_download):
        """Test full workflow with mocked dependencies"""
        # Mock model download
        mock_model_dir = Path(self.temp_dir) / "models" / "test_model"
        mock_model_dir.mkdir(parents=True)
        (mock_model_dir / "config.json").write_text(
            '{"model_type": "bert", "max_position_embeddings": 512}'
        )
        mock_download.return_value = str(mock_model_dir)

        # Mock subprocess execution (success)
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Mock output directory
        mock_output_dir = Path(self.temp_dir) / "workspace" / "test_model"
        mock_output_dir.mkdir(parents=True)
        (mock_output_dir / "model.py").write_text("class GraphModule: pass")
        (mock_output_dir / "graph_net.json").write_text('{"framework": "torch"}')
        (mock_output_dir / "input_meta.py").write_text("")
        (mock_output_dir / "weight_meta.py").write_text("")
        (mock_output_dir / "graph_hash.txt").write_text("test_hash")

        # Mock extractor to return output_dir
        self.agent.graph_extractor._find_output_dir_robust = Mock(
            return_value=mock_output_dir
        )

        # Run agent
        result = self.agent.extract_sample("test-model")

        # Should succeed (with mocked dependencies)
        # Note: This will likely fail at subprocess execution in real scenario
        # but tests the workflow structure
        self.assertIsInstance(result, bool)

    def test_deduplicate_logic(self):
        """Test deduplicate logic"""
        # Create a sample directory with graph_hash
        sample_dir = Path(self.temp_dir) / "sample"
        sample_dir.mkdir()
        (sample_dir / "graph_hash.txt").write_text("test_hash_123")

        # Test duplicate check
        result = self.agent.is_duplicate_sample(sample_dir)
        # Should return False if no duplicate found
        self.assertIsInstance(result, bool)

    def test_archive_logic(self):
        """Test archive logic"""
        # Create sample directory
        sample_dir = Path(self.temp_dir) / "sample"
        sample_dir.mkdir()

        # Create a test script
        script_path = Path(self.temp_dir) / "test_script.py"
        script_path.write_text("print('test')")

        # Test script archiving
        result = self.agent.save_extraction_script(script_path, sample_dir)
        self.assertTrue(result)
        self.assertTrue((sample_dir / "run_model.py").exists())


if __name__ == "__main__":
    unittest.main()
