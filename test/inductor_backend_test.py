"""
Unit tests for InductorBackend config templates.

Tests verify that all config templates are valid and produce expected options.
Uses relative paths to access the backend module.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))

try:
    import torch  # noqa: F401 - Used in @unittest.skipIf decorator
    import torch._inductor.config as inductor_config

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from graph_net_bench.torch.backend.inductor_backend import (  # noqa: E402 - Import after try/except is intentional
    _INDUCTOR_CONFIG_TEMPLATES,
    InductorBackend,
)


class TestInductorBackendTemplates(unittest.TestCase):
    """Test InductorBackend config templates."""

    def test_all_templates_exist(self):
        """Verify all expected templates are defined."""
        expected_templates = {
            "triton",
            "cpp_wrapper",
            "cudagraphs",
            "max_autotune",
            "freezing",
            "tma",
        }
        actual_templates = set(_INDUCTOR_CONFIG_TEMPLATES.keys())
        self.assertEqual(
            expected_templates,
            actual_templates,
            f"Expected templates {expected_templates}, got {actual_templates}",
        )

    def test_triton_template(self):
        """Test triton template configuration."""
        backend = InductorBackend({"graph_net_inductor_config_template": "triton"})
        options = backend._build_options()

        self.assertEqual(options, {"cpp_wrapper": False})

    def test_cpp_wrapper_template(self):
        """Test cpp_wrapper template configuration."""
        backend = InductorBackend({"graph_net_inductor_config_template": "cpp_wrapper"})
        options = backend._build_options()

        self.assertEqual(options, {"cpp_wrapper": True})

    def test_cudagraphs_template(self):
        """Test cudagraphs template configuration."""
        backend = InductorBackend({"graph_net_inductor_config_template": "cudagraphs"})
        options = backend._build_options()

        self.assertEqual(options, {"triton.cudagraphs": True})

    def test_max_autotune_template(self):
        """Test max_autotune template configuration."""
        backend = InductorBackend(
            {"graph_net_inductor_config_template": "max_autotune"}
        )
        options = backend._build_options()

        self.assertEqual(
            options,
            {
                "max_autotune": True,
                "max_autotune_gemm": True,
                "coordinate_descent_tuning": True,
                "epilogue_fusion": True,
            },
        )

    def test_freezing_template(self):
        """Test freezing template configuration."""
        backend = InductorBackend({"graph_net_inductor_config_template": "freezing"})
        options = backend._build_options()

        self.assertEqual(options, {"freezing": True})

    def test_tma_template(self):
        """Test TMA template configuration.

        Note: This test verifies that config is accepted. On GPUs without
        TMA support (e.g., A100), Inductor will gracefully fall back
        to non-TMA persistent kernels. The config itself is valid regardless
        of underlying hardware.
        """
        backend = InductorBackend({"graph_net_inductor_config_template": "tma"})
        options = backend._build_options()

        self.assertEqual(options, {"triton.enable_persistent_tma_matmul": True})

    def test_options_only(self):
        """Test options alone."""
        backend = InductorBackend({"options": {"triton.cudagraphs": True}})
        options = backend._build_options()

        self.assertEqual(options, {"triton.cudagraphs": True})

    def test_invalid_template_raises_error(self):
        """Test invalid template raises ValueError."""
        with self.assertRaises(ValueError) as context:
            backend = InductorBackend(
                {"graph_net_inductor_config_template": "invalid_template"}
            )
            backend._build_options()

        self.assertIn("Unknown config template", str(context.exception))
        self.assertIn("invalid_template", str(context.exception))

    def test_mutually_exclusive_params(self):
        """Test that template/mode/options are mutually exclusive."""
        configs = [
            {
                "graph_net_inductor_config_template": "triton",
                "mode": "reduce-overhead",
            },
            {
                "graph_net_inductor_config_template": "triton",
                "options": {"triton.cudagraphs": True},
            },
            {
                "mode": "reduce-overhead",
                "options": {"triton.cudagraphs": True},
            },
            {
                "graph_net_inductor_config_template": "triton",
                "mode": "reduce-overhead",
                "options": {"triton.cudagraphs": True},
            },
        ]

        for config in configs:
            with self.assertRaises(ValueError) as context:
                InductorBackend(config)

            self.assertIn("exactly one", str(context.exception))

    def test_empty_config(self):
        """Test empty config produces no options."""
        backend = InductorBackend({})
        options = backend._build_options()

        self.assertEqual(options, {})


class TestInductorBackendParameters(unittest.TestCase):
    """Test InductorBackend torch.compile parameters."""

    def test_fullgraph_parameter(self):
        """Test fullgraph parameter is stored."""
        backend = InductorBackend({"fullgraph": True})
        self.assertTrue(backend._fullgraph)

        backend = InductorBackend({})
        self.assertFalse(backend._fullgraph)

    def test_dynamic_parameter(self):
        """Test dynamic parameter is stored."""
        backend = InductorBackend({"dynamic": True})
        self.assertTrue(backend._dynamic)

        backend = InductorBackend({"dynamic": False})
        self.assertFalse(backend._dynamic)

        backend = InductorBackend({})
        self.assertIsNone(backend._dynamic)

    def test_mode_parameter(self):
        """Test mode parameter is stored."""
        backend = InductorBackend({"mode": "max-autotune"})
        self.assertEqual(backend._mode, "max-autotune")

        backend = InductorBackend({})
        self.assertIsNone(backend._mode)

    def test_options_parameter(self):
        """Test options parameter is stored."""
        options = {"triton.cudagraphs": True, "max_autotune": True}
        backend = InductorBackend({"options": options})
        self.assertEqual(backend._options, options)

    def test_all_compatible_parameters_combined(self):
        """Test all torch.compile parameters can be combined when compatible."""
        config = {
            "graph_net_inductor_config_template": "max_autotune",
            "fullgraph": True,
            "dynamic": True,
        }
        backend = InductorBackend(config)

        self.assertEqual(backend._config_template, "max_autotune")
        self.assertIsNone(backend._mode)
        self.assertTrue(backend._fullgraph)
        self.assertTrue(backend._dynamic)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestInductorConfigValidation(unittest.TestCase):
    """Test that all config options reference valid torch._inductor.config attributes."""

    def _check_config_key_exists(self, key):
        """Check if a config key (possibly nested) exists in torch._inductor.config."""
        parts = key.split(".")
        obj = inductor_config
        try:
            for part in parts:
                obj = getattr(obj, part)
            return True
        except AttributeError:
            return False

    def test_cpp_wrapper_config_exists(self):
        """Test cpp_wrapper config exists."""
        self.assertTrue(self._check_config_key_exists("cpp_wrapper"))

    def test_max_autotune_config_exists(self):
        """Test max_autotune config exists."""
        self.assertTrue(self._check_config_key_exists("max_autotune"))

    def test_max_autotune_gemm_config_exists(self):
        """Test max_autotune_gemm config exists."""
        self.assertTrue(self._check_config_key_exists("max_autotune_gemm"))

    def test_epilogue_fusion_config_exists(self):
        """Test epilogue_fusion config exists."""
        self.assertTrue(self._check_config_key_exists("epilogue_fusion"))

    def test_coordinate_descent_tuning_config_exists(self):
        """Test coordinate_descent_tuning config exists."""
        self.assertTrue(self._check_config_key_exists("coordinate_descent_tuning"))

    def test_freezing_config_exists(self):
        """Test freezing config exists."""
        self.assertTrue(self._check_config_key_exists("freezing"))

    def test_enable_persistent_tma_matmul_config_exists(self):
        """Test triton.enable_persistent_tma_matmul config exists."""
        self.assertTrue(
            self._check_config_key_exists("triton.enable_persistent_tma_matmul")
        )

    def test_triton_cudagraphs_config_exists(self):
        """Test triton.cudagraphs config exists."""
        self.assertTrue(self._check_config_key_exists("triton.cudagraphs"))

    def test_all_template_configs_exist(self):
        """Test all configs from templates exist in torch._inductor.config."""
        missing_configs = []
        for template_name, template_config in _INDUCTOR_CONFIG_TEMPLATES.items():
            for key in template_config.keys():
                if not self._check_config_key_exists(key):
                    missing_configs.append(f"{template_name}.{key}")

        if missing_configs:
            self.fail(f"Missing config keys: {missing_configs}")


class TestInductorBackendIntegration(unittest.TestCase):
    """Integration tests for InductorBackend."""

    def test_backend_initialization(self):
        """Test backend can be initialized with various configs."""
        configs = [
            {},
            {"graph_net_inductor_config_template": "triton"},
            {"mode": "max-autotune"},
            {"options": {"max_autotune": True}, "fullgraph": True},
            {"fullgraph": True, "dynamic": True},
        ]

        for config in configs:
            backend = InductorBackend(config)
            self.assertIsNotNone(backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
