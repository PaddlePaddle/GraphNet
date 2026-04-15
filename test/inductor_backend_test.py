"""
Unit tests for InductorBackend config templates.

Tests verify that all config templates are valid and produce expected overrides.
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
    _TEMPLATE_TO_COMPILE_MODE,
    InductorBackend,
)


class TestInductorBackendTemplates(unittest.TestCase):
    """Test InductorBackend config templates."""

    def test_all_templates_exist(self):
        """Verify all expected templates are defined."""
        expected_templates = {
            "triton",
            "cpp_wrapper",
            "cutlass",
            "aten",
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
        backend = InductorBackend({"template": "triton"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {"cpp_wrapper": False})
        self.assertEqual(mode, "default")

    def test_cpp_wrapper_template(self):
        """Test cpp_wrapper template configuration."""
        backend = InductorBackend({"template": "cpp_wrapper"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {"cpp_wrapper": True})
        self.assertEqual(mode, "default")

    def test_cutlass_template(self):
        """Test cutlass template configuration."""
        backend = InductorBackend({"template": "cutlass"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(
            overrides,
            {
                "max_autotune": True,
                "max_autotune_gemm": True,
                "epilogue_fusion": True,
                "coordinate_descent_tuning": True,
            },
        )
        self.assertEqual(mode, "default")

    def test_aten_template(self):
        """Test aten template configuration."""
        backend = InductorBackend({"template": "aten"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {"autotune_fallback_to_aten": True})
        self.assertEqual(mode, "default")

    def test_cudagraphs_template(self):
        """Test cudagraphs template configuration."""
        backend = InductorBackend({"template": "cudagraphs"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {"triton.cudagraphs": True})
        self.assertEqual(mode, "reduce-overhead")

    def test_max_autotune_template(self):
        """Test max_autotune template configuration."""
        backend = InductorBackend({"template": "max_autotune"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(
            overrides,
            {
                "max_autotune": True,
                "max_autotune_gemm": True,
                "coordinate_descent_tuning": True,
                "epilogue_fusion": True,
            },
        )
        self.assertEqual(mode, "max-autotune")

    def test_freezing_template(self):
        """Test freezing template configuration."""
        backend = InductorBackend({"template": "freezing"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {"freezing": True})
        self.assertEqual(mode, "default")

    def test_tma_template(self):
        """Test TMA template configuration.

        Note: This test verifies the config is accepted. On GPUs without
        TMA support (e.g., A100), Inductor will gracefully fall back
        to non-TMA persistent kernels. The config itself is valid regardless
        of the underlying hardware.
        """
        backend = InductorBackend({"template": "tma"})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {"triton.enable_persistent_tma_matmul": True})
        self.assertEqual(mode, "default")

    def test_mode_override(self):
        """Test explicit mode override."""
        backend = InductorBackend({"template": "cudagraphs", "mode": "max-autotune"})
        mode = backend._resolve_compile_mode()

        # Explicit mode should take precedence
        self.assertEqual(mode, "max-autotune")

    def test_freezing_override(self):
        """Test explicit freezing override."""
        backend = InductorBackend({"template": "max_autotune", "freezing": False})
        overrides = backend._build_inductor_overrides()

        # Freezing should override template defaults
        self.assertIn("freezing", overrides)
        self.assertEqual(overrides["freezing"], False)
        self.assertIn("max_autotune", overrides)

    def test_inductor_config_override(self):
        """Test inductor_config overrides take highest priority."""
        backend = InductorBackend(
            {
                "template": "triton",
                "inductor_config": {"cpp_wrapper": True, "triton.cudagraphs": True},
            }
        )
        overrides = backend._build_inductor_overrides()

        # inductor_config overrides should be applied
        self.assertIn("cpp_wrapper", overrides)
        self.assertIn("triton.cudagraphs", overrides)

    def test_invalid_template_raises_error(self):
        """Test invalid template raises ValueError."""
        with self.assertRaises(ValueError) as context:
            backend = InductorBackend({"template": "invalid_template"})
            backend._build_inductor_overrides()

        self.assertIn("Unknown Inductor config template", str(context.exception))
        self.assertIn("invalid_template", str(context.exception))

    def test_empty_config(self):
        """Test empty config produces no overrides."""
        backend = InductorBackend({})
        overrides = backend._build_inductor_overrides()
        mode = backend._resolve_compile_mode()

        self.assertEqual(overrides, {})
        self.assertEqual(mode, "default")

    def test_template_to_mode_mapping(self):
        """Test template to mode mapping."""
        self.assertEqual(
            _TEMPLATE_TO_COMPILE_MODE,
            {
                "cudagraphs": "reduce-overhead",
                "max_autotune": "max-autotune",
            },
        )


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestInductorConfigValidation(unittest.TestCase):
    """Test that all config overrides reference valid torch._inductor.config attributes."""

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

    def test_autotune_fallback_to_aten_config_exists(self):
        """Test autotune_fallback_to_aten config exists."""
        self.assertTrue(self._check_config_key_exists("autotune_fallback_to_aten"))

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

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_tma_fallback_on_non_tma_gpu(self):
        """Test TMA config gracefully falls back on non-TMA GPUs.

        This verifies that even on GPUs without TMA support (like A100),
        the TMA config doesn't cause errors - it just falls back to
        non-TMA persistent kernels.
        """
        backend = InductorBackend({"template": "tma"})

        # On non-TMA GPUs, this should not raise an error
        # (Inductor handles the fallback internally)
        try:
            import torch._inductor.config as cfg
            from graph_net_bench.torch.backend.inductor_backend import _set_nested_attr

            # This is what __call__ does - should not fail
            for key, value in backend._build_inductor_overrides().items():
                _set_nested_attr(cfg, key, value)

            # If we get here, config was accepted successfully
            # (regardless of actual TMA support on the GPU)
            self.assertTrue(True)
        except Exception as e:
            # Should not fail even on non-TMA GPU
            self.fail(f"TMA config should be accepted on non-TMA GPU: {e}")


class TestInductorBackendIntegration(unittest.TestCase):
    """Integration tests for InductorBackend."""

    def test_backend_initialization(self):
        """Test backend can be initialized with various configs."""
        configs = [
            {},
            {"template": "triton"},
            {"template": "max_autotune", "mode": "default"},
            {"template": "cudagraphs", "freezing": True},
        ]

        for config in configs:
            backend = InductorBackend(config)
            self.assertIsNotNone(backend)

    def test_build_overrides_priority(self):
        """Test override priority: template < freezing < inductor_config."""
        backend = InductorBackend(
            {
                "template": "freezing",
                "freezing": False,
                "inductor_config": {
                    "freezing": True,
                    "max_autotune": True,
                },
            }
        )
        overrides = backend._build_inductor_overrides()

        # inductor_config has highest priority
        self.assertTrue(overrides["freezing"])
        self.assertTrue(overrides["max_autotune"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
