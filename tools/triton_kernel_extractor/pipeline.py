"""Orchestrate the five-step extraction pipeline for a single run."""

from __future__ import annotations

import logging

from .compiler import compile_all_samples
from .config import PipelineConfig
from .empty_sample_cleaner import clean_empty_kernel_samples
from .kernel_extractor import extract_triton_kernels
from .sample_enumerator import enumerate_graph_dir, enumerate_list_samples
from .speedup_filter import filter_samples_by_speedup
from .temp_cleaner import clean_temp_files

logger = logging.getLogger(__name__)


def _load_samples(config: PipelineConfig) -> list[str]:
    """Load sample paths from allow-list or by scanning graph_dir."""
    if config.allow_list is not None:
        if not config.allow_list.is_file():
            raise FileNotFoundError(f"Allow-list file not found: {config.allow_list}")
        return enumerate_list_samples(config.allow_list, config.graph_dir)
    return enumerate_graph_dir(config.graph_dir)


def run_pipeline(
    config: PipelineConfig,
    *,
    enable_cache_analysis: bool = False,
) -> None:
    """Run the full five-step pipeline."""
    logger.info("")
    logger.info("======================================================")
    logger.info(" Graph dir:  %s", config.graph_dir)
    logger.info(" Output dir: %s", config.output_dir)
    logger.info(" GPUs:       %s", " ".join(str(g) for g in config.gpu_ids))
    if config.allow_list:
        logger.info(" Allow list: %s", config.allow_list)
    if config.max_autotune_no_cudagraphs:
        logger.info(" Autotune:   mode='max-autotune-no-cudagraphs'")
    logger.info("======================================================")

    samples = _load_samples(config)
    logger.info(" Samples:    %d", len(samples))

    if not samples:
        logger.error("No samples found.")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parallel compilation.
    num_gpus = len(config.gpu_ids)
    logger.info("")
    logger.info("=== Step 1: Parallel compilation (%d GPUs) ===", num_gpus)
    compile_all_samples(samples, config)

    # Step 2: Filter by speedup.
    logger.info("")
    logger.info("=== Step 2: Filter by speedup ===")
    filter_samples_by_speedup(config.output_dir)

    # Step 3: Clean temp files.
    logger.info("")
    logger.info("=== Step 3: Clean temp files ===")
    clean_temp_files(config.output_dir)

    # Step 4: Extract triton kernels.
    extraction_dir = config.output_dir / "extracted"
    logger.info("")
    logger.info(
        "=== Step 4: Extract autotuning-selected triton kernels and corresponding PTX ==="
    )
    extract_triton_kernels(config.output_dir, extraction_dir)

    # Step 5: Clean samples without triton kernels.
    logger.info("")
    logger.info("=== Step 5: Clean samples without triton kernels ===")
    clean_empty_kernel_samples(extraction_dir)

    if enable_cache_analysis:
        from .cache_analyzer import analyze_cache

        logger.info("")
        logger.info("=== Cache analysis ===")
        analysis_dir = config.output_dir / "analysis"
        analyze_cache(config.output_dir, analysis_dir)

    logger.info("")
    logger.info("Done.")
