# Triton Kernel Extractor

A pipeline that compiles computational subgraphs through TorchInductor, filters
the results by kernel-level speedup, and extracts the autotuning-selected Triton
kernel source together with the corresponding PTX assembly from the inductor
compilation cache.

## Background

When `torch.compile` processes a model via the TorchInductor backend with
`TORCH_COMPILE_DEBUG=1`, the compiler produces a per-graph cache directory
containing:

- **`output_code.py`** — the generated Python wrapper that calls into Triton
  kernels via `async_compile.triton('kernel_name', '''...''')`.  The kernels
  appearing here are the final, autotuning-selected implementations adopted by
  the inductor scheduler.
- **`triton/0/{HASH}/`** — one directory per autotuning candidate
  configuration (varying `XBLOCK`, `YBLOCK`, `num_warps`, etc.), each holding
  the compiled artifacts (`.ptx`, `.cubin`, `.ttir`, `.llir`, `.source`,
  `.json`).  When autotuning explores N configurations for a kernel, N
  directories are created.
- **`*.best_config`** — a JSON file written by the Triton autotuner recording
  the winning configuration.  Its `triton_cache_hash` field maps back to one of
  the `triton/0/{HASH}/` directories.

This pipeline automates the full workflow: compile → filter → clean → extract →
pair, producing clean `(subgraph, triton_kernel, ptx)` triples ready for
downstream analysis.

## Pipeline Steps

The pipeline executes five steps on the samples enumerated from `--graph-dir`
(recursive scan) or `--allow-list` (explicit paths):

### Step 1: Multi-GPU Parallel Compilation

Compiles each subgraph sample using `graph_net_bench.torch.test_compiler
--kernel-time` in an isolated subprocess.  Samples are distributed across
available GPUs in round-robin fashion, with one `ProcessPoolExecutor` worker per
GPU.  Each subprocess receives a dedicated `CUDA_VISIBLE_DEVICES` and an
isolated `TORCHINDUCTOR_CACHE_DIR`.  Pass `--max-autotune-no-cudagraphs` to enable
Inductor's `max-autotune-no-cudagraphs` mode (via
`torch.compile(mode="max-autotune-no-cudagraphs")`), which activates comprehensive
autotuning including `max_autotune_gemm`,
`coordinate_descent_tuning`, and `epilogue_fusion`.

### Step 2: Speedup Filtering

Parses the `[Speedup][kernel]:` metric from each sample's compilation log (the
last occurrence is used).  Samples achieving a speedup >= 1.0 are moved to
`kept/`; the rest are moved to `discarded/`.

### Step 3: Temporary File Cleanup

Recursively removes `__pycache__/` directories, `*.pyc`, and `*.pyo` files from
the output tree to reduce storage footprint before extraction.

### Step 4: Kernel and PTX Extraction

For each kept sample that contains `original_graph/graph_hash.txt`:

1. Copies `original_graph/model.py` (the source subgraph) into the output.
2. Parses `output_code.py` to extract all Triton kernel definitions using a
   regex equivalent of the original Perl one-liner.
3. Writes each kernel source to `triton_kernel/{kernel_name}.py`.
4. Locates the corresponding PTX for each kernel by scanning `triton/0/` and
   disambiguating via `.best_config` when multiple autotuning candidates exist,
   then writes it to `ptx/{kernel_name}.ptx`.

Output is written atomically (`.tmp` directory + `rename`) so that an
interrupted run never leaves half-written data.

### Step 5: Empty Sample Cleanup

Removes output samples that contain `original_graph/` but no `triton_kernel/`
directory (i.e., samples where no Triton kernels were extracted).

## PTX Resolution Algorithm

Each Triton kernel may have been compiled under multiple autotuning
configurations.  The algorithm to locate the winning PTX is:

1. Scan `triton/0/*/` for directories containing `{kernel_name}.ptx`.
2. If exactly one candidate exists, use it directly (no autotuning was needed).
3. If multiple candidates exist, collect `triton_cache_hash` values from all
   `*.best_config` files in the sample, and select the candidate whose directory
   name matches one of these hashes.

## Output Structure

```
{output-dir}/
    {sample_name}/                         # compilation cache (kept/discarded)
    kept/
    discarded/
    extracted/
        {sample_name}/
            original_graph/
                model.py                   # source subgraph
            triton_kernel/
                triton_poi_fused_xxx_0.py  # Triton kernel source
                triton_poi_fused_yyy_1.py
            ptx/
                triton_poi_fused_xxx_0.ptx # corresponding PTX assembly
                triton_poi_fused_yyy_1.ptx
    analysis/                              # if --enable-cache-analysis
```

## Usage

```bash
# With allow-list: read sample paths from file, resolve against --graph-dir
python3 -m tools.triton_kernel_extractor extract \
    --allow-list /data/typical_samples_expanded.txt \
    --graph-dir /data/graphs/typical_subgraphs \
    --output-dir /data/output/typical_inductor_dump \
    --gpu-ids 0 2 5 7

# Without allow-list: recursively find all model.py in --graph-dir
python3 -m tools.triton_kernel_extractor extract \
    --graph-dir /data/graphs/typical_subgraphs \
    --output-dir /data/output/typical_inductor_dump \
    --gpu-ids 0 2 5 7 \
    --max-autotune-no-cudagraphs \
    --enable-cache-analysis

# Cache analysis standalone:
python3 -m tools.triton_kernel_extractor analyze <cache_dir> [--output-dir DIR]
```

### CLI Arguments

#### `extract` subcommand

| Argument                  | Required | Default              | Description                                                                 |
|---------------------------|----------|----------------------|-----------------------------------------------------------------------------|
| `--allow-list`            | No       | `None`               | Text file with sample paths (one per line), relative to `--graph-dir`. When omitted, `--graph-dir` is scanned recursively for `model.py` |
| `--graph-dir`             | Yes      | —                    | Input graph data root. Scanned for `model.py` by default; path resolution base when `--allow-list` is given |
| `--output-dir`            | Yes      | —                    | Pipeline output directory (compilation cache, extracted kernels, analysis)   |
| `--gpu-ids`               | No       | Auto-detected        | GPU IDs for parallel compilation. Auto-detected via `CUDA_VISIBLE_DEVICES` or `nvidia-smi` when omitted |
| `--max-autotune-no-cudagraphs` | No  | `False`              | Enable Inductor `max-autotune-no-cudagraphs` mode for compilation |
| `--enable-cache-analysis` | No       | `False`              | Run cache analysis (statistics, plots) after extraction                     |

#### `analyze` subcommand

| Argument       | Required | Default              | Description                                     |
|----------------|----------|----------------------|-------------------------------------------------|
| `cache_dir`    | Yes      | —                    | Inductor cache directory to analyze             |
| `--output-dir` | No       | `<cache_dir>/analysis` | Directory for analysis output                 |

## Module Structure

```
triton_kernel_extractor/
    __init__.py              # package marker
    __main__.py              # CLI entry point (subcommands: extract, analyze)
    config.py                # PipelineConfig, constants
    sample_enumerator.py     # enumerate samples from graph-dir or allow-list
    compiler.py              # Step 1: multi-GPU parallel compilation
    speedup_filter.py        # Step 2: filter by kernel speedup
    temp_cleaner.py          # Step 3: remove __pycache__ / *.pyc / *.pyo
    kernel_extractor.py      # Step 4: extract Triton kernels and PTX
    empty_sample_cleaner.py  # Step 5: remove samples without Triton kernels
    pipeline.py              # orchestrate Steps 1-5
    cache_analyzer.py        # analyze cache: logs, statistics, plots
```

## Idempotency and Resume

Every step implements skip logic to support safe re-execution:

- **Compilation** skips samples whose log already contains `[Speedup][kernel]:`
  or that already exist under `kept/` or `discarded/`.
- **Filtering** skips samples already classified into `kept/` or `discarded/`.
- **Extraction** skips output samples that already exist in the output directory.
  Stale `.tmp` directories from prior interrupted runs are cleaned up
  automatically on startup.
