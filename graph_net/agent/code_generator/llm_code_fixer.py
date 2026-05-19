"""LLM-based script fixer using ducc -p (Claude Code non-interactive mode)"""

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from graph_net.agent.utils.exceptions import (
    CodeGenerationError,
    GraphExtractionErrorCategory,
)

# Candidate binary names / paths to search for ducc CLI
_DUCC_CANDIDATES = [
    "ducc",
    "claude",
    "/usr/local/bin/ducc",
    os.path.expanduser("~/.local/bin/ducc"),
]

_SYSTEM_PROMPT = """\
你是 PyTorch / HuggingFace 模型计算图抽取专家。
任务：修复一段失败的图抽取脚本，输出完整、可直接运行但最小化的 Python 脚本。

## 【硬性约束 - 违反即输出无效】
1. 抽取调用格式固定为：
   graph_net.torch.extract(name="{name}", dynamic=False)(model).eval()(**inputs)
   - name 值已指定，禁止修改；dynamic 必须为 False（Swin/F.pad 等动态模式会崩溃）
2. 模型加载必须用随机权重（禁止下载权重文件）：
   config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
   model = AutoModel.from_config(config)   # 或对应任务类
3. 设备选择固定写法：device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
4. 只允许使用 torch、transformers、graph_net 及 Python 标准库（os/pathlib/json 等）
5. 只输出代码块，格式：```python\\n...代码...\\n```，禁止输出任何说明文字
6. 必须输出完整但最小化的脚本，只保留：必要 import、模型/config 加载、输入 tensor 构造、graph_net.torch.extract(...)、一次 forward 调用
7. 禁止添加注释、helper 函数、错误处理、try/except、fallback 逻辑、重试逻辑、文件系统遍历、额外校验或无关打印。只修复导致报错的输入构造或调用方式，保持行数尽可能少

## 【输入构造规范 - 按 model_type 选择对应方案】

**文本类**（bert/roberta/electra/albert/distilbert/mpnet/xlm/deberta/camembert 等）：
  seq_len = 128
  vocab_upper = min(vocab_size - 1, 30000)   # 严格小于 vocab_size，防止 embedding 越界
  input_ids      = torch.randint(0, vocab_upper, (1, seq_len), dtype=torch.long).to(device)
  attention_mask = torch.ones((1, seq_len), dtype=torch.long).to(device)
  token_type_ids = torch.zeros((1, seq_len), dtype=torch.long).to(device)  # 仅 BERT 系需要
  ⚠️ 绝对禁止用 vocab_size 本身作为 randint 上界

**视觉类**（vit/swin/convnext/deit/resnet/efficientnet/mobilenet 等）：
  num_channels = config 中读取，默认 3
  image_size   = config 中读取（vision_config.image_size 或 image_size），默认 224
  pixel_values = torch.randn(1, num_channels, image_size, image_size).to(device)
  ⚠️ 纯视觉模型只传 pixel_values，禁止传 input_ids

**多模态类**（clip/blip/flava/align 等）：
  同时传文本分支（input_ids + attention_mask）和视觉分支（pixel_values）

**音频类**（wav2vec2/hubert/whisper/clap/unispeech 等）：
  - wav2vec2/hubert/unispeech：input_values = torch.randn(1, 16000).to(device)
  - whisper：input_features = torch.randn(1, 80, 3000).to(device)
  - clap：input_features=torch.randn(1, 1, 1001, 64).to(device), is_longer=torch.tensor([False]).to(device)
  - 通用回退：input_features = torch.randn(1, 128, 64).to(device)

**序列到序列类**（t5/bart/marian/pegasus/longt5 等）：
  input_ids      = torch.randint(0, min(vocab_size-1, 1000), (1, 64), dtype=torch.long).to(device)
  attention_mask = torch.ones((1, 64), dtype=torch.long).to(device)
  decoder_input_ids = torch.randint(0, min(vocab_size-1, 1000), (1, 32), dtype=torch.long).to(device)

**MoE 类**（mixtral/qwen2_moe/deepseek_v2/dbrx/olmoe 等）：
  - 架构上仍是文本模型，输入与文本类完全相同（input_ids + attention_mask）
  - 加载同样用 AutoModel.from_config(config)，无需任何特殊处理
  ⚠️ vocab_size 通常很大（32000+），严格用 min(vocab_size-1, 30000) 作为 randint 上界
  关键 config 字段：num_local_experts（Mixtral）/ num_experts（Qwen2-MoE）/ n_routed_experts（DeepSeek）

**扩散模型类**（UNet2DConditionModel / DiT / stable-diffusion / SDXL 等）：
  from diffusers import UNet2DConditionModel
  _config = UNet2DConditionModel.load_config(model_dir)
  model = UNet2DConditionModel.from_config(_config)
  # 从 config 读取关键维度
  in_channels        = _config.get("in_channels", 4)
  sample_size        = _config.get("sample_size", 64)
  cross_attention_dim = _config.get("cross_attention_dim", 768)
  sample                 = torch.randn(1, in_channels, sample_size, sample_size).to(device)
  timestep               = torch.tensor([1]).to(device)
  encoder_hidden_states  = torch.randn(1, 77, cross_attention_dim).to(device)
  # 调用必须用位置参数，不能 **inputs
  wrapped(sample, timestep, encoder_hidden_states)
  ⚠️ dynamic 必须为 False；调用格式固定为位置参数，禁止用 **inputs dict 展开

## 【常见报错 → 修复方法】
| 报错关键词 | 修复方法 |
|---|---|
| "You have to specify pixel_values" | 补充 pixel_values 输入 |
| "index out of range in self" / embedding 越界 | input_ids 上界 > vocab_size，改为 min(vocab_size-1, 30000) |
| "NoneType has no attribute" | 对应输入字段为 None，补充正确 tensor |
| "running_mean should contain X elements" | BatchNorm channel 维度不对，检查 input_features shape 的 channel 轴 |
| "size of tensor a must match tensor b" | 序列长度或 channel 不一致，统一固定值 |
| "Sizes of tensors must match except in dimension" | 注意 encoder/decoder 序列长度可以不同，不要强制相等 |
| "Expected input batch_size to match target batch_size" | batch size 统一为 1 |
| "sentencepiece" / "tiktoken" ImportError | 不使用 tokenizer，用 torch.randint 直接构造 input_ids |
| "PendingUnbackedSymbolNotFound" | 确认 dynamic=False（不要改为 True） |
| decoder_input_ids missing | Seq2Seq 模型需要同时传 input_ids 和 decoder_input_ids |
| "encoder_hidden_states" required（UNet） | 扩散模型必须以位置参数传入 encoder_hidden_states，不能省略 |
| UNet sample/timestep 形状错误 | 检查 in_channels/sample_size/cross_attention_dim 是否从 config 正确读取 |
| MoE expert 路由 RuntimeError | 输入格式与普通文本模型相同，通常是 vocab 越界，检查 randint 上界是否 < vocab_size |
"""


_CONFIG_JSON_MAX_CHARS = 4096


def _find_ducc() -> Optional[str]:
    """Find ducc/claude binary, return full path or None."""
    for candidate in _DUCC_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _extract_code_block(text: str) -> Optional[str]:
    """Extract first ```python ... ``` code block from LLM output."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: if entire output looks like python code
    stripped = text.strip()
    if stripped.startswith("import ") or stripped.startswith("from "):
        return stripped
    return None


class LLMCodeFixer:
    """Fix a failed extraction script by calling ducc/claude -p (non-interactive)."""

    def __init__(
        self,
        timeout: int = 360,
        model: Optional[str] = None,
    ):
        """
        Args:
            timeout: Max seconds to wait for ducc response (default 360s).
            model:   Override the LLM model (e.g. 'sonnet', 'haiku').
                     If None, uses whatever ducc default is configured.
        """
        self.timeout = timeout if timeout is not None else 360
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ducc_bin = _find_ducc()
        if self._ducc_bin:
            self.logger.info(f"LLMCodeFixer: using CLI at {self._ducc_bin}")
        else:
            self.logger.warning(
                "LLMCodeFixer: ducc/claude binary not found; "
                "LLM retry will be skipped. "
                "Add ducc or claude to PATH."
            )

    @property
    def available(self) -> bool:
        return self._ducc_bin is not None

    def fix(
        self,
        script_path: Path,
        error_msg: str,
        model_dir: Path,
        model_id: str,
        output_dir: Path,
        attempt: int = 1,
    ) -> Path:
        """
        Ask the LLM to fix a failed extraction script.

        Args:
            script_path: Path to the (failed) script to fix
            error_msg:   Captured stderr / GraphExtractionError message
            model_dir:   Local model directory (contains config.json)
            model_id:    HuggingFace model ID (e.g. 'prajjwal1/bert-tiny')
            output_dir:  Directory where the fixed script should be written
            attempt:     Retry index (1 or 2), affects output filename

        Returns:
            Path to the fixed script (run_model_llm_1.py / run_model_llm_2.py)

        Raises:
            CodeGenerationError: If LLM call fails or returns no valid code
        """
        if not self.available:
            raise CodeGenerationError(
                "ducc/claude binary not available; cannot perform LLM fix."
            )

        original_script = script_path.read_text(encoding="utf-8")
        config_json = self._read_config(model_dir)
        safe_name = model_id.replace("/", "_")

        prompt = self._build_prompt(
            original_script=original_script,
            error_msg=error_msg,
            config_json=config_json,
            model_id=model_id,
            safe_name=safe_name,
            model_dir=model_dir,
        )

        self.logger.info(
            f"Calling LLM to fix script for {model_id} (attempt {attempt}) ..."
        )
        llm_output = self._call_ducc(prompt)

        code = _extract_code_block(llm_output)
        if not code:
            raise CodeGenerationError(
                f"LLM response contained no Python code block.\n"
                f"Response (first 500 chars):\n{llm_output[:500]}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        fixed_path = output_dir / f"run_model_llm_{attempt}.py"
        fixed_path.write_text(code, encoding="utf-8")
        self.logger.info(f"LLM-fixed script written to: {fixed_path}")
        return fixed_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_script(script: str) -> str:
        """Remove blank lines and pure comment lines to shrink prompt size."""
        lines = script.splitlines()
        compacted = []
        for line in lines:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            compacted.append(line.rstrip())
        return "\n".join(compacted)

    @staticmethod
    def _truncate_error(error_msg: str, max_chars: int = 1200) -> str:
        if len(error_msg) <= max_chars:
            return error_msg
        # Keep tail (usually contains the actual error) + head for context
        half = max_chars // 2
        return error_msg[:half] + "\n... (truncated) ...\n" + error_msg[-half:]

    def _build_prompt(
        self,
        original_script: str,
        error_msg: str,
        config_json: str,
        model_id: str,
        safe_name: str,
        model_dir: Path,
    ) -> str:
        model_dir_str = str(model_dir).replace("\\", "/")
        system = _SYSTEM_PROMPT.format(name=safe_name)
        key_fields = self._extract_key_fields(model_dir)

        # Compact script to reduce prompt bloat (keep structure, drop empty/comment lines)
        compact_script = self._compact_script(original_script)
        # If still very long, fall back to raw script so we don't lose critical logic
        if len(compact_script) < len(original_script) * 0.3:
            compact_script = original_script

        truncated_error = self._truncate_error(error_msg)

        return (
            f"{system}\n\n"
            f"---\n\n"
            f"## 当前任务\n\n"
            f"### 模型信息\n"
            f"- model_id: `{model_id}`\n"
            f"- config_dir: `{model_dir_str}`\n"
            f"- 关键配置字段:\n```json\n{key_fields}\n```\n\n"
            f"### 失败脚本\n```python\n{compact_script}\n```\n\n"
            f"### 错误信息\n```\n{truncated_error}\n```\n\n"
            f"### 输出要求\n"
            f"直接输出修复后的完整最小脚本，用 ```python\\n...\\n``` 包裹，不附加任何说明。"
            f"只保留必要 import、模型/config 加载、输入 tensor 构造、extract 调用和一次 forward。"
            f"禁止注释、helper、try/except、fallback、重试、文件遍历、额外校验或无关打印。"
        )

    def _call_ducc(self, prompt: str) -> str:
        """Invoke ducc -p '<prompt>' and return stdout."""
        cmd = [
            self._ducc_bin,
            "-p",
            prompt,
        ]
        if self.model:
            cmd += ["--model", self.model]

        # Inherit current env so ANTHROPIC_* vars are passed through
        env = os.environ.copy()
        # Ensure the binary's directory is in PATH (handles non-PATH installs)
        bin_dir = str(Path(self._ducc_bin).parent)
        if bin_dir not in env.get("PATH", ""):
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise CodeGenerationError(
                f"ducc -p timed out after {self.timeout}s",
                error_category=GraphExtractionErrorCategory.LLM_TIMEOUT,
            )

        if result.returncode != 0:
            raise CodeGenerationError(
                f"ducc -p exited with code {result.returncode}.\n"
                f"stderr: {result.stderr[:500]}",
                error_category=GraphExtractionErrorCategory.LLM_EXIT_ERROR,
            )

        output = result.stdout.strip()
        if not output:
            raise CodeGenerationError("ducc -p returned empty output.")

        return output

    @staticmethod
    def _read_config(model_dir: Path) -> str:
        """Read config.json as a compact JSON string (max 4 KB)."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return "{}"
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            text = json.dumps(raw, ensure_ascii=False, indent=2)
            if len(text) > _CONFIG_JSON_MAX_CHARS:
                text = text[:_CONFIG_JSON_MAX_CHARS] + "\n... (truncated)"
            return text
        except Exception:
            return "{}"

    @staticmethod
    def _extract_key_fields(model_dir: Path) -> str:
        """Extract the most important input-construction fields from config.json for the LLM."""
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return "{}"
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return "{}"
        keys = [
            "model_type",
            "vocab_size",
            "max_position_embeddings",
            "image_size",
            "num_channels",
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            # audio/multimodal
            "audio_config",
            "vision_config",
            "text_config",
            "patch_size",
            "num_mel_bins",
            "chunk_length",
            # MoE routing (field names vary across models)
            "num_local_experts",
            "num_experts_per_tok",
            "num_experts",
            "n_routed_experts",
            "moe_intermediate_size",
            "num_shared_experts",
            # Diffusion / UNet
            "in_channels",
            "sample_size",
            "cross_attention_dim",
            "layers_per_block",
            # Seq2Seq
            "is_encoder_decoder",
            "decoder_start_token_id",
            # GQA (Llama/Mistral family)
            "num_key_value_heads",
            # Audio
            "feature_size",
            "sample_rate",
        ]
        result = {k: cfg[k] for k in keys if k in cfg}
        # Keep only key fields from nested configs.
        for nested in ("audio_config", "vision_config", "text_config"):
            if isinstance(result.get(nested), dict):
                result[nested] = {
                    k: result[nested][k]
                    for k in (
                        "model_type",
                        "vocab_size",
                        "image_size",
                        "num_channels",
                        "num_mel_bins",
                        "hidden_size",
                        "num_local_experts",
                        "num_experts",
                        "n_routed_experts",
                        "sample_rate",
                    )
                    if k in result[nested]
                }
        return json.dumps(result, ensure_ascii=False)
