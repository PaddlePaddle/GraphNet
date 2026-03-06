"""
wav2vec2_base 计算图抽取脚本
关键修改：禁用动态形状，确保符合 GraphNet 的"可静态分析"约束
"""

import torch
import torchaudio
from torchaudio.models import wav2vec2_base
from graph_net.torch.extractor import extract


class Wav2Vec2Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features, _lengths = self.model(waveforms)
        return features


def run_model(name: str = "wav2vec2_base", device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    print(f"\nTesting model: {name} on {device_str}")

    # ── 步骤 1：实例化模型（使用随机初始化） ──
    base_model = wav2vec2_base()
    sample_rate = 16_000
    print(f"[INFO] Using random initialization (sample_rate = {sample_rate} Hz)")

    model: torch.nn.Module = Wav2Vec2Wrapper(base_model).to(device).eval()

    # ── 步骤 2：构造固定形状输入 ──
    # 关键：使用固定长度确保输出形状可静态确定
    num_samples: int = sample_rate * 1  # 1s @ 16kHz = 16000 samples
    input_data: torch.Tensor = torch.rand(1, num_samples, device=device)
    print(f"[INFO] Input tensor shape: {input_data.shape}")

    # ── 步骤 3：Eager 模式验证 ──
    with torch.no_grad():
        out = model(input_data)
    print(f"[INFO] Eager mode output — shape: {out.shape}, dtype: {out.dtype}")

    # ── 步骤 4：计算图抽取（关键：dynamic=False） ──
    # 参考 PR #80，必须禁用动态形状追踪
    wrapped = extract(
        name=name,
        dynamic=False  # ✅ 关键修改：禁用动态形状
    )(model).eval()
    
    try:
        with torch.no_grad():
            wrapped(input_data)
        print(f"[OK] {name}: 计算图抽取成功")
    except Exception as exc:
        print(f"[FAIL] {name}: extract error — {exc}")


if __name__ == "__main__":
    run_model(name="wav2vec2_base", device_str="cpu")