import os

import timm
import torch
from graph_net.torch.extractor import extract

# 强制在运行脚本时注入环境变量，避免终端重置导致的环境丢失
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/home/aistudio/graphnet_workspace"
os.makedirs(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], exist_ok=True)


def main():
    name = "convnext_atto"
    device_str = "cpu"
    device = torch.device(device_str)
    print(f"开始测试模型: {name} on {device_str}")

    # 1. 实例化 timm 模型
    # 抽取计算图只关心网络结构 (OPs)，无需下载庞大的预训练权重，因此 pretrained=False
    try:
        model = timm.create_model(name, pretrained=False)
    except Exception as e:
        print(f"[FAIL] 实例化模型失败 - {e}")
        return

    # 2. 构造标准 CV 模型的输入张量 (BatchSize=1, Channels=3, H=224, W=224)
    input_data = torch.rand(1, 3, 224, 224, device=device)

    # 3. 包装并抽取计算图
    model = model.to(device).eval()
    wrapped = extract(name=name)(model).eval()

    try:
        with torch.no_grad():
            # 执行前向传播，GraphNet 装饰器会自动拦截并保存计算图
            wrapped(input_data)
        print(f"[OK] {name} 计算图抽取成功！")
    except Exception as e:
        print(f"[FAIL] {name} 抽取发生错误 - {e}")


if __name__ == "__main__":
    main()
