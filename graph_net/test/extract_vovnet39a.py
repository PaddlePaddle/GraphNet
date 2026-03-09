import os

import timm
import torch
from graph_net.torch.extractor import extract

# 强制注入环境变量，防止丢失
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "/home/aistudio/graphnet_workspace"
os.makedirs(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], exist_ok=True)


def main():
    name = "vovnet39a"
    device_str = "cpu"
    device = torch.device(device_str)
    print(f"开始测试模型: {name} on {device_str}")

    try:
        model = timm.create_model(name, pretrained=False)
    except Exception as e:
        print(f"[FAIL] 实例化模型失败 - {e}")
        return

    # VoVNet 标准输入分辨率是 224
    input_data = torch.rand(1, 3, 224, 224, device=device)

    model = model.to(device).eval()
    wrapped = extract(name=name)(model).eval()

    try:
        with torch.no_grad():
            wrapped(input_data)
        print(f"[OK] {name} 计算图抽取成功！")
    except Exception as e:
        print(f"[FAIL] {name} 抽取发生错误 - {e}")


if __name__ == "__main__":
    main()
