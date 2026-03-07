import torch
from torchvision.models import get_model, get_model_weights
from graph_net.torch.extractor import extract

def main():
    name = 'resnet18'
    device_str = 'cpu' # 推荐先用 CPU 测试
    device = torch.device(device_str)
    print(f"开始测试模型: {name} on {device_str}")

    # 加载模型与权重
    try:
        w = get_model_weights(name)
        model = get_model(name, weights=w.DEFAULT)
    except Exception as e:
        print(f"[FAIL] 实例化模型失败 - {e}")
        return

    # 构造标准 CV 模型的输入张量 (1, 3, 224, 224)
    input_data = torch.rand(1, 3, 224, 224, device=device)

    # 使用 GraphNet 官方 API 包装并抽取计算图
    model = model.to(device).eval()
    wrapped = extract(name=name)(model).eval()
    
    try:
        with torch.no_grad():
            # 执行前向传播，计算图会自动保存到 GRAPH_NET_EXTRACT_WORKSPACE 目录
            wrapped(input_data)
        print(f"[OK] {name} 计算图抽取成功！")
    except Exception as e:
        print(f"[FAIL] {name} 抽取发生错误 - {e}")

if __name__ == "__main__":
    main()