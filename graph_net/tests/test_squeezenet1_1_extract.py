import paddle
import os
from paddle.vision.models import squeezenet1_1
from graph_net.paddle.extractor import GraphExtractor

# 1. 环境准备
os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = os.path.abspath("./my_samples")
if not os.path.exists("./my_samples"):
    os.makedirs("./my_samples")

# 2. 准备模型
model = squeezenet1_1(pretrained=False)
model.eval()

# 3. 定义 InputSpec (关键改动：name='inputs')
input_spec = [paddle.static.InputSpec(shape=[1, 3, 224, 224], dtype='float32', name='inputs')]

# 4. 手动实例化提取器
print("正在初始化提取器...")
extractor = GraphExtractor(model, name="squeezenet1_1", dynamic=False, input_spec=input_spec)

# 5. 执行提取 (关键改动：Key 名改为 'inputs')
print("正在执行提取流程...")
model_dump_path = os.path.join(os.environ["GRAPH_NET_EXTRACT_WORKSPACE"], "squeezenet1_1")
dummy_data = {"inputs": paddle.randn([1, 3, 224, 224])}

try:
    # 绕过 __call__ 直接运行内部 dump 逻辑
    extractor.run_model_with_dump_enabled(model_dump_path, **dummy_data)
    print("\n✨✨✨ 奇迹发生了！提取流程成功完成！ ✨✨✨")
    print(f"产物已存至: {model_dump_path}")
except Exception as e:
    print(f"\n❌ 捕获到错误: {e}")