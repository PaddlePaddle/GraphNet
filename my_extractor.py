import paddle
import os
from paddle.vision.models import squeezenet1_1

# 1. 环境强制设定
os.environ["FLAGS_enable_pir_api"] = "1"
paddle.enable_static() # 切换到静态图模式

save_dir = "./my_samples/squeezenet1_1"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2. 定义静态图容器
main_program = paddle.static.Program()
startup_program = paddle.static.Program()

with paddle.static.program_guard(main_program, startup_program):
    # 3. 在静态图内定义输入
    inputs = paddle.static.data(name='inputs', shape=[1, 3, 224, 224], dtype='float32')
    
    # 4. 实例化模型并运squeezenet1_1squeezenet1_1
    model = squeezenet1_1(pretrained=False)
    out = model(inputs)

# 5. 此时 main_program 已经包含 PIR 计算图
save_path = os.path.join(save_dir, "squeezenet1_1.json")
with open(save_path, "w") as f:
    # 导出 PIR 的文本序列化
    f.write(str(main_program))

print(f"DONE! PIR Graph captured.")
print(f"File: {save_path}")
print(f"Size: {os.path.getsize(save_path)} bytes")
