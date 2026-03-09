import paddle
import os
from paddle.vision.models import squeezenet1_1


def extract_squeezenet():
    # 1. 环境准备
    os.environ["FLAGS_enable_pir_api"] = "1"
    paddle.enable_static()

    # 2. 捕获计算图
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program):
        # 统一使用 'image' 作为输入名，确保与 input_meta.py 一致
        inputs = paddle.static.data(
            name="image", shape=[1, 3, 224, 224], dtype="float32"
        )
        model = squeezenet1_1(pretrained=False)
        model(inputs)

    # 3. 强制指定绝对路径和标准 graph_net.json
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    target_dir = os.path.join(project_root, "paddle_samples/PaddleX/squeezenet1_1")
    save_path = os.path.join(target_dir, "graph_net.json")

    # 4. 确保目录存在并写入
    os.makedirs(target_dir, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(str(main_program))

    print(f"✅ 提取成功！标准文件已生成至: {save_path}")


if __name__ == "__main__":
    extract_squeezenet()
