import paddle
import os
from paddle.vision.models import squeezenet1_1


def extract_squeezenet():
    # 环境准备
    os.environ["FLAGS_enable_pir_api"] = "1"
    paddle.enable_static()

    # 捕获计算图
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program):
        inputs = paddle.static.data(
            name="inputs", shape=[1, 3, 224, 224], dtype="float32"
        )
        model = squeezenet1_1(pretrained=False)
        out = model(inputs)

    # 保存结果
    save_path = "squeezenet1_1.json"
    with open(save_path, "w") as f:
        f.write(str(main_program))
    print(f"提取完成，文件已生成至: {save_path}")


if __name__ == "__main__":
    extract_squeezenet()
