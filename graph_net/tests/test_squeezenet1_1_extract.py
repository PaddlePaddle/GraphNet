import paddle
import os
from paddle.vision.models import squeezenet1_1


def extract_squeezenet():
    # Enable PIR and static mode
    os.environ["FLAGS_enable_pir_api"] = "1"
    paddle.enable_static()

    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program):
        # 关键点：name 必须等于 "image"
        inputs = paddle.static.data(
            name="image", shape=[1, 3, 224, 224], dtype="float32"
        )
        model = squeezenet1_1(pretrained=False)
        model(inputs)

    # Define save path to root-level paddle_samples
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    save_path = os.path.join(
        base_dir, "paddle_samples/PaddleX/squeezenet1_1/squeezenet1_1.json"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write(str(main_program))
    print(f"Extract success! Saved to: {save_path}")


if __name__ == "__main__":
    extract_squeezenet()
