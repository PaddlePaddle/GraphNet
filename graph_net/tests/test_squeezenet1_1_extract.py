import paddle
import os
import json
from paddle.vision.models import squeezenet1_1


def extract_squeezenet():
    # 1. Environment setup
    os.environ["FLAGS_enable_pir_api"] = "1"
    paddle.enable_static()

    # 2. Capture computation graph
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program):
        inputs = paddle.static.data(
            name="image", shape=[1, 3, 224, 224], dtype="float32"
        )
        model = squeezenet1_1(pretrained=False)
        model(inputs)

    # 3. Get absolute save path
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    target_dir = os.path.join(project_root, "paddle_samples/PaddleX/squeezenet1_1")
    save_path = os.path.join(target_dir, "graph_net.json")

    # 4. Standardize format: Convert program to a valid JSON string
    # We wrap it in a dictionary to ensure valid JSON structure with double quotes
    program_content = str(main_program)
    json_data = {"protocol": "PIR", "program_desc": program_content}

    # 5. Write to file using json.dump (this enforces double quotes for keys)
    os.makedirs(target_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Extraction successful! Standard JSON generated at: {save_path}")


if __name__ == "__main__":
    extract_squeezenet()
