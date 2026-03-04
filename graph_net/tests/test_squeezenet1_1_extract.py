import paddle
import os
from paddle.vision.models import squeezenet1_1

def extract_squeezenet():
    os.environ["FLAGS_enable_pir_api"] = "1"
    paddle.enable_static()
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program):
        inputs = paddle.static.data(name="inputs", shape=[1, 3, 224, 224], dtype="float32")
        model = squeezenet1_1(pretrained=False)
        model(inputs)
    
    save_path = "graph_net/samples/paddle_samples/vision/squeezenet1_1/squeezenet1_1.json"
    with open(save_path, "w") as f:
        f.write(str(main_program))
    print("Extract success")

if __name__ == "__main__":
    extract_squeezenet()
