# 这是 subgraph_0 的输出，同时也是 subgraph_1 的输入
class Program_weight_tensor_meta_input_3:
    name = "input_3"
    shape = [1, 16, 14, 14]  # 28x28 经过一次 2x2 池化后变为 14x14
    dtype = "torch.float32"
    device = "cuda:0"
    mean = None
    std = None
    data = None


class Program_weight_tensor_meta_L_self_modules_conv2_parameters_weight_:
    name = "L_self_modules_conv2_parameters_weight_"
    shape = [32, 16, 3, 3]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = -0.002
    std = 0.055
    data = None


class Program_weight_tensor_meta_L_self_modules_conv2_parameters_bias_:
    name = "L_self_modules_conv2_parameters_bias_"
    shape = [32]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.0
    std = 0.05
    data = None
