# 这是 subgraph_1 的输出，同时也是 subgraph_2 的输入
class Program_weight_tensor_meta_input_6:
    name = "input_6"
    shape = [1, 32, 7, 7]  # 14x14 经过一次 2x2 池化后变为 7x7
    dtype = "torch.float32"
    device = "cuda:0"
    mean = None
    std = None
    data = None


class Program_weight_tensor_meta_L_self_modules_fc1_parameters_weight_:
    name = "L_self_modules_fc1_parameters_weight_"
    shape = [128, 1568]  # 1568 = 32 * 7 * 7
    dtype = "torch.float32"
    device = "cuda:0"
    mean = -0.000
    std = 0.025
    data = None


class Program_weight_tensor_meta_L_self_modules_fc1_parameters_bias_:
    name = "L_self_modules_fc1_parameters_bias_"
    shape = [128]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.0
    std = 0.02
    data = None


class Program_weight_tensor_meta_L_self_modules_fc2_parameters_weight_:
    name = "L_self_modules_fc2_parameters_weight_"
    shape = [10, 128]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.001
    std = 0.088
    data = None


class Program_weight_tensor_meta_L_self_modules_fc2_parameters_bias_:
    name = "L_self_modules_fc2_parameters_bias_"
    shape = [10]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.0
    std = 0.09
    data = None
