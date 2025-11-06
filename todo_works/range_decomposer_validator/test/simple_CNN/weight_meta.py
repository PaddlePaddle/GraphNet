class Program_weight_tensor_meta_L_x_:
    name = "L_x_"
    shape = [1, 1, 28, 28]  # Batch size 1, 1 channel, 28x28 image
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.130
    std = 0.308
    data = None


class Program_weight_tensor_meta_L_self_modules_conv1_parameters_weight_:
    name = "L_self_modules_conv1_parameters_weight_"
    shape = [16, 1, 3, 3]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.001
    std = 0.108
    data = None


class Program_weight_tensor_meta_L_self_modules_conv1_parameters_bias_:
    name = "L_self_modules_conv1_parameters_bias_"
    shape = [16]
    dtype = "torch.float32"
    device = "cuda:0"
    mean = 0.0
    std = 0.1
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
