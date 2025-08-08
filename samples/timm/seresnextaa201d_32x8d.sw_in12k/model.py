import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_15_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_15_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_15_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_16_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_16_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_16_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_17_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_17_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_17_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_18_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_18_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_18_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_19_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_19_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_19_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_20_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_20_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_20_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_21_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_21_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_21_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_22_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_22_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_23_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_23_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_23_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_23_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_23_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_23_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_24_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_24_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_24_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_25_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_25_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_25_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_26_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_26_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_26_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_27_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_27_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_27_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_28_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_28_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_28_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_29_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_29_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_29_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_30_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_30_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_30_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_31_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_31_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_31_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_32_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_32_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_32_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_33_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_33_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_33_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_34_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_34_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_34_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_35_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_35_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_35_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_conv1_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_conv1_modules_1_buffers_running_mean_ = (
            L_self_modules_conv1_modules_1_buffers_running_mean_
        )
        l_self_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_conv1_modules_3_parameters_weight_ = (
            L_self_modules_conv1_modules_3_parameters_weight_
        )
        l_self_modules_conv1_modules_4_buffers_running_mean_ = (
            L_self_modules_conv1_modules_4_buffers_running_mean_
        )
        l_self_modules_conv1_modules_4_buffers_running_var_ = (
            L_self_modules_conv1_modules_4_buffers_running_var_
        )
        l_self_modules_conv1_modules_4_parameters_weight_ = (
            L_self_modules_conv1_modules_4_parameters_weight_
        )
        l_self_modules_conv1_modules_4_parameters_bias_ = (
            L_self_modules_conv1_modules_4_parameters_bias_
        )
        l_self_modules_conv1_modules_6_parameters_weight_ = (
            L_self_modules_conv1_modules_6_parameters_weight_
        )
        l_self_modules_bn1_buffers_running_mean_ = (
            L_self_modules_bn1_buffers_running_mean_
        )
        l_self_modules_bn1_buffers_running_var_ = (
            L_self_modules_bn1_buffers_running_var_
        )
        l_self_modules_bn1_parameters_weight_ = L_self_modules_bn1_parameters_weight_
        l_self_modules_bn1_parameters_bias_ = L_self_modules_bn1_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_5_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_6_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_7_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_7_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_8_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_8_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_8_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_8_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_8_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_8_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_8_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_8_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_8_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_9_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_9_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_9_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_9_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_9_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_9_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_9_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_9_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_9_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_10_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_10_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_10_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_10_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_10_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_10_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_10_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_10_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_10_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_11_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_11_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_11_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_11_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_11_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_11_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_11_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_11_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_11_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_12_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_12_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_12_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_12_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_12_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_12_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_12_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_12_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_12_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_13_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_13_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_13_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_13_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_13_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_13_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_13_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_13_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_13_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_14_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_14_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_14_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_14_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_14_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_14_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_14_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_14_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_14_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_15_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_15_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_15_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_15_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_15_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_15_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_15_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_15_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_15_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_16_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_16_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_16_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_16_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_16_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_16_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_16_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_16_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_16_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_17_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_17_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_17_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_17_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_17_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_17_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_17_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_17_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_17_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_18_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_18_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_18_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_18_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_18_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_18_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_18_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_18_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_18_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_19_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_19_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_19_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_19_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_19_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_19_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_19_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_19_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_19_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_20_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_20_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_20_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_20_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_20_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_20_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_20_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_20_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_20_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_21_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_21_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_21_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_21_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_21_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_21_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_21_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_21_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_21_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_22_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_22_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_22_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_22_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_22_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_22_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_22_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_22_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_22_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer2_modules_23_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_23_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_23_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_23_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_23_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_23_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_23_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_23_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_23_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_23_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_23_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_23_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_24_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_24_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_24_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_25_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_25_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_25_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_26_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_26_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_26_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_27_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_27_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_27_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_28_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_28_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_28_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_29_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_29_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_29_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_30_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_30_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_30_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_31_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_31_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_31_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_32_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_32_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_32_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_33_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_33_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_33_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_34_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_34_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_34_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_35_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_35_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_35_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_layer4_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_3_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_conv1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_1_parameters_weight_
        ) = l_self_modules_conv1_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_conv1_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_conv1_modules_3_parameters_weight_ = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_conv1_modules_4_buffers_running_mean_,
            l_self_modules_conv1_modules_4_buffers_running_var_,
            l_self_modules_conv1_modules_4_parameters_weight_,
            l_self_modules_conv1_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_conv1_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_4_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_4_parameters_weight_
        ) = l_self_modules_conv1_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_conv1_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_conv1_modules_6_parameters_weight_ = None
        x = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(x, inplace=True)
        x = None
        x_2 = torch._C._nn.avg_pool2d(x_1, 2, 2, 0, False, True, None)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_5 = l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
        x_se = x_10.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = (
            l_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = (
            l_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_11 = x_10 * sigmoid
        x_10 = sigmoid = None
        input_8 = torch.conv2d(
            x_2,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_11 += input_9
        x_12 = x_11
        x_11 = input_9 = None
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = None
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_16 = l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = None
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = None
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = None
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = None
        x_se_4 = x_21.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = (
            l_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = (
            l_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_22 = x_21 * sigmoid_1
        x_21 = sigmoid_1 = None
        x_22 += x_13
        x_23 = x_22
        x_22 = x_13 = None
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_ = None
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_ = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_27 = l_self_modules_layer1_modules_2_modules_conv2_parameters_weight_ = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn2_parameters_bias_ = None
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_ = None
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_ = None
        x_se_8 = x_32.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = (
            l_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = (
            l_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_33 = x_32 * sigmoid_2
        x_32 = sigmoid_2 = None
        x_33 += x_24
        x_34 = x_33
        x_33 = x_24 = None
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_38 = l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = None
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch._C._nn.avg_pool2d(x_41, 2, 2, 0, False, True, None)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        x_se_12 = x_44.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = (
            l_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = (
            l_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_45 = x_44 * sigmoid_3
        x_44 = sigmoid_3 = None
        input_10 = torch._C._nn.avg_pool2d(x_35, 2, 2, 0, True, False, None)
        x_35 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_45 += input_12
        x_46 = x_45
        x_45 = input_12 = None
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_50 = l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = None
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = None
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        x_se_16 = x_55.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = (
            l_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = (
            l_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_56 = x_55 * sigmoid_4
        x_55 = sigmoid_4 = None
        x_56 += x_47
        x_57 = x_56
        x_56 = x_47 = None
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = None
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = None
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_61 = l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_ = None
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_ = None
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = None
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = None
        x_se_20 = x_66.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = (
            l_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = (
            l_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_67 = x_66 * sigmoid_5
        x_66 = sigmoid_5 = None
        x_67 += x_58
        x_68 = x_67
        x_67 = x_58 = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = None
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = None
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_72 = l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_ = None
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_ = None
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = None
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = None
        x_se_24 = x_77.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = (
            l_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = (
            l_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_78 = x_77 * sigmoid_6
        x_77 = sigmoid_6 = None
        x_78 += x_69
        x_79 = x_78
        x_78 = x_69 = None
        x_80 = torch.nn.functional.relu(x_79, inplace=True)
        x_79 = None
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_ = None
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = (
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_ = None
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_83 = l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_ = None
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_ = None
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_ = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_ = None
        x_se_28 = x_88.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = (
            l_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_4_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_30 = torch.nn.functional.relu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = (
            l_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_4_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_89 = x_88 * sigmoid_7
        x_88 = sigmoid_7 = None
        x_89 += x_80
        x_90 = x_89
        x_89 = x_80 = None
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_ = None
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = (
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_ = None
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_94 = l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_ = None
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_ = None
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_ = None
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = (
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_ = None
        x_se_32 = x_99.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = (
            l_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_5_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_34 = torch.nn.functional.relu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = (
            l_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_5_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_100 = x_99 * sigmoid_8
        x_99 = sigmoid_8 = None
        x_100 += x_91
        x_101 = x_100
        x_100 = x_91 = None
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_ = None
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = (
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_ = None
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_105 = l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_ = None
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = (
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_ = None
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_ = None
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = (
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_ = None
        x_se_36 = x_110.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = (
            l_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_6_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_38 = torch.nn.functional.relu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = (
            l_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_6_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_111 = x_110 * sigmoid_9
        x_110 = sigmoid_9 = None
        x_111 += x_102
        x_112 = x_111
        x_111 = x_102 = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_ = None
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = (
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_ = None
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_116 = l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_ = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_ = None
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_ = None
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = (
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_ = None
        x_se_40 = x_121.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = (
            l_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_7_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_42 = torch.nn.functional.relu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = (
            l_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_7_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_122 = x_121 * sigmoid_10
        x_121 = sigmoid_10 = None
        x_122 += x_113
        x_123 = x_122
        x_122 = x_113 = None
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_layer2_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_8_modules_conv1_parameters_weight_ = None
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = (
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_8_modules_bn1_parameters_bias_ = None
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_layer2_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_127 = l_self_modules_layer2_modules_8_modules_conv2_parameters_weight_ = None
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = (
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_8_modules_bn2_parameters_bias_ = None
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_layer2_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_layer2_modules_8_modules_conv3_parameters_weight_ = None
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = (
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_8_modules_bn3_parameters_bias_ = None
        x_se_44 = x_132.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = (
            l_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_8_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_46 = torch.nn.functional.relu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = (
            l_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_8_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_133 = x_132 * sigmoid_11
        x_132 = sigmoid_11 = None
        x_133 += x_124
        x_134 = x_133
        x_133 = x_124 = None
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_layer2_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_9_modules_conv1_parameters_weight_ = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = (
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_9_modules_bn1_parameters_bias_ = None
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_layer2_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_138 = l_self_modules_layer2_modules_9_modules_conv2_parameters_weight_ = None
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = (
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_9_modules_bn2_parameters_bias_ = None
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_layer2_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_layer2_modules_9_modules_conv3_parameters_weight_ = None
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = (
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_9_modules_bn3_parameters_bias_ = None
        x_se_48 = x_143.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = (
            l_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_9_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_50 = torch.nn.functional.relu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = (
            l_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_9_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_144 = x_143 * sigmoid_12
        x_143 = sigmoid_12 = None
        x_144 += x_135
        x_145 = x_144
        x_144 = x_135 = None
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_layer2_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_10_modules_conv1_parameters_weight_ = None
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = (
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_10_modules_bn1_parameters_bias_ = None
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_layer2_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_149 = l_self_modules_layer2_modules_10_modules_conv2_parameters_weight_ = None
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = (
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_10_modules_bn2_parameters_bias_ = None
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_layer2_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_layer2_modules_10_modules_conv3_parameters_weight_ = None
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = (
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_10_modules_bn3_parameters_bias_ = None
        x_se_52 = x_154.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = (
            l_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_10_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_54 = torch.nn.functional.relu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = (
            l_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_10_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_155 = x_154 * sigmoid_13
        x_154 = sigmoid_13 = None
        x_155 += x_146
        x_156 = x_155
        x_155 = x_146 = None
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_layer2_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_11_modules_conv1_parameters_weight_ = None
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = (
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_11_modules_bn1_parameters_bias_ = None
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_layer2_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_160 = l_self_modules_layer2_modules_11_modules_conv2_parameters_weight_ = None
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = (
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_11_modules_bn2_parameters_bias_ = None
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_layer2_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_layer2_modules_11_modules_conv3_parameters_weight_ = None
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = (
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_11_modules_bn3_parameters_bias_ = None
        x_se_56 = x_165.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = (
            l_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_11_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_58 = torch.nn.functional.relu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = (
            l_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_11_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        x_166 = x_165 * sigmoid_14
        x_165 = sigmoid_14 = None
        x_166 += x_157
        x_167 = x_166
        x_166 = x_157 = None
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_layer2_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_12_modules_conv1_parameters_weight_ = None
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = (
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_12_modules_bn1_parameters_bias_ = None
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_layer2_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_171 = l_self_modules_layer2_modules_12_modules_conv2_parameters_weight_ = None
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = (
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_12_modules_bn2_parameters_bias_ = None
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_layer2_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_layer2_modules_12_modules_conv3_parameters_weight_ = None
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = (
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_12_modules_bn3_parameters_bias_ = None
        x_se_60 = x_176.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = (
            l_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_12_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_62 = torch.nn.functional.relu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = (
            l_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_12_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        x_177 = x_176 * sigmoid_15
        x_176 = sigmoid_15 = None
        x_177 += x_168
        x_178 = x_177
        x_177 = x_168 = None
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_layer2_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_13_modules_conv1_parameters_weight_ = None
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = (
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_13_modules_bn1_parameters_bias_ = None
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_layer2_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_182 = l_self_modules_layer2_modules_13_modules_conv2_parameters_weight_ = None
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = (
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_13_modules_bn2_parameters_bias_ = None
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_layer2_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_layer2_modules_13_modules_conv3_parameters_weight_ = None
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = (
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_13_modules_bn3_parameters_bias_ = None
        x_se_64 = x_187.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = (
            l_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_13_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_66 = torch.nn.functional.relu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = (
            l_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_13_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        x_188 = x_187 * sigmoid_16
        x_187 = sigmoid_16 = None
        x_188 += x_179
        x_189 = x_188
        x_188 = x_179 = None
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_layer2_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_14_modules_conv1_parameters_weight_ = None
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = (
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_14_modules_bn1_parameters_bias_ = None
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_layer2_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_193 = l_self_modules_layer2_modules_14_modules_conv2_parameters_weight_ = None
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = (
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_14_modules_bn2_parameters_bias_ = None
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_layer2_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_layer2_modules_14_modules_conv3_parameters_weight_ = None
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = (
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_14_modules_bn3_parameters_bias_ = None
        x_se_68 = x_198.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = (
            l_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_14_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_70 = torch.nn.functional.relu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = (
            l_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_14_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        x_199 = x_198 * sigmoid_17
        x_198 = sigmoid_17 = None
        x_199 += x_190
        x_200 = x_199
        x_199 = x_190 = None
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_layer2_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_15_modules_conv1_parameters_weight_ = None
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = (
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_15_modules_bn1_parameters_bias_ = None
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_layer2_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_204 = l_self_modules_layer2_modules_15_modules_conv2_parameters_weight_ = None
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_205 = (
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_15_modules_bn2_parameters_bias_ = None
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_layer2_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_layer2_modules_15_modules_conv3_parameters_weight_ = None
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = (
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_15_modules_bn3_parameters_bias_ = None
        x_se_72 = x_209.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = (
            l_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_15_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_74 = torch.nn.functional.relu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = (
            l_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_15_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        x_210 = x_209 * sigmoid_18
        x_209 = sigmoid_18 = None
        x_210 += x_201
        x_211 = x_210
        x_210 = x_201 = None
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_layer2_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_16_modules_conv1_parameters_weight_ = None
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_213 = (
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_16_modules_bn1_parameters_bias_ = None
        x_215 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_layer2_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_215 = l_self_modules_layer2_modules_16_modules_conv2_parameters_weight_ = None
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = (
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_16_modules_bn2_parameters_bias_ = None
        x_218 = torch.nn.functional.relu(x_217, inplace=True)
        x_217 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_layer2_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_layer2_modules_16_modules_conv3_parameters_weight_ = None
        x_220 = torch.nn.functional.batch_norm(
            x_219,
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_219 = (
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_16_modules_bn3_parameters_bias_ = None
        x_se_76 = x_220.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = (
            l_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_16_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_78 = torch.nn.functional.relu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = (
            l_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_16_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        x_221 = x_220 * sigmoid_19
        x_220 = sigmoid_19 = None
        x_221 += x_212
        x_222 = x_221
        x_221 = x_212 = None
        x_223 = torch.nn.functional.relu(x_222, inplace=True)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_layer2_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_17_modules_conv1_parameters_weight_ = None
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = (
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_17_modules_bn1_parameters_bias_ = None
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_layer2_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_226 = l_self_modules_layer2_modules_17_modules_conv2_parameters_weight_ = None
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = (
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_17_modules_bn2_parameters_bias_ = None
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_layer2_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_layer2_modules_17_modules_conv3_parameters_weight_ = None
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = (
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_17_modules_bn3_parameters_bias_ = None
        x_se_80 = x_231.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = (
            l_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_17_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_82 = torch.nn.functional.relu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = (
            l_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_17_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        x_232 = x_231 * sigmoid_20
        x_231 = sigmoid_20 = None
        x_232 += x_223
        x_233 = x_232
        x_232 = x_223 = None
        x_234 = torch.nn.functional.relu(x_233, inplace=True)
        x_233 = None
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_layer2_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_18_modules_conv1_parameters_weight_ = None
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_235 = (
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_18_modules_bn1_parameters_bias_ = None
        x_237 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_layer2_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_237 = l_self_modules_layer2_modules_18_modules_conv2_parameters_weight_ = None
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = (
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_18_modules_bn2_parameters_bias_ = None
        x_240 = torch.nn.functional.relu(x_239, inplace=True)
        x_239 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_layer2_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_layer2_modules_18_modules_conv3_parameters_weight_ = None
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = (
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_18_modules_bn3_parameters_bias_ = None
        x_se_84 = x_242.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = (
            l_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_18_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_86 = torch.nn.functional.relu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = (
            l_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_18_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        x_243 = x_242 * sigmoid_21
        x_242 = sigmoid_21 = None
        x_243 += x_234
        x_244 = x_243
        x_243 = x_234 = None
        x_245 = torch.nn.functional.relu(x_244, inplace=True)
        x_244 = None
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_layer2_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_19_modules_conv1_parameters_weight_ = None
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_246 = (
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_19_modules_bn1_parameters_bias_ = None
        x_248 = torch.nn.functional.relu(x_247, inplace=True)
        x_247 = None
        x_249 = torch.conv2d(
            x_248,
            l_self_modules_layer2_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_248 = l_self_modules_layer2_modules_19_modules_conv2_parameters_weight_ = None
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = (
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_19_modules_bn2_parameters_bias_ = None
        x_251 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_layer2_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_layer2_modules_19_modules_conv3_parameters_weight_ = None
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = (
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_19_modules_bn3_parameters_bias_ = None
        x_se_88 = x_253.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = (
            l_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_19_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_90 = torch.nn.functional.relu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = (
            l_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_19_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_22 = x_se_91.sigmoid()
        x_se_91 = None
        x_254 = x_253 * sigmoid_22
        x_253 = sigmoid_22 = None
        x_254 += x_245
        x_255 = x_254
        x_254 = x_245 = None
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_layer2_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_20_modules_conv1_parameters_weight_ = None
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = (
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_20_modules_bn1_parameters_bias_ = None
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_layer2_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_259 = l_self_modules_layer2_modules_20_modules_conv2_parameters_weight_ = None
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = (
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_20_modules_bn2_parameters_bias_ = None
        x_262 = torch.nn.functional.relu(x_261, inplace=True)
        x_261 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_layer2_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_layer2_modules_20_modules_conv3_parameters_weight_ = None
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = (
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_20_modules_bn3_parameters_bias_ = None
        x_se_92 = x_264.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = (
            l_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_20_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_94 = torch.nn.functional.relu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = (
            l_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_20_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_23 = x_se_95.sigmoid()
        x_se_95 = None
        x_265 = x_264 * sigmoid_23
        x_264 = sigmoid_23 = None
        x_265 += x_256
        x_266 = x_265
        x_265 = x_256 = None
        x_267 = torch.nn.functional.relu(x_266, inplace=True)
        x_266 = None
        x_268 = torch.conv2d(
            x_267,
            l_self_modules_layer2_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_21_modules_conv1_parameters_weight_ = None
        x_269 = torch.nn.functional.batch_norm(
            x_268,
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_268 = (
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_21_modules_bn1_parameters_bias_ = None
        x_270 = torch.nn.functional.relu(x_269, inplace=True)
        x_269 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_layer2_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_270 = l_self_modules_layer2_modules_21_modules_conv2_parameters_weight_ = None
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = (
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_21_modules_bn2_parameters_bias_ = None
        x_273 = torch.nn.functional.relu(x_272, inplace=True)
        x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_layer2_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_layer2_modules_21_modules_conv3_parameters_weight_ = None
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = (
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_21_modules_bn3_parameters_bias_ = None
        x_se_96 = x_275.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = (
            l_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_21_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_98 = torch.nn.functional.relu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = (
            l_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_21_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_24 = x_se_99.sigmoid()
        x_se_99 = None
        x_276 = x_275 * sigmoid_24
        x_275 = sigmoid_24 = None
        x_276 += x_267
        x_277 = x_276
        x_276 = x_267 = None
        x_278 = torch.nn.functional.relu(x_277, inplace=True)
        x_277 = None
        x_279 = torch.conv2d(
            x_278,
            l_self_modules_layer2_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_22_modules_conv1_parameters_weight_ = None
        x_280 = torch.nn.functional.batch_norm(
            x_279,
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_279 = (
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_22_modules_bn1_parameters_bias_ = None
        x_281 = torch.nn.functional.relu(x_280, inplace=True)
        x_280 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_layer2_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_281 = l_self_modules_layer2_modules_22_modules_conv2_parameters_weight_ = None
        x_283 = torch.nn.functional.batch_norm(
            x_282,
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_282 = (
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_22_modules_bn2_parameters_bias_ = None
        x_284 = torch.nn.functional.relu(x_283, inplace=True)
        x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_layer2_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_layer2_modules_22_modules_conv3_parameters_weight_ = None
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_285 = (
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_22_modules_bn3_parameters_bias_ = None
        x_se_100 = x_286.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = (
            l_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_22_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_102 = torch.nn.functional.relu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = (
            l_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_22_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_25 = x_se_103.sigmoid()
        x_se_103 = None
        x_287 = x_286 * sigmoid_25
        x_286 = sigmoid_25 = None
        x_287 += x_278
        x_288 = x_287
        x_287 = x_278 = None
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_layer2_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_23_modules_conv1_parameters_weight_ = None
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = (
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_23_modules_bn1_parameters_bias_ = None
        x_292 = torch.nn.functional.relu(x_291, inplace=True)
        x_291 = None
        x_293 = torch.conv2d(
            x_292,
            l_self_modules_layer2_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_292 = l_self_modules_layer2_modules_23_modules_conv2_parameters_weight_ = None
        x_294 = torch.nn.functional.batch_norm(
            x_293,
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_293 = (
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_23_modules_bn2_parameters_bias_ = None
        x_295 = torch.nn.functional.relu(x_294, inplace=True)
        x_294 = None
        x_296 = torch.conv2d(
            x_295,
            l_self_modules_layer2_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_295 = l_self_modules_layer2_modules_23_modules_conv3_parameters_weight_ = None
        x_297 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_296 = (
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_23_modules_bn3_parameters_bias_ = None
        x_se_104 = x_297.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = (
            l_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_23_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_106 = torch.nn.functional.relu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = (
            l_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_23_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_26 = x_se_107.sigmoid()
        x_se_107 = None
        x_298 = x_297 * sigmoid_26
        x_297 = sigmoid_26 = None
        x_298 += x_289
        x_299 = x_298
        x_298 = x_289 = None
        x_300 = torch.nn.functional.relu(x_299, inplace=True)
        x_299 = None
        x_301 = torch.conv2d(
            x_300,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_303 = torch.nn.functional.relu(x_302, inplace=True)
        x_302 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_303 = l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_305 = torch.nn.functional.batch_norm(
            x_304,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_304 = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        x_306 = torch.nn.functional.relu(x_305, inplace=True)
        x_305 = None
        x_307 = torch._C._nn.avg_pool2d(x_306, 2, 2, 0, False, True, None)
        x_306 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        x_se_108 = x_309.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_110 = torch.nn.functional.relu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_27 = x_se_111.sigmoid()
        x_se_111 = None
        x_310 = x_309 * sigmoid_27
        x_309 = sigmoid_27 = None
        input_13 = torch._C._nn.avg_pool2d(x_300, 2, 2, 0, True, False, None)
        x_300 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_310 += input_15
        x_311 = x_310
        x_310 = input_15 = None
        x_312 = torch.nn.functional.relu(x_311, inplace=True)
        x_311 = None
        x_313 = torch.conv2d(
            x_312,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_314 = torch.nn.functional.batch_norm(
            x_313,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_313 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_315 = torch.nn.functional.relu(x_314, inplace=True)
        x_314 = None
        x_316 = torch.conv2d(
            x_315,
            l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_315 = l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_317 = torch.nn.functional.batch_norm(
            x_316,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_316 = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_318 = torch.nn.functional.relu(x_317, inplace=True)
        x_317 = None
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_318 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        x_320 = torch.nn.functional.batch_norm(
            x_319,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_319 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        x_se_112 = x_320.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_114 = torch.nn.functional.relu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_28 = x_se_115.sigmoid()
        x_se_115 = None
        x_321 = x_320 * sigmoid_28
        x_320 = sigmoid_28 = None
        x_321 += x_312
        x_322 = x_321
        x_321 = x_312 = None
        x_323 = torch.nn.functional.relu(x_322, inplace=True)
        x_322 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        x_325 = torch.nn.functional.batch_norm(
            x_324,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_324 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        x_326 = torch.nn.functional.relu(x_325, inplace=True)
        x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_326 = l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_ = None
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_327 = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_ = None
        x_329 = torch.nn.functional.relu(x_328, inplace=True)
        x_328 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        x_331 = torch.nn.functional.batch_norm(
            x_330,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_330 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        x_se_116 = x_331.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_118 = torch.nn.functional.relu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_29 = x_se_119.sigmoid()
        x_se_119 = None
        x_332 = x_331 * sigmoid_29
        x_331 = sigmoid_29 = None
        x_332 += x_323
        x_333 = x_332
        x_332 = x_323 = None
        x_334 = torch.nn.functional.relu(x_333, inplace=True)
        x_333 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_335 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        x_337 = torch.nn.functional.relu(x_336, inplace=True)
        x_336 = None
        x_338 = torch.conv2d(
            x_337,
            l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_337 = l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_ = None
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_ = None
        x_340 = torch.nn.functional.relu(x_339, inplace=True)
        x_339 = None
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_340 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_341 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        x_se_120 = x_342.mean((2, 3), keepdim=True)
        x_se_121 = torch.conv2d(
            x_se_120,
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_120 = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_122 = torch.nn.functional.relu(x_se_121, inplace=True)
        x_se_121 = None
        x_se_123 = torch.conv2d(
            x_se_122,
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_122 = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_30 = x_se_123.sigmoid()
        x_se_123 = None
        x_343 = x_342 * sigmoid_30
        x_342 = sigmoid_30 = None
        x_343 += x_334
        x_344 = x_343
        x_343 = x_334 = None
        x_345 = torch.nn.functional.relu(x_344, inplace=True)
        x_344 = None
        x_346 = torch.conv2d(
            x_345,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        x_347 = torch.nn.functional.batch_norm(
            x_346,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_346 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        x_348 = torch.nn.functional.relu(x_347, inplace=True)
        x_347 = None
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_348 = l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_ = None
        x_350 = torch.nn.functional.batch_norm(
            x_349,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_349 = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_ = None
        x_351 = torch.nn.functional.relu(x_350, inplace=True)
        x_350 = None
        x_352 = torch.conv2d(
            x_351,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_351 = l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = None
        x_353 = torch.nn.functional.batch_norm(
            x_352,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_352 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        x_se_124 = x_353.mean((2, 3), keepdim=True)
        x_se_125 = torch.conv2d(
            x_se_124,
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_124 = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_126 = torch.nn.functional.relu(x_se_125, inplace=True)
        x_se_125 = None
        x_se_127 = torch.conv2d(
            x_se_126,
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_126 = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_31 = x_se_127.sigmoid()
        x_se_127 = None
        x_354 = x_353 * sigmoid_31
        x_353 = sigmoid_31 = None
        x_354 += x_345
        x_355 = x_354
        x_354 = x_345 = None
        x_356 = torch.nn.functional.relu(x_355, inplace=True)
        x_355 = None
        x_357 = torch.conv2d(
            x_356,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        x_358 = torch.nn.functional.batch_norm(
            x_357,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_357 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        x_359 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        x_360 = torch.conv2d(
            x_359,
            l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_359 = l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_ = None
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_360 = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_ = None
        x_362 = torch.nn.functional.relu(x_361, inplace=True)
        x_361 = None
        x_363 = torch.conv2d(
            x_362,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_362 = l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = None
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_363 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        x_se_128 = x_364.mean((2, 3), keepdim=True)
        x_se_129 = torch.conv2d(
            x_se_128,
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_128 = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_130 = torch.nn.functional.relu(x_se_129, inplace=True)
        x_se_129 = None
        x_se_131 = torch.conv2d(
            x_se_130,
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_130 = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_32 = x_se_131.sigmoid()
        x_se_131 = None
        x_365 = x_364 * sigmoid_32
        x_364 = sigmoid_32 = None
        x_365 += x_356
        x_366 = x_365
        x_365 = x_356 = None
        x_367 = torch.nn.functional.relu(x_366, inplace=True)
        x_366 = None
        x_368 = torch.conv2d(
            x_367,
            l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = None
        x_369 = torch.nn.functional.batch_norm(
            x_368,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_368 = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = None
        x_370 = torch.nn.functional.relu(x_369, inplace=True)
        x_369 = None
        x_371 = torch.conv2d(
            x_370,
            l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_370 = l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_ = None
        x_372 = torch.nn.functional.batch_norm(
            x_371,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_371 = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_ = None
        x_373 = torch.nn.functional.relu(x_372, inplace=True)
        x_372 = None
        x_374 = torch.conv2d(
            x_373,
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_373 = l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = None
        x_375 = torch.nn.functional.batch_norm(
            x_374,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_374 = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = None
        x_se_132 = x_375.mean((2, 3), keepdim=True)
        x_se_133 = torch.conv2d(
            x_se_132,
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_132 = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_134 = torch.nn.functional.relu(x_se_133, inplace=True)
        x_se_133 = None
        x_se_135 = torch.conv2d(
            x_se_134,
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_134 = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_33 = x_se_135.sigmoid()
        x_se_135 = None
        x_376 = x_375 * sigmoid_33
        x_375 = sigmoid_33 = None
        x_376 += x_367
        x_377 = x_376
        x_376 = x_367 = None
        x_378 = torch.nn.functional.relu(x_377, inplace=True)
        x_377 = None
        x_379 = torch.conv2d(
            x_378,
            l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = None
        x_380 = torch.nn.functional.batch_norm(
            x_379,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_379 = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = None
        x_381 = torch.nn.functional.relu(x_380, inplace=True)
        x_380 = None
        x_382 = torch.conv2d(
            x_381,
            l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_381 = l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_ = None
        x_383 = torch.nn.functional.batch_norm(
            x_382,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_382 = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_ = None
        x_384 = torch.nn.functional.relu(x_383, inplace=True)
        x_383 = None
        x_385 = torch.conv2d(
            x_384,
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_384 = l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = None
        x_386 = torch.nn.functional.batch_norm(
            x_385,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_385 = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = None
        x_se_136 = x_386.mean((2, 3), keepdim=True)
        x_se_137 = torch.conv2d(
            x_se_136,
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_136 = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_138 = torch.nn.functional.relu(x_se_137, inplace=True)
        x_se_137 = None
        x_se_139 = torch.conv2d(
            x_se_138,
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_138 = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_34 = x_se_139.sigmoid()
        x_se_139 = None
        x_387 = x_386 * sigmoid_34
        x_386 = sigmoid_34 = None
        x_387 += x_378
        x_388 = x_387
        x_387 = x_378 = None
        x_389 = torch.nn.functional.relu(x_388, inplace=True)
        x_388 = None
        x_390 = torch.conv2d(
            x_389,
            l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = None
        x_391 = torch.nn.functional.batch_norm(
            x_390,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_390 = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = None
        x_392 = torch.nn.functional.relu(x_391, inplace=True)
        x_391 = None
        x_393 = torch.conv2d(
            x_392,
            l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_392 = l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_ = None
        x_394 = torch.nn.functional.batch_norm(
            x_393,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_393 = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_ = None
        x_395 = torch.nn.functional.relu(x_394, inplace=True)
        x_394 = None
        x_396 = torch.conv2d(
            x_395,
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_395 = l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = None
        x_397 = torch.nn.functional.batch_norm(
            x_396,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_396 = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = None
        x_se_140 = x_397.mean((2, 3), keepdim=True)
        x_se_141 = torch.conv2d(
            x_se_140,
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_140 = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_142 = torch.nn.functional.relu(x_se_141, inplace=True)
        x_se_141 = None
        x_se_143 = torch.conv2d(
            x_se_142,
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_142 = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_35 = x_se_143.sigmoid()
        x_se_143 = None
        x_398 = x_397 * sigmoid_35
        x_397 = sigmoid_35 = None
        x_398 += x_389
        x_399 = x_398
        x_398 = x_389 = None
        x_400 = torch.nn.functional.relu(x_399, inplace=True)
        x_399 = None
        x_401 = torch.conv2d(
            x_400,
            l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = None
        x_402 = torch.nn.functional.batch_norm(
            x_401,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_401 = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = None
        x_403 = torch.nn.functional.relu(x_402, inplace=True)
        x_402 = None
        x_404 = torch.conv2d(
            x_403,
            l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_403 = l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_ = None
        x_405 = torch.nn.functional.batch_norm(
            x_404,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_404 = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_ = None
        x_406 = torch.nn.functional.relu(x_405, inplace=True)
        x_405 = None
        x_407 = torch.conv2d(
            x_406,
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_406 = l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = None
        x_408 = torch.nn.functional.batch_norm(
            x_407,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_407 = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = None
        x_se_144 = x_408.mean((2, 3), keepdim=True)
        x_se_145 = torch.conv2d(
            x_se_144,
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_144 = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_146 = torch.nn.functional.relu(x_se_145, inplace=True)
        x_se_145 = None
        x_se_147 = torch.conv2d(
            x_se_146,
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_146 = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_36 = x_se_147.sigmoid()
        x_se_147 = None
        x_409 = x_408 * sigmoid_36
        x_408 = sigmoid_36 = None
        x_409 += x_400
        x_410 = x_409
        x_409 = x_400 = None
        x_411 = torch.nn.functional.relu(x_410, inplace=True)
        x_410 = None
        x_412 = torch.conv2d(
            x_411,
            l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = None
        x_413 = torch.nn.functional.batch_norm(
            x_412,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_412 = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = None
        x_414 = torch.nn.functional.relu(x_413, inplace=True)
        x_413 = None
        x_415 = torch.conv2d(
            x_414,
            l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_414 = l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_ = None
        x_416 = torch.nn.functional.batch_norm(
            x_415,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_415 = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_ = None
        x_417 = torch.nn.functional.relu(x_416, inplace=True)
        x_416 = None
        x_418 = torch.conv2d(
            x_417,
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_417 = l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = None
        x_419 = torch.nn.functional.batch_norm(
            x_418,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_418 = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = None
        x_se_148 = x_419.mean((2, 3), keepdim=True)
        x_se_149 = torch.conv2d(
            x_se_148,
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_148 = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_150 = torch.nn.functional.relu(x_se_149, inplace=True)
        x_se_149 = None
        x_se_151 = torch.conv2d(
            x_se_150,
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_150 = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_37 = x_se_151.sigmoid()
        x_se_151 = None
        x_420 = x_419 * sigmoid_37
        x_419 = sigmoid_37 = None
        x_420 += x_411
        x_421 = x_420
        x_420 = x_411 = None
        x_422 = torch.nn.functional.relu(x_421, inplace=True)
        x_421 = None
        x_423 = torch.conv2d(
            x_422,
            l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = None
        x_424 = torch.nn.functional.batch_norm(
            x_423,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_423 = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = None
        x_425 = torch.nn.functional.relu(x_424, inplace=True)
        x_424 = None
        x_426 = torch.conv2d(
            x_425,
            l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_425 = l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_ = None
        x_427 = torch.nn.functional.batch_norm(
            x_426,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_426 = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_ = None
        x_428 = torch.nn.functional.relu(x_427, inplace=True)
        x_427 = None
        x_429 = torch.conv2d(
            x_428,
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_428 = l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = None
        x_430 = torch.nn.functional.batch_norm(
            x_429,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_429 = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = None
        x_se_152 = x_430.mean((2, 3), keepdim=True)
        x_se_153 = torch.conv2d(
            x_se_152,
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_152 = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_154 = torch.nn.functional.relu(x_se_153, inplace=True)
        x_se_153 = None
        x_se_155 = torch.conv2d(
            x_se_154,
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_154 = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_38 = x_se_155.sigmoid()
        x_se_155 = None
        x_431 = x_430 * sigmoid_38
        x_430 = sigmoid_38 = None
        x_431 += x_422
        x_432 = x_431
        x_431 = x_422 = None
        x_433 = torch.nn.functional.relu(x_432, inplace=True)
        x_432 = None
        x_434 = torch.conv2d(
            x_433,
            l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = None
        x_435 = torch.nn.functional.batch_norm(
            x_434,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_434 = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = None
        x_436 = torch.nn.functional.relu(x_435, inplace=True)
        x_435 = None
        x_437 = torch.conv2d(
            x_436,
            l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_436 = l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_ = None
        x_438 = torch.nn.functional.batch_norm(
            x_437,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_437 = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_ = None
        x_439 = torch.nn.functional.relu(x_438, inplace=True)
        x_438 = None
        x_440 = torch.conv2d(
            x_439,
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_439 = l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = None
        x_441 = torch.nn.functional.batch_norm(
            x_440,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_440 = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = None
        x_se_156 = x_441.mean((2, 3), keepdim=True)
        x_se_157 = torch.conv2d(
            x_se_156,
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_156 = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_158 = torch.nn.functional.relu(x_se_157, inplace=True)
        x_se_157 = None
        x_se_159 = torch.conv2d(
            x_se_158,
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_158 = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_39 = x_se_159.sigmoid()
        x_se_159 = None
        x_442 = x_441 * sigmoid_39
        x_441 = sigmoid_39 = None
        x_442 += x_433
        x_443 = x_442
        x_442 = x_433 = None
        x_444 = torch.nn.functional.relu(x_443, inplace=True)
        x_443 = None
        x_445 = torch.conv2d(
            x_444,
            l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = None
        x_446 = torch.nn.functional.batch_norm(
            x_445,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_445 = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = None
        x_447 = torch.nn.functional.relu(x_446, inplace=True)
        x_446 = None
        x_448 = torch.conv2d(
            x_447,
            l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_447 = l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_ = None
        x_449 = torch.nn.functional.batch_norm(
            x_448,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_448 = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_ = None
        x_450 = torch.nn.functional.relu(x_449, inplace=True)
        x_449 = None
        x_451 = torch.conv2d(
            x_450,
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_450 = l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = None
        x_452 = torch.nn.functional.batch_norm(
            x_451,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_451 = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = None
        x_se_160 = x_452.mean((2, 3), keepdim=True)
        x_se_161 = torch.conv2d(
            x_se_160,
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_160 = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_162 = torch.nn.functional.relu(x_se_161, inplace=True)
        x_se_161 = None
        x_se_163 = torch.conv2d(
            x_se_162,
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_162 = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_40 = x_se_163.sigmoid()
        x_se_163 = None
        x_453 = x_452 * sigmoid_40
        x_452 = sigmoid_40 = None
        x_453 += x_444
        x_454 = x_453
        x_453 = x_444 = None
        x_455 = torch.nn.functional.relu(x_454, inplace=True)
        x_454 = None
        x_456 = torch.conv2d(
            x_455,
            l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = None
        x_457 = torch.nn.functional.batch_norm(
            x_456,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_456 = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = None
        x_458 = torch.nn.functional.relu(x_457, inplace=True)
        x_457 = None
        x_459 = torch.conv2d(
            x_458,
            l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_458 = l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_ = None
        x_460 = torch.nn.functional.batch_norm(
            x_459,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_459 = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_ = None
        x_461 = torch.nn.functional.relu(x_460, inplace=True)
        x_460 = None
        x_462 = torch.conv2d(
            x_461,
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_461 = l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = None
        x_463 = torch.nn.functional.batch_norm(
            x_462,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_462 = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = None
        x_se_164 = x_463.mean((2, 3), keepdim=True)
        x_se_165 = torch.conv2d(
            x_se_164,
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_164 = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_166 = torch.nn.functional.relu(x_se_165, inplace=True)
        x_se_165 = None
        x_se_167 = torch.conv2d(
            x_se_166,
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_166 = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_41 = x_se_167.sigmoid()
        x_se_167 = None
        x_464 = x_463 * sigmoid_41
        x_463 = sigmoid_41 = None
        x_464 += x_455
        x_465 = x_464
        x_464 = x_455 = None
        x_466 = torch.nn.functional.relu(x_465, inplace=True)
        x_465 = None
        x_467 = torch.conv2d(
            x_466,
            l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = None
        x_468 = torch.nn.functional.batch_norm(
            x_467,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_467 = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = None
        x_469 = torch.nn.functional.relu(x_468, inplace=True)
        x_468 = None
        x_470 = torch.conv2d(
            x_469,
            l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_469 = l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_ = None
        x_471 = torch.nn.functional.batch_norm(
            x_470,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_470 = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_ = None
        x_472 = torch.nn.functional.relu(x_471, inplace=True)
        x_471 = None
        x_473 = torch.conv2d(
            x_472,
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_472 = l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = None
        x_474 = torch.nn.functional.batch_norm(
            x_473,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_473 = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = None
        x_se_168 = x_474.mean((2, 3), keepdim=True)
        x_se_169 = torch.conv2d(
            x_se_168,
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_168 = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_170 = torch.nn.functional.relu(x_se_169, inplace=True)
        x_se_169 = None
        x_se_171 = torch.conv2d(
            x_se_170,
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_170 = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_42 = x_se_171.sigmoid()
        x_se_171 = None
        x_475 = x_474 * sigmoid_42
        x_474 = sigmoid_42 = None
        x_475 += x_466
        x_476 = x_475
        x_475 = x_466 = None
        x_477 = torch.nn.functional.relu(x_476, inplace=True)
        x_476 = None
        x_478 = torch.conv2d(
            x_477,
            l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = None
        x_479 = torch.nn.functional.batch_norm(
            x_478,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_478 = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = None
        x_480 = torch.nn.functional.relu(x_479, inplace=True)
        x_479 = None
        x_481 = torch.conv2d(
            x_480,
            l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_480 = l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_ = None
        x_482 = torch.nn.functional.batch_norm(
            x_481,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_481 = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_ = None
        x_483 = torch.nn.functional.relu(x_482, inplace=True)
        x_482 = None
        x_484 = torch.conv2d(
            x_483,
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_483 = l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = None
        x_485 = torch.nn.functional.batch_norm(
            x_484,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_484 = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = None
        x_se_172 = x_485.mean((2, 3), keepdim=True)
        x_se_173 = torch.conv2d(
            x_se_172,
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_172 = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_174 = torch.nn.functional.relu(x_se_173, inplace=True)
        x_se_173 = None
        x_se_175 = torch.conv2d(
            x_se_174,
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_174 = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_43 = x_se_175.sigmoid()
        x_se_175 = None
        x_486 = x_485 * sigmoid_43
        x_485 = sigmoid_43 = None
        x_486 += x_477
        x_487 = x_486
        x_486 = x_477 = None
        x_488 = torch.nn.functional.relu(x_487, inplace=True)
        x_487 = None
        x_489 = torch.conv2d(
            x_488,
            l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = None
        x_490 = torch.nn.functional.batch_norm(
            x_489,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_489 = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = None
        x_491 = torch.nn.functional.relu(x_490, inplace=True)
        x_490 = None
        x_492 = torch.conv2d(
            x_491,
            l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_491 = l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_ = None
        x_493 = torch.nn.functional.batch_norm(
            x_492,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_492 = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_ = None
        x_494 = torch.nn.functional.relu(x_493, inplace=True)
        x_493 = None
        x_495 = torch.conv2d(
            x_494,
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_494 = l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = None
        x_496 = torch.nn.functional.batch_norm(
            x_495,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_495 = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = None
        x_se_176 = x_496.mean((2, 3), keepdim=True)
        x_se_177 = torch.conv2d(
            x_se_176,
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_176 = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_178 = torch.nn.functional.relu(x_se_177, inplace=True)
        x_se_177 = None
        x_se_179 = torch.conv2d(
            x_se_178,
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_178 = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_44 = x_se_179.sigmoid()
        x_se_179 = None
        x_497 = x_496 * sigmoid_44
        x_496 = sigmoid_44 = None
        x_497 += x_488
        x_498 = x_497
        x_497 = x_488 = None
        x_499 = torch.nn.functional.relu(x_498, inplace=True)
        x_498 = None
        x_500 = torch.conv2d(
            x_499,
            l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = None
        x_501 = torch.nn.functional.batch_norm(
            x_500,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_500 = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = None
        x_502 = torch.nn.functional.relu(x_501, inplace=True)
        x_501 = None
        x_503 = torch.conv2d(
            x_502,
            l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_502 = l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_ = None
        x_504 = torch.nn.functional.batch_norm(
            x_503,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_503 = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_ = None
        x_505 = torch.nn.functional.relu(x_504, inplace=True)
        x_504 = None
        x_506 = torch.conv2d(
            x_505,
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_505 = l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = None
        x_507 = torch.nn.functional.batch_norm(
            x_506,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_506 = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = None
        x_se_180 = x_507.mean((2, 3), keepdim=True)
        x_se_181 = torch.conv2d(
            x_se_180,
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_180 = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_182 = torch.nn.functional.relu(x_se_181, inplace=True)
        x_se_181 = None
        x_se_183 = torch.conv2d(
            x_se_182,
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_182 = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_45 = x_se_183.sigmoid()
        x_se_183 = None
        x_508 = x_507 * sigmoid_45
        x_507 = sigmoid_45 = None
        x_508 += x_499
        x_509 = x_508
        x_508 = x_499 = None
        x_510 = torch.nn.functional.relu(x_509, inplace=True)
        x_509 = None
        x_511 = torch.conv2d(
            x_510,
            l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = None
        x_512 = torch.nn.functional.batch_norm(
            x_511,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_511 = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = None
        x_513 = torch.nn.functional.relu(x_512, inplace=True)
        x_512 = None
        x_514 = torch.conv2d(
            x_513,
            l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_513 = l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_ = None
        x_515 = torch.nn.functional.batch_norm(
            x_514,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_514 = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_ = None
        x_516 = torch.nn.functional.relu(x_515, inplace=True)
        x_515 = None
        x_517 = torch.conv2d(
            x_516,
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_516 = l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = None
        x_518 = torch.nn.functional.batch_norm(
            x_517,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_517 = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = None
        x_se_184 = x_518.mean((2, 3), keepdim=True)
        x_se_185 = torch.conv2d(
            x_se_184,
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_184 = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_186 = torch.nn.functional.relu(x_se_185, inplace=True)
        x_se_185 = None
        x_se_187 = torch.conv2d(
            x_se_186,
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_186 = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_46 = x_se_187.sigmoid()
        x_se_187 = None
        x_519 = x_518 * sigmoid_46
        x_518 = sigmoid_46 = None
        x_519 += x_510
        x_520 = x_519
        x_519 = x_510 = None
        x_521 = torch.nn.functional.relu(x_520, inplace=True)
        x_520 = None
        x_522 = torch.conv2d(
            x_521,
            l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = None
        x_523 = torch.nn.functional.batch_norm(
            x_522,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_522 = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = None
        x_524 = torch.nn.functional.relu(x_523, inplace=True)
        x_523 = None
        x_525 = torch.conv2d(
            x_524,
            l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_524 = l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_ = None
        x_526 = torch.nn.functional.batch_norm(
            x_525,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_525 = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_ = None
        x_527 = torch.nn.functional.relu(x_526, inplace=True)
        x_526 = None
        x_528 = torch.conv2d(
            x_527,
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_527 = l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = None
        x_529 = torch.nn.functional.batch_norm(
            x_528,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_528 = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = None
        x_se_188 = x_529.mean((2, 3), keepdim=True)
        x_se_189 = torch.conv2d(
            x_se_188,
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_188 = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_190 = torch.nn.functional.relu(x_se_189, inplace=True)
        x_se_189 = None
        x_se_191 = torch.conv2d(
            x_se_190,
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_190 = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_47 = x_se_191.sigmoid()
        x_se_191 = None
        x_530 = x_529 * sigmoid_47
        x_529 = sigmoid_47 = None
        x_530 += x_521
        x_531 = x_530
        x_530 = x_521 = None
        x_532 = torch.nn.functional.relu(x_531, inplace=True)
        x_531 = None
        x_533 = torch.conv2d(
            x_532,
            l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = None
        x_534 = torch.nn.functional.batch_norm(
            x_533,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_533 = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = None
        x_535 = torch.nn.functional.relu(x_534, inplace=True)
        x_534 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_535 = l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_ = None
        x_537 = torch.nn.functional.batch_norm(
            x_536,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_536 = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_ = None
        x_538 = torch.nn.functional.relu(x_537, inplace=True)
        x_537 = None
        x_539 = torch.conv2d(
            x_538,
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_538 = l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = None
        x_540 = torch.nn.functional.batch_norm(
            x_539,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_539 = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = None
        x_se_192 = x_540.mean((2, 3), keepdim=True)
        x_se_193 = torch.conv2d(
            x_se_192,
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_192 = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_194 = torch.nn.functional.relu(x_se_193, inplace=True)
        x_se_193 = None
        x_se_195 = torch.conv2d(
            x_se_194,
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_194 = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_48 = x_se_195.sigmoid()
        x_se_195 = None
        x_541 = x_540 * sigmoid_48
        x_540 = sigmoid_48 = None
        x_541 += x_532
        x_542 = x_541
        x_541 = x_532 = None
        x_543 = torch.nn.functional.relu(x_542, inplace=True)
        x_542 = None
        x_544 = torch.conv2d(
            x_543,
            l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = None
        x_545 = torch.nn.functional.batch_norm(
            x_544,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_544 = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = None
        x_546 = torch.nn.functional.relu(x_545, inplace=True)
        x_545 = None
        x_547 = torch.conv2d(
            x_546,
            l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_546 = l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_ = None
        x_548 = torch.nn.functional.batch_norm(
            x_547,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_547 = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_ = None
        x_549 = torch.nn.functional.relu(x_548, inplace=True)
        x_548 = None
        x_550 = torch.conv2d(
            x_549,
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_549 = l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = None
        x_551 = torch.nn.functional.batch_norm(
            x_550,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_550 = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = None
        x_se_196 = x_551.mean((2, 3), keepdim=True)
        x_se_197 = torch.conv2d(
            x_se_196,
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_196 = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_198 = torch.nn.functional.relu(x_se_197, inplace=True)
        x_se_197 = None
        x_se_199 = torch.conv2d(
            x_se_198,
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_198 = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_49 = x_se_199.sigmoid()
        x_se_199 = None
        x_552 = x_551 * sigmoid_49
        x_551 = sigmoid_49 = None
        x_552 += x_543
        x_553 = x_552
        x_552 = x_543 = None
        x_554 = torch.nn.functional.relu(x_553, inplace=True)
        x_553 = None
        x_555 = torch.conv2d(
            x_554,
            l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_ = None
        x_556 = torch.nn.functional.batch_norm(
            x_555,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_555 = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_ = None
        x_557 = torch.nn.functional.relu(x_556, inplace=True)
        x_556 = None
        x_558 = torch.conv2d(
            x_557,
            l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_557 = l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_ = None
        x_559 = torch.nn.functional.batch_norm(
            x_558,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_558 = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_ = None
        x_560 = torch.nn.functional.relu(x_559, inplace=True)
        x_559 = None
        x_561 = torch.conv2d(
            x_560,
            l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_560 = l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_ = None
        x_562 = torch.nn.functional.batch_norm(
            x_561,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_561 = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_ = None
        x_se_200 = x_562.mean((2, 3), keepdim=True)
        x_se_201 = torch.conv2d(
            x_se_200,
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_200 = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_202 = torch.nn.functional.relu(x_se_201, inplace=True)
        x_se_201 = None
        x_se_203 = torch.conv2d(
            x_se_202,
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_202 = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_50 = x_se_203.sigmoid()
        x_se_203 = None
        x_563 = x_562 * sigmoid_50
        x_562 = sigmoid_50 = None
        x_563 += x_554
        x_564 = x_563
        x_563 = x_554 = None
        x_565 = torch.nn.functional.relu(x_564, inplace=True)
        x_564 = None
        x_566 = torch.conv2d(
            x_565,
            l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_ = None
        x_567 = torch.nn.functional.batch_norm(
            x_566,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_566 = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_ = None
        x_568 = torch.nn.functional.relu(x_567, inplace=True)
        x_567 = None
        x_569 = torch.conv2d(
            x_568,
            l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_568 = l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_ = None
        x_570 = torch.nn.functional.batch_norm(
            x_569,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_569 = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_ = None
        x_571 = torch.nn.functional.relu(x_570, inplace=True)
        x_570 = None
        x_572 = torch.conv2d(
            x_571,
            l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_571 = l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_ = None
        x_573 = torch.nn.functional.batch_norm(
            x_572,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_572 = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_ = None
        x_se_204 = x_573.mean((2, 3), keepdim=True)
        x_se_205 = torch.conv2d(
            x_se_204,
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_204 = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_206 = torch.nn.functional.relu(x_se_205, inplace=True)
        x_se_205 = None
        x_se_207 = torch.conv2d(
            x_se_206,
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_206 = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_51 = x_se_207.sigmoid()
        x_se_207 = None
        x_574 = x_573 * sigmoid_51
        x_573 = sigmoid_51 = None
        x_574 += x_565
        x_575 = x_574
        x_574 = x_565 = None
        x_576 = torch.nn.functional.relu(x_575, inplace=True)
        x_575 = None
        x_577 = torch.conv2d(
            x_576,
            l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_ = None
        x_578 = torch.nn.functional.batch_norm(
            x_577,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_577 = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_ = None
        x_579 = torch.nn.functional.relu(x_578, inplace=True)
        x_578 = None
        x_580 = torch.conv2d(
            x_579,
            l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_579 = l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_ = None
        x_581 = torch.nn.functional.batch_norm(
            x_580,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_580 = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_ = None
        x_582 = torch.nn.functional.relu(x_581, inplace=True)
        x_581 = None
        x_583 = torch.conv2d(
            x_582,
            l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_582 = l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_ = None
        x_584 = torch.nn.functional.batch_norm(
            x_583,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_583 = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_ = None
        x_se_208 = x_584.mean((2, 3), keepdim=True)
        x_se_209 = torch.conv2d(
            x_se_208,
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_208 = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_210 = torch.nn.functional.relu(x_se_209, inplace=True)
        x_se_209 = None
        x_se_211 = torch.conv2d(
            x_se_210,
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_210 = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_52 = x_se_211.sigmoid()
        x_se_211 = None
        x_585 = x_584 * sigmoid_52
        x_584 = sigmoid_52 = None
        x_585 += x_576
        x_586 = x_585
        x_585 = x_576 = None
        x_587 = torch.nn.functional.relu(x_586, inplace=True)
        x_586 = None
        x_588 = torch.conv2d(
            x_587,
            l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_ = None
        x_589 = torch.nn.functional.batch_norm(
            x_588,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_588 = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_ = None
        x_590 = torch.nn.functional.relu(x_589, inplace=True)
        x_589 = None
        x_591 = torch.conv2d(
            x_590,
            l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_590 = l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_ = None
        x_592 = torch.nn.functional.batch_norm(
            x_591,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_591 = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_ = None
        x_593 = torch.nn.functional.relu(x_592, inplace=True)
        x_592 = None
        x_594 = torch.conv2d(
            x_593,
            l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_593 = l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_ = None
        x_595 = torch.nn.functional.batch_norm(
            x_594,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_594 = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_ = None
        x_se_212 = x_595.mean((2, 3), keepdim=True)
        x_se_213 = torch.conv2d(
            x_se_212,
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_212 = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_214 = torch.nn.functional.relu(x_se_213, inplace=True)
        x_se_213 = None
        x_se_215 = torch.conv2d(
            x_se_214,
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_214 = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_53 = x_se_215.sigmoid()
        x_se_215 = None
        x_596 = x_595 * sigmoid_53
        x_595 = sigmoid_53 = None
        x_596 += x_587
        x_597 = x_596
        x_596 = x_587 = None
        x_598 = torch.nn.functional.relu(x_597, inplace=True)
        x_597 = None
        x_599 = torch.conv2d(
            x_598,
            l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_ = None
        x_600 = torch.nn.functional.batch_norm(
            x_599,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_599 = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_ = None
        x_601 = torch.nn.functional.relu(x_600, inplace=True)
        x_600 = None
        x_602 = torch.conv2d(
            x_601,
            l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_601 = l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_ = None
        x_603 = torch.nn.functional.batch_norm(
            x_602,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_602 = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_ = None
        x_604 = torch.nn.functional.relu(x_603, inplace=True)
        x_603 = None
        x_605 = torch.conv2d(
            x_604,
            l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_604 = l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_ = None
        x_606 = torch.nn.functional.batch_norm(
            x_605,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_605 = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_ = None
        x_se_216 = x_606.mean((2, 3), keepdim=True)
        x_se_217 = torch.conv2d(
            x_se_216,
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_216 = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_218 = torch.nn.functional.relu(x_se_217, inplace=True)
        x_se_217 = None
        x_se_219 = torch.conv2d(
            x_se_218,
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_218 = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_54 = x_se_219.sigmoid()
        x_se_219 = None
        x_607 = x_606 * sigmoid_54
        x_606 = sigmoid_54 = None
        x_607 += x_598
        x_608 = x_607
        x_607 = x_598 = None
        x_609 = torch.nn.functional.relu(x_608, inplace=True)
        x_608 = None
        x_610 = torch.conv2d(
            x_609,
            l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_ = None
        x_611 = torch.nn.functional.batch_norm(
            x_610,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_610 = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_ = None
        x_612 = torch.nn.functional.relu(x_611, inplace=True)
        x_611 = None
        x_613 = torch.conv2d(
            x_612,
            l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_612 = l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_ = None
        x_614 = torch.nn.functional.batch_norm(
            x_613,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_613 = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_ = None
        x_615 = torch.nn.functional.relu(x_614, inplace=True)
        x_614 = None
        x_616 = torch.conv2d(
            x_615,
            l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_615 = l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_ = None
        x_617 = torch.nn.functional.batch_norm(
            x_616,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_616 = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_ = None
        x_se_220 = x_617.mean((2, 3), keepdim=True)
        x_se_221 = torch.conv2d(
            x_se_220,
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_220 = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_222 = torch.nn.functional.relu(x_se_221, inplace=True)
        x_se_221 = None
        x_se_223 = torch.conv2d(
            x_se_222,
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_222 = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_55 = x_se_223.sigmoid()
        x_se_223 = None
        x_618 = x_617 * sigmoid_55
        x_617 = sigmoid_55 = None
        x_618 += x_609
        x_619 = x_618
        x_618 = x_609 = None
        x_620 = torch.nn.functional.relu(x_619, inplace=True)
        x_619 = None
        x_621 = torch.conv2d(
            x_620,
            l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_ = None
        x_622 = torch.nn.functional.batch_norm(
            x_621,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_621 = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_ = None
        x_623 = torch.nn.functional.relu(x_622, inplace=True)
        x_622 = None
        x_624 = torch.conv2d(
            x_623,
            l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_623 = l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_ = None
        x_625 = torch.nn.functional.batch_norm(
            x_624,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_624 = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_ = None
        x_626 = torch.nn.functional.relu(x_625, inplace=True)
        x_625 = None
        x_627 = torch.conv2d(
            x_626,
            l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_626 = l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_ = None
        x_628 = torch.nn.functional.batch_norm(
            x_627,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_627 = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_ = None
        x_se_224 = x_628.mean((2, 3), keepdim=True)
        x_se_225 = torch.conv2d(
            x_se_224,
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_224 = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_226 = torch.nn.functional.relu(x_se_225, inplace=True)
        x_se_225 = None
        x_se_227 = torch.conv2d(
            x_se_226,
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_226 = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_56 = x_se_227.sigmoid()
        x_se_227 = None
        x_629 = x_628 * sigmoid_56
        x_628 = sigmoid_56 = None
        x_629 += x_620
        x_630 = x_629
        x_629 = x_620 = None
        x_631 = torch.nn.functional.relu(x_630, inplace=True)
        x_630 = None
        x_632 = torch.conv2d(
            x_631,
            l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_ = None
        x_633 = torch.nn.functional.batch_norm(
            x_632,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_632 = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_ = None
        x_634 = torch.nn.functional.relu(x_633, inplace=True)
        x_633 = None
        x_635 = torch.conv2d(
            x_634,
            l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_634 = l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_ = None
        x_636 = torch.nn.functional.batch_norm(
            x_635,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_635 = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_ = None
        x_637 = torch.nn.functional.relu(x_636, inplace=True)
        x_636 = None
        x_638 = torch.conv2d(
            x_637,
            l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_637 = l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_ = None
        x_639 = torch.nn.functional.batch_norm(
            x_638,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_638 = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_ = None
        x_se_228 = x_639.mean((2, 3), keepdim=True)
        x_se_229 = torch.conv2d(
            x_se_228,
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_228 = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_230 = torch.nn.functional.relu(x_se_229, inplace=True)
        x_se_229 = None
        x_se_231 = torch.conv2d(
            x_se_230,
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_230 = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_57 = x_se_231.sigmoid()
        x_se_231 = None
        x_640 = x_639 * sigmoid_57
        x_639 = sigmoid_57 = None
        x_640 += x_631
        x_641 = x_640
        x_640 = x_631 = None
        x_642 = torch.nn.functional.relu(x_641, inplace=True)
        x_641 = None
        x_643 = torch.conv2d(
            x_642,
            l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_ = None
        x_644 = torch.nn.functional.batch_norm(
            x_643,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_643 = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_ = None
        x_645 = torch.nn.functional.relu(x_644, inplace=True)
        x_644 = None
        x_646 = torch.conv2d(
            x_645,
            l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_645 = l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_ = None
        x_647 = torch.nn.functional.batch_norm(
            x_646,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_646 = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_ = None
        x_648 = torch.nn.functional.relu(x_647, inplace=True)
        x_647 = None
        x_649 = torch.conv2d(
            x_648,
            l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_648 = l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_ = None
        x_650 = torch.nn.functional.batch_norm(
            x_649,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_649 = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_ = None
        x_se_232 = x_650.mean((2, 3), keepdim=True)
        x_se_233 = torch.conv2d(
            x_se_232,
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_232 = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_234 = torch.nn.functional.relu(x_se_233, inplace=True)
        x_se_233 = None
        x_se_235 = torch.conv2d(
            x_se_234,
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_234 = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_58 = x_se_235.sigmoid()
        x_se_235 = None
        x_651 = x_650 * sigmoid_58
        x_650 = sigmoid_58 = None
        x_651 += x_642
        x_652 = x_651
        x_651 = x_642 = None
        x_653 = torch.nn.functional.relu(x_652, inplace=True)
        x_652 = None
        x_654 = torch.conv2d(
            x_653,
            l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_ = None
        x_655 = torch.nn.functional.batch_norm(
            x_654,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_654 = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_ = None
        x_656 = torch.nn.functional.relu(x_655, inplace=True)
        x_655 = None
        x_657 = torch.conv2d(
            x_656,
            l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_656 = l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_ = None
        x_658 = torch.nn.functional.batch_norm(
            x_657,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_657 = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_ = None
        x_659 = torch.nn.functional.relu(x_658, inplace=True)
        x_658 = None
        x_660 = torch.conv2d(
            x_659,
            l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_659 = l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_ = None
        x_661 = torch.nn.functional.batch_norm(
            x_660,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_660 = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_ = None
        x_se_236 = x_661.mean((2, 3), keepdim=True)
        x_se_237 = torch.conv2d(
            x_se_236,
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_236 = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_238 = torch.nn.functional.relu(x_se_237, inplace=True)
        x_se_237 = None
        x_se_239 = torch.conv2d(
            x_se_238,
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_238 = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_59 = x_se_239.sigmoid()
        x_se_239 = None
        x_662 = x_661 * sigmoid_59
        x_661 = sigmoid_59 = None
        x_662 += x_653
        x_663 = x_662
        x_662 = x_653 = None
        x_664 = torch.nn.functional.relu(x_663, inplace=True)
        x_663 = None
        x_665 = torch.conv2d(
            x_664,
            l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_ = None
        x_666 = torch.nn.functional.batch_norm(
            x_665,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_665 = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_ = None
        x_667 = torch.nn.functional.relu(x_666, inplace=True)
        x_666 = None
        x_668 = torch.conv2d(
            x_667,
            l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_667 = l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_ = None
        x_669 = torch.nn.functional.batch_norm(
            x_668,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_668 = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_ = None
        x_670 = torch.nn.functional.relu(x_669, inplace=True)
        x_669 = None
        x_671 = torch.conv2d(
            x_670,
            l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_670 = l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_ = None
        x_672 = torch.nn.functional.batch_norm(
            x_671,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_671 = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_ = None
        x_se_240 = x_672.mean((2, 3), keepdim=True)
        x_se_241 = torch.conv2d(
            x_se_240,
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_240 = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_242 = torch.nn.functional.relu(x_se_241, inplace=True)
        x_se_241 = None
        x_se_243 = torch.conv2d(
            x_se_242,
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_242 = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_60 = x_se_243.sigmoid()
        x_se_243 = None
        x_673 = x_672 * sigmoid_60
        x_672 = sigmoid_60 = None
        x_673 += x_664
        x_674 = x_673
        x_673 = x_664 = None
        x_675 = torch.nn.functional.relu(x_674, inplace=True)
        x_674 = None
        x_676 = torch.conv2d(
            x_675,
            l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_ = None
        x_677 = torch.nn.functional.batch_norm(
            x_676,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_676 = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_ = None
        x_678 = torch.nn.functional.relu(x_677, inplace=True)
        x_677 = None
        x_679 = torch.conv2d(
            x_678,
            l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_678 = l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_ = None
        x_680 = torch.nn.functional.batch_norm(
            x_679,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_679 = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_ = None
        x_681 = torch.nn.functional.relu(x_680, inplace=True)
        x_680 = None
        x_682 = torch.conv2d(
            x_681,
            l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_681 = l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_ = None
        x_683 = torch.nn.functional.batch_norm(
            x_682,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_682 = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_ = None
        x_se_244 = x_683.mean((2, 3), keepdim=True)
        x_se_245 = torch.conv2d(
            x_se_244,
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_244 = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_246 = torch.nn.functional.relu(x_se_245, inplace=True)
        x_se_245 = None
        x_se_247 = torch.conv2d(
            x_se_246,
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_246 = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_61 = x_se_247.sigmoid()
        x_se_247 = None
        x_684 = x_683 * sigmoid_61
        x_683 = sigmoid_61 = None
        x_684 += x_675
        x_685 = x_684
        x_684 = x_675 = None
        x_686 = torch.nn.functional.relu(x_685, inplace=True)
        x_685 = None
        x_687 = torch.conv2d(
            x_686,
            l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_ = None
        x_688 = torch.nn.functional.batch_norm(
            x_687,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_687 = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_ = None
        x_689 = torch.nn.functional.relu(x_688, inplace=True)
        x_688 = None
        x_690 = torch.conv2d(
            x_689,
            l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_689 = l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_ = None
        x_691 = torch.nn.functional.batch_norm(
            x_690,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_690 = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_ = None
        x_692 = torch.nn.functional.relu(x_691, inplace=True)
        x_691 = None
        x_693 = torch.conv2d(
            x_692,
            l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_692 = l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_ = None
        x_694 = torch.nn.functional.batch_norm(
            x_693,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_693 = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_ = None
        x_se_248 = x_694.mean((2, 3), keepdim=True)
        x_se_249 = torch.conv2d(
            x_se_248,
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_248 = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_250 = torch.nn.functional.relu(x_se_249, inplace=True)
        x_se_249 = None
        x_se_251 = torch.conv2d(
            x_se_250,
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_250 = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_62 = x_se_251.sigmoid()
        x_se_251 = None
        x_695 = x_694 * sigmoid_62
        x_694 = sigmoid_62 = None
        x_695 += x_686
        x_696 = x_695
        x_695 = x_686 = None
        x_697 = torch.nn.functional.relu(x_696, inplace=True)
        x_696 = None
        x_698 = torch.conv2d(
            x_697,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_699 = torch.nn.functional.batch_norm(
            x_698,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_698 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_700 = torch.nn.functional.relu(x_699, inplace=True)
        x_699 = None
        x_701 = torch.conv2d(
            x_700,
            l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_700 = l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_702 = torch.nn.functional.batch_norm(
            x_701,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_701 = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        x_703 = torch.nn.functional.relu(x_702, inplace=True)
        x_702 = None
        x_704 = torch._C._nn.avg_pool2d(x_703, 2, 2, 0, False, True, None)
        x_703 = None
        x_705 = torch.conv2d(
            x_704,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_704 = l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = None
        x_706 = torch.nn.functional.batch_norm(
            x_705,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_705 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        x_se_252 = x_706.mean((2, 3), keepdim=True)
        x_se_253 = torch.conv2d(
            x_se_252,
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_252 = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_254 = torch.nn.functional.relu(x_se_253, inplace=True)
        x_se_253 = None
        x_se_255 = torch.conv2d(
            x_se_254,
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_254 = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_63 = x_se_255.sigmoid()
        x_se_255 = None
        x_707 = x_706 * sigmoid_63
        x_706 = sigmoid_63 = None
        input_16 = torch._C._nn.avg_pool2d(x_697, 2, 2, 0, True, False, None)
        x_697 = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_707 += input_18
        x_708 = x_707
        x_707 = input_18 = None
        x_709 = torch.nn.functional.relu(x_708, inplace=True)
        x_708 = None
        x_710 = torch.conv2d(
            x_709,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_711 = torch.nn.functional.batch_norm(
            x_710,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_710 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_712 = torch.nn.functional.relu(x_711, inplace=True)
        x_711 = None
        x_713 = torch.conv2d(
            x_712,
            l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_712 = l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_714 = torch.nn.functional.batch_norm(
            x_713,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_713 = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_715 = torch.nn.functional.relu(x_714, inplace=True)
        x_714 = None
        x_716 = torch.conv2d(
            x_715,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_715 = l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = None
        x_717 = torch.nn.functional.batch_norm(
            x_716,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_716 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        x_se_256 = x_717.mean((2, 3), keepdim=True)
        x_se_257 = torch.conv2d(
            x_se_256,
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_256 = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_258 = torch.nn.functional.relu(x_se_257, inplace=True)
        x_se_257 = None
        x_se_259 = torch.conv2d(
            x_se_258,
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_258 = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_64 = x_se_259.sigmoid()
        x_se_259 = None
        x_718 = x_717 * sigmoid_64
        x_717 = sigmoid_64 = None
        x_718 += x_709
        x_719 = x_718
        x_718 = x_709 = None
        x_720 = torch.nn.functional.relu(x_719, inplace=True)
        x_719 = None
        x_721 = torch.conv2d(
            x_720,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        x_722 = torch.nn.functional.batch_norm(
            x_721,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_721 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        x_723 = torch.nn.functional.relu(x_722, inplace=True)
        x_722 = None
        x_724 = torch.conv2d(
            x_723,
            l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_723 = l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_ = None
        x_725 = torch.nn.functional.batch_norm(
            x_724,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_724 = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_ = None
        x_726 = torch.nn.functional.relu(x_725, inplace=True)
        x_725 = None
        x_727 = torch.conv2d(
            x_726,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_726 = l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = None
        x_728 = torch.nn.functional.batch_norm(
            x_727,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_727 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        x_se_260 = x_728.mean((2, 3), keepdim=True)
        x_se_261 = torch.conv2d(
            x_se_260,
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_260 = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_262 = torch.nn.functional.relu(x_se_261, inplace=True)
        x_se_261 = None
        x_se_263 = torch.conv2d(
            x_se_262,
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_262 = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_65 = x_se_263.sigmoid()
        x_se_263 = None
        x_729 = x_728 * sigmoid_65
        x_728 = sigmoid_65 = None
        x_729 += x_720
        x_730 = x_729
        x_729 = x_720 = None
        x_731 = torch.nn.functional.relu(x_730, inplace=True)
        x_730 = None
        x_732 = torch.conv2d(
            x_731,
            l_self_modules_layer4_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_3_modules_conv1_parameters_weight_ = None
        x_733 = torch.nn.functional.batch_norm(
            x_732,
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_732 = (
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_3_modules_bn1_parameters_bias_ = None
        x_734 = torch.nn.functional.relu(x_733, inplace=True)
        x_733 = None
        x_735 = torch.conv2d(
            x_734,
            l_self_modules_layer4_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_734 = l_self_modules_layer4_modules_3_modules_conv2_parameters_weight_ = None
        x_736 = torch.nn.functional.batch_norm(
            x_735,
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_735 = (
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_3_modules_bn2_parameters_bias_ = None
        x_737 = torch.nn.functional.relu(x_736, inplace=True)
        x_736 = None
        x_738 = torch.conv2d(
            x_737,
            l_self_modules_layer4_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_737 = l_self_modules_layer4_modules_3_modules_conv3_parameters_weight_ = None
        x_739 = torch.nn.functional.batch_norm(
            x_738,
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_738 = (
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_3_modules_bn3_parameters_bias_ = None
        x_se_264 = x_739.mean((2, 3), keepdim=True)
        x_se_265 = torch.conv2d(
            x_se_264,
            l_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_264 = (
            l_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_3_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_266 = torch.nn.functional.relu(x_se_265, inplace=True)
        x_se_265 = None
        x_se_267 = torch.conv2d(
            x_se_266,
            l_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_266 = (
            l_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_3_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_66 = x_se_267.sigmoid()
        x_se_267 = None
        x_740 = x_739 * sigmoid_66
        x_739 = sigmoid_66 = None
        x_740 += x_731
        x_741 = x_740
        x_740 = x_731 = None
        x_742 = torch.nn.functional.relu(x_741, inplace=True)
        x_741 = None
        x_743 = torch.nn.functional.adaptive_avg_pool2d(x_742, 1)
        x_742 = None
        x_744 = x_743.flatten(1, -1)
        x_743 = None
        x_745 = torch._C._nn.linear(
            x_744,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_744 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_745,)
