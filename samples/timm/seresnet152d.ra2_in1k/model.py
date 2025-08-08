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
        x_2 = torch.nn.functional.max_pool2d(
            x_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
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
            1,
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
            1,
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
            1,
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
            (2, 2),
            (1, 1),
            (1, 1),
            1,
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
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        x_se_12 = x_43.mean((2, 3), keepdim=True)
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
        x_44 = x_43 * sigmoid_3
        x_43 = sigmoid_3 = None
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
        x_44 += input_12
        x_45 = x_44
        x_44 = input_12 = None
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        x_se_16 = x_54.mean((2, 3), keepdim=True)
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
        x_55 = x_54 * sigmoid_4
        x_54 = sigmoid_4 = None
        x_55 += x_46
        x_56 = x_55
        x_55 = x_46 = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = None
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = None
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_ = None
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_ = None
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = None
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = None
        x_se_20 = x_65.mean((2, 3), keepdim=True)
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
        x_66 = x_65 * sigmoid_5
        x_65 = sigmoid_5 = None
        x_66 += x_57
        x_67 = x_66
        x_66 = x_57 = None
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = None
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_ = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_ = None
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = None
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = None
        x_se_24 = x_76.mean((2, 3), keepdim=True)
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
        x_77 = x_76 * sigmoid_6
        x_76 = sigmoid_6 = None
        x_77 += x_68
        x_78 = x_77
        x_77 = x_68 = None
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_ = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_ = None
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_ = None
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_ = None
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_ = None
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = (
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_ = None
        x_se_28 = x_87.mean((2, 3), keepdim=True)
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
        x_88 = x_87 * sigmoid_7
        x_87 = sigmoid_7 = None
        x_88 += x_79
        x_89 = x_88
        x_88 = x_79 = None
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_ = None
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_ = None
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_ = None
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = (
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_ = None
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_ = None
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = (
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_ = None
        x_se_32 = x_98.mean((2, 3), keepdim=True)
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
        x_99 = x_98 * sigmoid_8
        x_98 = sigmoid_8 = None
        x_99 += x_90
        x_100 = x_99
        x_99 = x_90 = None
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_ = None
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = (
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_ = None
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_ = None
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = (
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_ = None
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_ = None
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = (
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_ = None
        x_se_36 = x_109.mean((2, 3), keepdim=True)
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
        x_110 = x_109 * sigmoid_9
        x_109 = sigmoid_9 = None
        x_110 += x_101
        x_111 = x_110
        x_110 = x_101 = None
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_ = None
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_ = None
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_ = None
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = (
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_ = None
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_ = None
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = (
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_ = None
        x_se_40 = x_120.mean((2, 3), keepdim=True)
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
        x_121 = x_120 * sigmoid_10
        x_120 = sigmoid_10 = None
        x_121 += x_112
        x_122 = x_121
        x_121 = x_112 = None
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        x_se_44 = x_131.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_46 = torch.nn.functional.relu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_132 = x_131 * sigmoid_11
        x_131 = sigmoid_11 = None
        input_13 = torch._C._nn.avg_pool2d(x_123, 2, 2, 0, True, False, None)
        x_123 = None
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
        x_132 += input_15
        x_133 = x_132
        x_132 = input_15 = None
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        x_se_48 = x_142.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_50 = torch.nn.functional.relu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_143 = x_142 * sigmoid_12
        x_142 = sigmoid_12 = None
        x_143 += x_134
        x_144 = x_143
        x_143 = x_134 = None
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_ = None
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_ = None
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        x_se_52 = x_153.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_54 = torch.nn.functional.relu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_154 = x_153 * sigmoid_13
        x_153 = sigmoid_13 = None
        x_154 += x_145
        x_155 = x_154
        x_154 = x_145 = None
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_ = None
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_ = None
        x_162 = torch.nn.functional.relu(x_161, inplace=True)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        x_se_56 = x_164.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_58 = torch.nn.functional.relu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        x_165 = x_164 * sigmoid_14
        x_164 = sigmoid_14 = None
        x_165 += x_156
        x_166 = x_165
        x_165 = x_156 = None
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_ = None
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_ = None
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = None
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        x_se_60 = x_175.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_62 = torch.nn.functional.relu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        x_176 = x_175 * sigmoid_15
        x_175 = sigmoid_15 = None
        x_176 += x_167
        x_177 = x_176
        x_176 = x_167 = None
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_ = None
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_ = None
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = None
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        x_se_64 = x_186.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_66 = torch.nn.functional.relu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        x_187 = x_186 * sigmoid_16
        x_186 = sigmoid_16 = None
        x_187 += x_178
        x_188 = x_187
        x_187 = x_178 = None
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = None
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = None
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_ = None
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_ = None
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = None
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = None
        x_se_68 = x_197.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_70 = torch.nn.functional.relu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        x_198 = x_197 * sigmoid_17
        x_197 = sigmoid_17 = None
        x_198 += x_189
        x_199 = x_198
        x_198 = x_189 = None
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = None
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = None
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_ = None
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_ = None
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = None
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = None
        x_se_72 = x_208.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_74 = torch.nn.functional.relu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        x_209 = x_208 * sigmoid_18
        x_208 = sigmoid_18 = None
        x_209 += x_200
        x_210 = x_209
        x_209 = x_200 = None
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = None
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = None
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_ = None
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_215 = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_ = None
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = None
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = None
        x_se_76 = x_219.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_78 = torch.nn.functional.relu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        x_220 = x_219 * sigmoid_19
        x_219 = sigmoid_19 = None
        x_220 += x_211
        x_221 = x_220
        x_220 = x_211 = None
        x_222 = torch.nn.functional.relu(x_221, inplace=True)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = None
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = None
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_ = None
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_ = None
        x_228 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = None
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = None
        x_se_80 = x_230.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_82 = torch.nn.functional.relu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        x_231 = x_230 * sigmoid_20
        x_230 = sigmoid_20 = None
        x_231 += x_222
        x_232 = x_231
        x_231 = x_222 = None
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = None
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = None
        x_236 = torch.nn.functional.relu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_ = None
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_ = None
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = None
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = None
        x_se_84 = x_241.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_86 = torch.nn.functional.relu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        x_242 = x_241 * sigmoid_21
        x_241 = sigmoid_21 = None
        x_242 += x_233
        x_243 = x_242
        x_242 = x_233 = None
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = None
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = None
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_ = None
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_ = None
        x_250 = torch.nn.functional.relu(x_249, inplace=True)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = None
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = None
        x_se_88 = x_252.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_90 = torch.nn.functional.relu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_22 = x_se_91.sigmoid()
        x_se_91 = None
        x_253 = x_252 * sigmoid_22
        x_252 = sigmoid_22 = None
        x_253 += x_244
        x_254 = x_253
        x_253 = x_244 = None
        x_255 = torch.nn.functional.relu(x_254, inplace=True)
        x_254 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = None
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = None
        x_258 = torch.nn.functional.relu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_ = None
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_ = None
        x_261 = torch.nn.functional.relu(x_260, inplace=True)
        x_260 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_261 = l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = None
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = None
        x_se_92 = x_263.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_94 = torch.nn.functional.relu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_23 = x_se_95.sigmoid()
        x_se_95 = None
        x_264 = x_263 * sigmoid_23
        x_263 = sigmoid_23 = None
        x_264 += x_255
        x_265 = x_264
        x_264 = x_255 = None
        x_266 = torch.nn.functional.relu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = None
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = None
        x_269 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_ = None
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_ = None
        x_272 = torch.nn.functional.relu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = None
        x_274 = torch.nn.functional.batch_norm(
            x_273,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_273 = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = None
        x_se_96 = x_274.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_98 = torch.nn.functional.relu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_24 = x_se_99.sigmoid()
        x_se_99 = None
        x_275 = x_274 * sigmoid_24
        x_274 = sigmoid_24 = None
        x_275 += x_266
        x_276 = x_275
        x_275 = x_266 = None
        x_277 = torch.nn.functional.relu(x_276, inplace=True)
        x_276 = None
        x_278 = torch.conv2d(
            x_277,
            l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = None
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = None
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_ = None
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_ = None
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = None
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = None
        x_se_100 = x_285.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_102 = torch.nn.functional.relu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_25 = x_se_103.sigmoid()
        x_se_103 = None
        x_286 = x_285 * sigmoid_25
        x_285 = sigmoid_25 = None
        x_286 += x_277
        x_287 = x_286
        x_286 = x_277 = None
        x_288 = torch.nn.functional.relu(x_287, inplace=True)
        x_287 = None
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = None
        x_290 = torch.nn.functional.batch_norm(
            x_289,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_289 = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = None
        x_291 = torch.nn.functional.relu(x_290, inplace=True)
        x_290 = None
        x_292 = torch.conv2d(
            x_291,
            l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_291 = l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_ = None
        x_293 = torch.nn.functional.batch_norm(
            x_292,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_292 = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_ = None
        x_294 = torch.nn.functional.relu(x_293, inplace=True)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = None
        x_296 = torch.nn.functional.batch_norm(
            x_295,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_295 = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = None
        x_se_104 = x_296.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_106 = torch.nn.functional.relu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_26 = x_se_107.sigmoid()
        x_se_107 = None
        x_297 = x_296 * sigmoid_26
        x_296 = sigmoid_26 = None
        x_297 += x_288
        x_298 = x_297
        x_297 = x_288 = None
        x_299 = torch.nn.functional.relu(x_298, inplace=True)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = None
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_300 = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = None
        x_302 = torch.nn.functional.relu(x_301, inplace=True)
        x_301 = None
        x_303 = torch.conv2d(
            x_302,
            l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_302 = l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_ = None
        x_304 = torch.nn.functional.batch_norm(
            x_303,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_303 = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_ = None
        x_305 = torch.nn.functional.relu(x_304, inplace=True)
        x_304 = None
        x_306 = torch.conv2d(
            x_305,
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_305 = l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = None
        x_307 = torch.nn.functional.batch_norm(
            x_306,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_306 = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = None
        x_se_108 = x_307.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_110 = torch.nn.functional.relu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_27 = x_se_111.sigmoid()
        x_se_111 = None
        x_308 = x_307 * sigmoid_27
        x_307 = sigmoid_27 = None
        x_308 += x_299
        x_309 = x_308
        x_308 = x_299 = None
        x_310 = torch.nn.functional.relu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = None
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = None
        x_313 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_ = None
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_ = None
        x_316 = torch.nn.functional.relu(x_315, inplace=True)
        x_315 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_316 = l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = None
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_317 = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = None
        x_se_112 = x_318.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_114 = torch.nn.functional.relu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_28 = x_se_115.sigmoid()
        x_se_115 = None
        x_319 = x_318 * sigmoid_28
        x_318 = sigmoid_28 = None
        x_319 += x_310
        x_320 = x_319
        x_319 = x_310 = None
        x_321 = torch.nn.functional.relu(x_320, inplace=True)
        x_320 = None
        x_322 = torch.conv2d(
            x_321,
            l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = None
        x_323 = torch.nn.functional.batch_norm(
            x_322,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_322 = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = None
        x_324 = torch.nn.functional.relu(x_323, inplace=True)
        x_323 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_324 = l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_ = None
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_ = None
        x_327 = torch.nn.functional.relu(x_326, inplace=True)
        x_326 = None
        x_328 = torch.conv2d(
            x_327,
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_327 = l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = None
        x_329 = torch.nn.functional.batch_norm(
            x_328,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_328 = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = None
        x_se_116 = x_329.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_118 = torch.nn.functional.relu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_29 = x_se_119.sigmoid()
        x_se_119 = None
        x_330 = x_329 * sigmoid_29
        x_329 = sigmoid_29 = None
        x_330 += x_321
        x_331 = x_330
        x_330 = x_321 = None
        x_332 = torch.nn.functional.relu(x_331, inplace=True)
        x_331 = None
        x_333 = torch.conv2d(
            x_332,
            l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = None
        x_334 = torch.nn.functional.batch_norm(
            x_333,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_333 = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = None
        x_335 = torch.nn.functional.relu(x_334, inplace=True)
        x_334 = None
        x_336 = torch.conv2d(
            x_335,
            l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_335 = l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_ = None
        x_337 = torch.nn.functional.batch_norm(
            x_336,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_336 = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_ = None
        x_338 = torch.nn.functional.relu(x_337, inplace=True)
        x_337 = None
        x_339 = torch.conv2d(
            x_338,
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_338 = l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = None
        x_340 = torch.nn.functional.batch_norm(
            x_339,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_339 = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = None
        x_se_120 = x_340.mean((2, 3), keepdim=True)
        x_se_121 = torch.conv2d(
            x_se_120,
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_120 = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_122 = torch.nn.functional.relu(x_se_121, inplace=True)
        x_se_121 = None
        x_se_123 = torch.conv2d(
            x_se_122,
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_122 = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_30 = x_se_123.sigmoid()
        x_se_123 = None
        x_341 = x_340 * sigmoid_30
        x_340 = sigmoid_30 = None
        x_341 += x_332
        x_342 = x_341
        x_341 = x_332 = None
        x_343 = torch.nn.functional.relu(x_342, inplace=True)
        x_342 = None
        x_344 = torch.conv2d(
            x_343,
            l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = None
        x_345 = torch.nn.functional.batch_norm(
            x_344,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_344 = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = None
        x_346 = torch.nn.functional.relu(x_345, inplace=True)
        x_345 = None
        x_347 = torch.conv2d(
            x_346,
            l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_346 = l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_ = None
        x_348 = torch.nn.functional.batch_norm(
            x_347,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_347 = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_ = None
        x_349 = torch.nn.functional.relu(x_348, inplace=True)
        x_348 = None
        x_350 = torch.conv2d(
            x_349,
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_349 = l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = None
        x_351 = torch.nn.functional.batch_norm(
            x_350,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_350 = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = None
        x_se_124 = x_351.mean((2, 3), keepdim=True)
        x_se_125 = torch.conv2d(
            x_se_124,
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_124 = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_126 = torch.nn.functional.relu(x_se_125, inplace=True)
        x_se_125 = None
        x_se_127 = torch.conv2d(
            x_se_126,
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_126 = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_31 = x_se_127.sigmoid()
        x_se_127 = None
        x_352 = x_351 * sigmoid_31
        x_351 = sigmoid_31 = None
        x_352 += x_343
        x_353 = x_352
        x_352 = x_343 = None
        x_354 = torch.nn.functional.relu(x_353, inplace=True)
        x_353 = None
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = None
        x_356 = torch.nn.functional.batch_norm(
            x_355,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_355 = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = None
        x_357 = torch.nn.functional.relu(x_356, inplace=True)
        x_356 = None
        x_358 = torch.conv2d(
            x_357,
            l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_357 = l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_ = None
        x_359 = torch.nn.functional.batch_norm(
            x_358,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_358 = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_ = None
        x_360 = torch.nn.functional.relu(x_359, inplace=True)
        x_359 = None
        x_361 = torch.conv2d(
            x_360,
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_360 = l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = None
        x_362 = torch.nn.functional.batch_norm(
            x_361,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_361 = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = None
        x_se_128 = x_362.mean((2, 3), keepdim=True)
        x_se_129 = torch.conv2d(
            x_se_128,
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_128 = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_130 = torch.nn.functional.relu(x_se_129, inplace=True)
        x_se_129 = None
        x_se_131 = torch.conv2d(
            x_se_130,
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_130 = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_32 = x_se_131.sigmoid()
        x_se_131 = None
        x_363 = x_362 * sigmoid_32
        x_362 = sigmoid_32 = None
        x_363 += x_354
        x_364 = x_363
        x_363 = x_354 = None
        x_365 = torch.nn.functional.relu(x_364, inplace=True)
        x_364 = None
        x_366 = torch.conv2d(
            x_365,
            l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = None
        x_367 = torch.nn.functional.batch_norm(
            x_366,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_366 = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = None
        x_368 = torch.nn.functional.relu(x_367, inplace=True)
        x_367 = None
        x_369 = torch.conv2d(
            x_368,
            l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_368 = l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_ = None
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_ = None
        x_371 = torch.nn.functional.relu(x_370, inplace=True)
        x_370 = None
        x_372 = torch.conv2d(
            x_371,
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_371 = l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = None
        x_373 = torch.nn.functional.batch_norm(
            x_372,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_372 = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = None
        x_se_132 = x_373.mean((2, 3), keepdim=True)
        x_se_133 = torch.conv2d(
            x_se_132,
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_132 = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_134 = torch.nn.functional.relu(x_se_133, inplace=True)
        x_se_133 = None
        x_se_135 = torch.conv2d(
            x_se_134,
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_134 = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_33 = x_se_135.sigmoid()
        x_se_135 = None
        x_374 = x_373 * sigmoid_33
        x_373 = sigmoid_33 = None
        x_374 += x_365
        x_375 = x_374
        x_374 = x_365 = None
        x_376 = torch.nn.functional.relu(x_375, inplace=True)
        x_375 = None
        x_377 = torch.conv2d(
            x_376,
            l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_ = None
        x_378 = torch.nn.functional.batch_norm(
            x_377,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_377 = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_ = None
        x_379 = torch.nn.functional.relu(x_378, inplace=True)
        x_378 = None
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_ = None
        x_381 = torch.nn.functional.batch_norm(
            x_380,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_380 = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_ = None
        x_382 = torch.nn.functional.relu(x_381, inplace=True)
        x_381 = None
        x_383 = torch.conv2d(
            x_382,
            l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_382 = l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_ = None
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_383 = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_ = None
        x_se_136 = x_384.mean((2, 3), keepdim=True)
        x_se_137 = torch.conv2d(
            x_se_136,
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_136 = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_138 = torch.nn.functional.relu(x_se_137, inplace=True)
        x_se_137 = None
        x_se_139 = torch.conv2d(
            x_se_138,
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_138 = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_23_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_34 = x_se_139.sigmoid()
        x_se_139 = None
        x_385 = x_384 * sigmoid_34
        x_384 = sigmoid_34 = None
        x_385 += x_376
        x_386 = x_385
        x_385 = x_376 = None
        x_387 = torch.nn.functional.relu(x_386, inplace=True)
        x_386 = None
        x_388 = torch.conv2d(
            x_387,
            l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_ = None
        x_389 = torch.nn.functional.batch_norm(
            x_388,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_388 = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_ = None
        x_390 = torch.nn.functional.relu(x_389, inplace=True)
        x_389 = None
        x_391 = torch.conv2d(
            x_390,
            l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_390 = l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_ = None
        x_392 = torch.nn.functional.batch_norm(
            x_391,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_391 = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_ = None
        x_393 = torch.nn.functional.relu(x_392, inplace=True)
        x_392 = None
        x_394 = torch.conv2d(
            x_393,
            l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_393 = l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_ = None
        x_395 = torch.nn.functional.batch_norm(
            x_394,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_394 = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_ = None
        x_se_140 = x_395.mean((2, 3), keepdim=True)
        x_se_141 = torch.conv2d(
            x_se_140,
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_140 = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_142 = torch.nn.functional.relu(x_se_141, inplace=True)
        x_se_141 = None
        x_se_143 = torch.conv2d(
            x_se_142,
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_142 = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_24_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_35 = x_se_143.sigmoid()
        x_se_143 = None
        x_396 = x_395 * sigmoid_35
        x_395 = sigmoid_35 = None
        x_396 += x_387
        x_397 = x_396
        x_396 = x_387 = None
        x_398 = torch.nn.functional.relu(x_397, inplace=True)
        x_397 = None
        x_399 = torch.conv2d(
            x_398,
            l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_ = None
        x_400 = torch.nn.functional.batch_norm(
            x_399,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_399 = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_ = None
        x_401 = torch.nn.functional.relu(x_400, inplace=True)
        x_400 = None
        x_402 = torch.conv2d(
            x_401,
            l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_401 = l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_ = None
        x_403 = torch.nn.functional.batch_norm(
            x_402,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_402 = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_ = None
        x_404 = torch.nn.functional.relu(x_403, inplace=True)
        x_403 = None
        x_405 = torch.conv2d(
            x_404,
            l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_404 = l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_ = None
        x_406 = torch.nn.functional.batch_norm(
            x_405,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_405 = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_ = None
        x_se_144 = x_406.mean((2, 3), keepdim=True)
        x_se_145 = torch.conv2d(
            x_se_144,
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_144 = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_146 = torch.nn.functional.relu(x_se_145, inplace=True)
        x_se_145 = None
        x_se_147 = torch.conv2d(
            x_se_146,
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_146 = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_25_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_36 = x_se_147.sigmoid()
        x_se_147 = None
        x_407 = x_406 * sigmoid_36
        x_406 = sigmoid_36 = None
        x_407 += x_398
        x_408 = x_407
        x_407 = x_398 = None
        x_409 = torch.nn.functional.relu(x_408, inplace=True)
        x_408 = None
        x_410 = torch.conv2d(
            x_409,
            l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_ = None
        x_411 = torch.nn.functional.batch_norm(
            x_410,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_410 = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_ = None
        x_412 = torch.nn.functional.relu(x_411, inplace=True)
        x_411 = None
        x_413 = torch.conv2d(
            x_412,
            l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_412 = l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_ = None
        x_414 = torch.nn.functional.batch_norm(
            x_413,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_413 = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_ = None
        x_415 = torch.nn.functional.relu(x_414, inplace=True)
        x_414 = None
        x_416 = torch.conv2d(
            x_415,
            l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_415 = l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_ = None
        x_417 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_416 = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_ = None
        x_se_148 = x_417.mean((2, 3), keepdim=True)
        x_se_149 = torch.conv2d(
            x_se_148,
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_148 = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_150 = torch.nn.functional.relu(x_se_149, inplace=True)
        x_se_149 = None
        x_se_151 = torch.conv2d(
            x_se_150,
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_150 = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_26_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_37 = x_se_151.sigmoid()
        x_se_151 = None
        x_418 = x_417 * sigmoid_37
        x_417 = sigmoid_37 = None
        x_418 += x_409
        x_419 = x_418
        x_418 = x_409 = None
        x_420 = torch.nn.functional.relu(x_419, inplace=True)
        x_419 = None
        x_421 = torch.conv2d(
            x_420,
            l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_ = None
        x_422 = torch.nn.functional.batch_norm(
            x_421,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_421 = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_ = None
        x_423 = torch.nn.functional.relu(x_422, inplace=True)
        x_422 = None
        x_424 = torch.conv2d(
            x_423,
            l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_423 = l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_ = None
        x_425 = torch.nn.functional.batch_norm(
            x_424,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_424 = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_ = None
        x_426 = torch.nn.functional.relu(x_425, inplace=True)
        x_425 = None
        x_427 = torch.conv2d(
            x_426,
            l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_426 = l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_ = None
        x_428 = torch.nn.functional.batch_norm(
            x_427,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_427 = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_ = None
        x_se_152 = x_428.mean((2, 3), keepdim=True)
        x_se_153 = torch.conv2d(
            x_se_152,
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_152 = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_154 = torch.nn.functional.relu(x_se_153, inplace=True)
        x_se_153 = None
        x_se_155 = torch.conv2d(
            x_se_154,
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_154 = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_27_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_38 = x_se_155.sigmoid()
        x_se_155 = None
        x_429 = x_428 * sigmoid_38
        x_428 = sigmoid_38 = None
        x_429 += x_420
        x_430 = x_429
        x_429 = x_420 = None
        x_431 = torch.nn.functional.relu(x_430, inplace=True)
        x_430 = None
        x_432 = torch.conv2d(
            x_431,
            l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_ = None
        x_433 = torch.nn.functional.batch_norm(
            x_432,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_432 = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_ = None
        x_434 = torch.nn.functional.relu(x_433, inplace=True)
        x_433 = None
        x_435 = torch.conv2d(
            x_434,
            l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_434 = l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_ = None
        x_436 = torch.nn.functional.batch_norm(
            x_435,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_435 = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_ = None
        x_437 = torch.nn.functional.relu(x_436, inplace=True)
        x_436 = None
        x_438 = torch.conv2d(
            x_437,
            l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_437 = l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_ = None
        x_439 = torch.nn.functional.batch_norm(
            x_438,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_438 = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_ = None
        x_se_156 = x_439.mean((2, 3), keepdim=True)
        x_se_157 = torch.conv2d(
            x_se_156,
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_156 = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_158 = torch.nn.functional.relu(x_se_157, inplace=True)
        x_se_157 = None
        x_se_159 = torch.conv2d(
            x_se_158,
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_158 = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_28_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_39 = x_se_159.sigmoid()
        x_se_159 = None
        x_440 = x_439 * sigmoid_39
        x_439 = sigmoid_39 = None
        x_440 += x_431
        x_441 = x_440
        x_440 = x_431 = None
        x_442 = torch.nn.functional.relu(x_441, inplace=True)
        x_441 = None
        x_443 = torch.conv2d(
            x_442,
            l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_ = None
        x_444 = torch.nn.functional.batch_norm(
            x_443,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_443 = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_ = None
        x_445 = torch.nn.functional.relu(x_444, inplace=True)
        x_444 = None
        x_446 = torch.conv2d(
            x_445,
            l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_445 = l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_ = None
        x_447 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_446 = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_ = None
        x_448 = torch.nn.functional.relu(x_447, inplace=True)
        x_447 = None
        x_449 = torch.conv2d(
            x_448,
            l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_448 = l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_ = None
        x_450 = torch.nn.functional.batch_norm(
            x_449,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_449 = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_ = None
        x_se_160 = x_450.mean((2, 3), keepdim=True)
        x_se_161 = torch.conv2d(
            x_se_160,
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_160 = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_162 = torch.nn.functional.relu(x_se_161, inplace=True)
        x_se_161 = None
        x_se_163 = torch.conv2d(
            x_se_162,
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_162 = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_29_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_40 = x_se_163.sigmoid()
        x_se_163 = None
        x_451 = x_450 * sigmoid_40
        x_450 = sigmoid_40 = None
        x_451 += x_442
        x_452 = x_451
        x_451 = x_442 = None
        x_453 = torch.nn.functional.relu(x_452, inplace=True)
        x_452 = None
        x_454 = torch.conv2d(
            x_453,
            l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_ = None
        x_455 = torch.nn.functional.batch_norm(
            x_454,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_454 = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_ = None
        x_456 = torch.nn.functional.relu(x_455, inplace=True)
        x_455 = None
        x_457 = torch.conv2d(
            x_456,
            l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_456 = l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_ = None
        x_458 = torch.nn.functional.batch_norm(
            x_457,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_457 = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_ = None
        x_459 = torch.nn.functional.relu(x_458, inplace=True)
        x_458 = None
        x_460 = torch.conv2d(
            x_459,
            l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_459 = l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_ = None
        x_461 = torch.nn.functional.batch_norm(
            x_460,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_460 = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_ = None
        x_se_164 = x_461.mean((2, 3), keepdim=True)
        x_se_165 = torch.conv2d(
            x_se_164,
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_164 = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_166 = torch.nn.functional.relu(x_se_165, inplace=True)
        x_se_165 = None
        x_se_167 = torch.conv2d(
            x_se_166,
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_166 = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_30_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_41 = x_se_167.sigmoid()
        x_se_167 = None
        x_462 = x_461 * sigmoid_41
        x_461 = sigmoid_41 = None
        x_462 += x_453
        x_463 = x_462
        x_462 = x_453 = None
        x_464 = torch.nn.functional.relu(x_463, inplace=True)
        x_463 = None
        x_465 = torch.conv2d(
            x_464,
            l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_ = None
        x_466 = torch.nn.functional.batch_norm(
            x_465,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_465 = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_ = None
        x_467 = torch.nn.functional.relu(x_466, inplace=True)
        x_466 = None
        x_468 = torch.conv2d(
            x_467,
            l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_467 = l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_ = None
        x_469 = torch.nn.functional.batch_norm(
            x_468,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_468 = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_ = None
        x_470 = torch.nn.functional.relu(x_469, inplace=True)
        x_469 = None
        x_471 = torch.conv2d(
            x_470,
            l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_470 = l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_ = None
        x_472 = torch.nn.functional.batch_norm(
            x_471,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_471 = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_ = None
        x_se_168 = x_472.mean((2, 3), keepdim=True)
        x_se_169 = torch.conv2d(
            x_se_168,
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_168 = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_170 = torch.nn.functional.relu(x_se_169, inplace=True)
        x_se_169 = None
        x_se_171 = torch.conv2d(
            x_se_170,
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_170 = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_31_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_42 = x_se_171.sigmoid()
        x_se_171 = None
        x_473 = x_472 * sigmoid_42
        x_472 = sigmoid_42 = None
        x_473 += x_464
        x_474 = x_473
        x_473 = x_464 = None
        x_475 = torch.nn.functional.relu(x_474, inplace=True)
        x_474 = None
        x_476 = torch.conv2d(
            x_475,
            l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_ = None
        x_477 = torch.nn.functional.batch_norm(
            x_476,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_476 = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_ = None
        x_478 = torch.nn.functional.relu(x_477, inplace=True)
        x_477 = None
        x_479 = torch.conv2d(
            x_478,
            l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_478 = l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_ = None
        x_480 = torch.nn.functional.batch_norm(
            x_479,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_479 = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_ = None
        x_481 = torch.nn.functional.relu(x_480, inplace=True)
        x_480 = None
        x_482 = torch.conv2d(
            x_481,
            l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_481 = l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_ = None
        x_483 = torch.nn.functional.batch_norm(
            x_482,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_482 = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_ = None
        x_se_172 = x_483.mean((2, 3), keepdim=True)
        x_se_173 = torch.conv2d(
            x_se_172,
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_172 = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_174 = torch.nn.functional.relu(x_se_173, inplace=True)
        x_se_173 = None
        x_se_175 = torch.conv2d(
            x_se_174,
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_174 = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_32_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_43 = x_se_175.sigmoid()
        x_se_175 = None
        x_484 = x_483 * sigmoid_43
        x_483 = sigmoid_43 = None
        x_484 += x_475
        x_485 = x_484
        x_484 = x_475 = None
        x_486 = torch.nn.functional.relu(x_485, inplace=True)
        x_485 = None
        x_487 = torch.conv2d(
            x_486,
            l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_ = None
        x_488 = torch.nn.functional.batch_norm(
            x_487,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_487 = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_ = None
        x_489 = torch.nn.functional.relu(x_488, inplace=True)
        x_488 = None
        x_490 = torch.conv2d(
            x_489,
            l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_489 = l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_ = None
        x_491 = torch.nn.functional.batch_norm(
            x_490,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_490 = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_ = None
        x_492 = torch.nn.functional.relu(x_491, inplace=True)
        x_491 = None
        x_493 = torch.conv2d(
            x_492,
            l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_492 = l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_ = None
        x_494 = torch.nn.functional.batch_norm(
            x_493,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_493 = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_ = None
        x_se_176 = x_494.mean((2, 3), keepdim=True)
        x_se_177 = torch.conv2d(
            x_se_176,
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_176 = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_178 = torch.nn.functional.relu(x_se_177, inplace=True)
        x_se_177 = None
        x_se_179 = torch.conv2d(
            x_se_178,
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_178 = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_33_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_44 = x_se_179.sigmoid()
        x_se_179 = None
        x_495 = x_494 * sigmoid_44
        x_494 = sigmoid_44 = None
        x_495 += x_486
        x_496 = x_495
        x_495 = x_486 = None
        x_497 = torch.nn.functional.relu(x_496, inplace=True)
        x_496 = None
        x_498 = torch.conv2d(
            x_497,
            l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_ = None
        x_499 = torch.nn.functional.batch_norm(
            x_498,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_498 = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_ = None
        x_500 = torch.nn.functional.relu(x_499, inplace=True)
        x_499 = None
        x_501 = torch.conv2d(
            x_500,
            l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_500 = l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_ = None
        x_502 = torch.nn.functional.batch_norm(
            x_501,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_501 = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_ = None
        x_503 = torch.nn.functional.relu(x_502, inplace=True)
        x_502 = None
        x_504 = torch.conv2d(
            x_503,
            l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_503 = l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_ = None
        x_505 = torch.nn.functional.batch_norm(
            x_504,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_504 = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_ = None
        x_se_180 = x_505.mean((2, 3), keepdim=True)
        x_se_181 = torch.conv2d(
            x_se_180,
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_180 = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_182 = torch.nn.functional.relu(x_se_181, inplace=True)
        x_se_181 = None
        x_se_183 = torch.conv2d(
            x_se_182,
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_182 = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_34_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_45 = x_se_183.sigmoid()
        x_se_183 = None
        x_506 = x_505 * sigmoid_45
        x_505 = sigmoid_45 = None
        x_506 += x_497
        x_507 = x_506
        x_506 = x_497 = None
        x_508 = torch.nn.functional.relu(x_507, inplace=True)
        x_507 = None
        x_509 = torch.conv2d(
            x_508,
            l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_ = None
        x_510 = torch.nn.functional.batch_norm(
            x_509,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_509 = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_ = None
        x_511 = torch.nn.functional.relu(x_510, inplace=True)
        x_510 = None
        x_512 = torch.conv2d(
            x_511,
            l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_511 = l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_ = None
        x_513 = torch.nn.functional.batch_norm(
            x_512,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_512 = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_ = None
        x_514 = torch.nn.functional.relu(x_513, inplace=True)
        x_513 = None
        x_515 = torch.conv2d(
            x_514,
            l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_514 = l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_ = None
        x_516 = torch.nn.functional.batch_norm(
            x_515,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_515 = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_ = None
        x_se_184 = x_516.mean((2, 3), keepdim=True)
        x_se_185 = torch.conv2d(
            x_se_184,
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_184 = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_186 = torch.nn.functional.relu(x_se_185, inplace=True)
        x_se_185 = None
        x_se_187 = torch.conv2d(
            x_se_186,
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_186 = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_35_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_46 = x_se_187.sigmoid()
        x_se_187 = None
        x_517 = x_516 * sigmoid_46
        x_516 = sigmoid_46 = None
        x_517 += x_508
        x_518 = x_517
        x_517 = x_508 = None
        x_519 = torch.nn.functional.relu(x_518, inplace=True)
        x_518 = None
        x_520 = torch.conv2d(
            x_519,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_521 = torch.nn.functional.batch_norm(
            x_520,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_520 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_522 = torch.nn.functional.relu(x_521, inplace=True)
        x_521 = None
        x_523 = torch.conv2d(
            x_522,
            l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_522 = l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_524 = torch.nn.functional.batch_norm(
            x_523,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_523 = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        x_525 = torch.nn.functional.relu(x_524, inplace=True)
        x_524 = None
        x_526 = torch.conv2d(
            x_525,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_525 = l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = None
        x_527 = torch.nn.functional.batch_norm(
            x_526,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_526 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        x_se_188 = x_527.mean((2, 3), keepdim=True)
        x_se_189 = torch.conv2d(
            x_se_188,
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_188 = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_190 = torch.nn.functional.relu(x_se_189, inplace=True)
        x_se_189 = None
        x_se_191 = torch.conv2d(
            x_se_190,
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_190 = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_47 = x_se_191.sigmoid()
        x_se_191 = None
        x_528 = x_527 * sigmoid_47
        x_527 = sigmoid_47 = None
        input_16 = torch._C._nn.avg_pool2d(x_519, 2, 2, 0, True, False, None)
        x_519 = None
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
        x_528 += input_18
        x_529 = x_528
        x_528 = input_18 = None
        x_530 = torch.nn.functional.relu(x_529, inplace=True)
        x_529 = None
        x_531 = torch.conv2d(
            x_530,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_532 = torch.nn.functional.batch_norm(
            x_531,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_531 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_533 = torch.nn.functional.relu(x_532, inplace=True)
        x_532 = None
        x_534 = torch.conv2d(
            x_533,
            l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_533 = l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_535 = torch.nn.functional.batch_norm(
            x_534,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_534 = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_536 = torch.nn.functional.relu(x_535, inplace=True)
        x_535 = None
        x_537 = torch.conv2d(
            x_536,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_536 = l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = None
        x_538 = torch.nn.functional.batch_norm(
            x_537,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_537 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        x_se_192 = x_538.mean((2, 3), keepdim=True)
        x_se_193 = torch.conv2d(
            x_se_192,
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_192 = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_194 = torch.nn.functional.relu(x_se_193, inplace=True)
        x_se_193 = None
        x_se_195 = torch.conv2d(
            x_se_194,
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_194 = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_48 = x_se_195.sigmoid()
        x_se_195 = None
        x_539 = x_538 * sigmoid_48
        x_538 = sigmoid_48 = None
        x_539 += x_530
        x_540 = x_539
        x_539 = x_530 = None
        x_541 = torch.nn.functional.relu(x_540, inplace=True)
        x_540 = None
        x_542 = torch.conv2d(
            x_541,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        x_543 = torch.nn.functional.batch_norm(
            x_542,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_542 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        x_544 = torch.nn.functional.relu(x_543, inplace=True)
        x_543 = None
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_ = None
        x_546 = torch.nn.functional.batch_norm(
            x_545,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_545 = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_ = None
        x_547 = torch.nn.functional.relu(x_546, inplace=True)
        x_546 = None
        x_548 = torch.conv2d(
            x_547,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_547 = l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = None
        x_549 = torch.nn.functional.batch_norm(
            x_548,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_548 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        x_se_196 = x_549.mean((2, 3), keepdim=True)
        x_se_197 = torch.conv2d(
            x_se_196,
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_196 = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc1_parameters_bias_
        ) = None
        x_se_198 = torch.nn.functional.relu(x_se_197, inplace=True)
        x_se_197 = None
        x_se_199 = torch.conv2d(
            x_se_198,
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_198 = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_49 = x_se_199.sigmoid()
        x_se_199 = None
        x_550 = x_549 * sigmoid_49
        x_549 = sigmoid_49 = None
        x_550 += x_541
        x_551 = x_550
        x_550 = x_541 = None
        x_552 = torch.nn.functional.relu(x_551, inplace=True)
        x_551 = None
        x_553 = torch.nn.functional.adaptive_avg_pool2d(x_552, 1)
        x_552 = None
        x_554 = x_553.flatten(1, -1)
        x_553 = None
        x_555 = torch._C._nn.linear(
            x_554,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_554 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_555,)
