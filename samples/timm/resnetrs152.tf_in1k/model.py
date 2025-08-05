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
        L_self_modules_maxpool_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_maxpool_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_maxpool_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_maxpool_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_maxpool_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_maxpool_modules_0_parameters_weight_ = (
            L_self_modules_maxpool_modules_0_parameters_weight_
        )
        l_self_modules_maxpool_modules_1_buffers_running_mean_ = (
            L_self_modules_maxpool_modules_1_buffers_running_mean_
        )
        l_self_modules_maxpool_modules_1_buffers_running_var_ = (
            L_self_modules_maxpool_modules_1_buffers_running_var_
        )
        l_self_modules_maxpool_modules_1_parameters_weight_ = (
            L_self_modules_maxpool_modules_1_parameters_weight_
        )
        l_self_modules_maxpool_modules_1_parameters_bias_ = (
            L_self_modules_maxpool_modules_1_parameters_bias_
        )
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
        input_8 = torch.conv2d(
            x_1,
            l_self_modules_maxpool_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = l_self_modules_maxpool_modules_0_parameters_weight_ = None
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_maxpool_modules_1_buffers_running_mean_,
            l_self_modules_maxpool_modules_1_buffers_running_var_,
            l_self_modules_maxpool_modules_1_parameters_weight_,
            l_self_modules_maxpool_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = (
            l_self_modules_maxpool_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_maxpool_modules_1_buffers_running_var_
        ) = (
            l_self_modules_maxpool_modules_1_parameters_weight_
        ) = l_self_modules_maxpool_modules_1_parameters_bias_ = None
        input_10 = torch.nn.functional.relu(input_9, inplace=True)
        input_9 = None
        x_2 = torch.conv2d(
            input_10,
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        x_3 = torch.nn.functional.batch_norm(
            x_2,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_2 = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = None
        x_6 = torch.nn.functional.batch_norm(
            x_5,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_5 = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = None
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
        x_se = x_9.mean((2, 3), keepdim=True)
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
        x_10 = x_9 * sigmoid
        x_9 = sigmoid = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_10 += input_12
        x_11 = x_10
        x_10 = input_12 = None
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = None
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = None
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = None
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = None
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = None
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = None
        x_se_4 = x_20.mean((2, 3), keepdim=True)
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
        x_21 = x_20 * sigmoid_1
        x_20 = sigmoid_1 = None
        x_21 += x_12
        x_22 = x_21
        x_21 = x_12 = None
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_ = None
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_ = None
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_layer1_modules_2_modules_conv2_parameters_weight_ = None
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = (
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn2_parameters_bias_ = None
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_ = None
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_ = None
        x_se_8 = x_31.mean((2, 3), keepdim=True)
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
        x_32 = x_31 * sigmoid_2
        x_31 = sigmoid_2 = None
        x_32 += x_23
        x_33 = x_32
        x_32 = x_23 = None
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = None
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = None
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        x_se_12 = x_42.mean((2, 3), keepdim=True)
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
        x_43 = x_42 * sigmoid_3
        x_42 = sigmoid_3 = None
        input_13 = torch._C._nn.avg_pool2d(x_34, 2, 2, 0, True, False, None)
        x_34 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_43 += input_15
        x_44 = x_43
        x_43 = input_15 = None
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = None
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = None
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        x_se_16 = x_53.mean((2, 3), keepdim=True)
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
        x_54 = x_53 * sigmoid_4
        x_53 = sigmoid_4 = None
        x_54 += x_45
        x_55 = x_54
        x_54 = x_45 = None
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = None
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = None
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_ = None
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_ = None
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = None
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = None
        x_se_20 = x_64.mean((2, 3), keepdim=True)
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
        x_65 = x_64 * sigmoid_5
        x_64 = sigmoid_5 = None
        x_65 += x_56
        x_66 = x_65
        x_65 = x_56 = None
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = None
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = None
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_ = None
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_ = None
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = None
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = None
        x_se_24 = x_75.mean((2, 3), keepdim=True)
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
        x_76 = x_75 * sigmoid_6
        x_75 = sigmoid_6 = None
        x_76 += x_67
        x_77 = x_76
        x_76 = x_67 = None
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_4_modules_conv1_parameters_weight_ = None
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn1_parameters_bias_ = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_layer2_modules_4_modules_conv2_parameters_weight_ = None
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = (
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn2_parameters_bias_ = None
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_layer2_modules_4_modules_conv3_parameters_weight_ = None
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = (
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_4_modules_bn3_parameters_bias_ = None
        x_se_28 = x_86.mean((2, 3), keepdim=True)
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
        x_87 = x_86 * sigmoid_7
        x_86 = sigmoid_7 = None
        x_87 += x_78
        x_88 = x_87
        x_87 = x_78 = None
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_5_modules_conv1_parameters_weight_ = None
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = (
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn1_parameters_bias_ = None
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_layer2_modules_5_modules_conv2_parameters_weight_ = None
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = (
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn2_parameters_bias_ = None
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_layer2_modules_5_modules_conv3_parameters_weight_ = None
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = (
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_5_modules_bn3_parameters_bias_ = None
        x_se_32 = x_97.mean((2, 3), keepdim=True)
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
        x_98 = x_97 * sigmoid_8
        x_97 = sigmoid_8 = None
        x_98 += x_89
        x_99 = x_98
        x_98 = x_89 = None
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_6_modules_conv1_parameters_weight_ = None
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = (
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn1_parameters_bias_ = None
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_layer2_modules_6_modules_conv2_parameters_weight_ = None
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = (
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn2_parameters_bias_ = None
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_layer2_modules_6_modules_conv3_parameters_weight_ = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = (
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_6_modules_bn3_parameters_bias_ = None
        x_se_36 = x_108.mean((2, 3), keepdim=True)
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
        x_109 = x_108 * sigmoid_9
        x_108 = sigmoid_9 = None
        x_109 += x_100
        x_110 = x_109
        x_109 = x_100 = None
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_7_modules_conv1_parameters_weight_ = None
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = (
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn1_parameters_bias_ = None
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_layer2_modules_7_modules_conv2_parameters_weight_ = None
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = (
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn2_parameters_bias_ = None
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_layer2_modules_7_modules_conv3_parameters_weight_ = None
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = (
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_7_modules_bn3_parameters_bias_ = None
        x_se_40 = x_119.mean((2, 3), keepdim=True)
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
        x_120 = x_119 * sigmoid_10
        x_119 = sigmoid_10 = None
        x_120 += x_111
        x_121 = x_120
        x_120 = x_111 = None
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        x_se_44 = x_130.mean((2, 3), keepdim=True)
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
        x_131 = x_130 * sigmoid_11
        x_130 = sigmoid_11 = None
        input_16 = torch._C._nn.avg_pool2d(x_122, 2, 2, 0, True, False, None)
        x_122 = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_131 += input_18
        x_132 = x_131
        x_131 = input_18 = None
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        x_se_48 = x_141.mean((2, 3), keepdim=True)
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
        x_142 = x_141 * sigmoid_12
        x_141 = sigmoid_12 = None
        x_142 += x_133
        x_143 = x_142
        x_142 = x_133 = None
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_ = None
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_ = None
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        x_se_52 = x_152.mean((2, 3), keepdim=True)
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
        x_153 = x_152 * sigmoid_13
        x_152 = sigmoid_13 = None
        x_153 += x_144
        x_154 = x_153
        x_153 = x_144 = None
        x_155 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_ = None
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_ = None
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        x_se_56 = x_163.mean((2, 3), keepdim=True)
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
        x_164 = x_163 * sigmoid_14
        x_163 = sigmoid_14 = None
        x_164 += x_155
        x_165 = x_164
        x_164 = x_155 = None
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_ = None
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_ = None
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = None
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        x_se_60 = x_174.mean((2, 3), keepdim=True)
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
        x_175 = x_174 * sigmoid_15
        x_174 = sigmoid_15 = None
        x_175 += x_166
        x_176 = x_175
        x_175 = x_166 = None
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_ = None
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_ = None
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = None
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        x_se_64 = x_185.mean((2, 3), keepdim=True)
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
        x_186 = x_185 * sigmoid_16
        x_185 = sigmoid_16 = None
        x_186 += x_177
        x_187 = x_186
        x_186 = x_177 = None
        x_188 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = None
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = None
        x_191 = torch.nn.functional.relu(x_190, inplace=True)
        x_190 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_ = None
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_ = None
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = None
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_195 = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = None
        x_se_68 = x_196.mean((2, 3), keepdim=True)
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
        x_197 = x_196 * sigmoid_17
        x_196 = sigmoid_17 = None
        x_197 += x_188
        x_198 = x_197
        x_197 = x_188 = None
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = None
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = None
        x_202 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_ = None
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_ = None
        x_205 = torch.nn.functional.relu(x_204, inplace=True)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = None
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = None
        x_se_72 = x_207.mean((2, 3), keepdim=True)
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
        x_208 = x_207 * sigmoid_18
        x_207 = sigmoid_18 = None
        x_208 += x_199
        x_209 = x_208
        x_208 = x_199 = None
        x_210 = torch.nn.functional.relu(x_209, inplace=True)
        x_209 = None
        x_211 = torch.conv2d(
            x_210,
            l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = None
        x_212 = torch.nn.functional.batch_norm(
            x_211,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_211 = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = None
        x_213 = torch.nn.functional.relu(x_212, inplace=True)
        x_212 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_ = None
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_ = None
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = None
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = None
        x_se_76 = x_218.mean((2, 3), keepdim=True)
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
        x_219 = x_218 * sigmoid_19
        x_218 = sigmoid_19 = None
        x_219 += x_210
        x_220 = x_219
        x_219 = x_210 = None
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = None
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_222 = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = None
        x_224 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_ = None
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_ = None
        x_227 = torch.nn.functional.relu(x_226, inplace=True)
        x_226 = None
        x_228 = torch.conv2d(
            x_227,
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_227 = l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = None
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_228 = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = None
        x_se_80 = x_229.mean((2, 3), keepdim=True)
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
        x_230 = x_229 * sigmoid_20
        x_229 = sigmoid_20 = None
        x_230 += x_221
        x_231 = x_230
        x_230 = x_221 = None
        x_232 = torch.nn.functional.relu(x_231, inplace=True)
        x_231 = None
        x_233 = torch.conv2d(
            x_232,
            l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = None
        x_234 = torch.nn.functional.batch_norm(
            x_233,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_233 = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = None
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_ = None
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_ = None
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = None
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = None
        x_se_84 = x_240.mean((2, 3), keepdim=True)
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
        x_241 = x_240 * sigmoid_21
        x_240 = sigmoid_21 = None
        x_241 += x_232
        x_242 = x_241
        x_241 = x_232 = None
        x_243 = torch.nn.functional.relu(x_242, inplace=True)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = None
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = None
        x_246 = torch.nn.functional.relu(x_245, inplace=True)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_ = None
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_247 = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_ = None
        x_249 = torch.nn.functional.relu(x_248, inplace=True)
        x_248 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = None
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_250 = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = None
        x_se_88 = x_251.mean((2, 3), keepdim=True)
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
        x_252 = x_251 * sigmoid_22
        x_251 = sigmoid_22 = None
        x_252 += x_243
        x_253 = x_252
        x_252 = x_243 = None
        x_254 = torch.nn.functional.relu(x_253, inplace=True)
        x_253 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = None
        x_256 = torch.nn.functional.batch_norm(
            x_255,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_255 = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = None
        x_257 = torch.nn.functional.relu(x_256, inplace=True)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_ = None
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_ = None
        x_260 = torch.nn.functional.relu(x_259, inplace=True)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = None
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_261 = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = None
        x_se_92 = x_262.mean((2, 3), keepdim=True)
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
        x_263 = x_262 * sigmoid_23
        x_262 = sigmoid_23 = None
        x_263 += x_254
        x_264 = x_263
        x_263 = x_254 = None
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = None
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = None
        x_268 = torch.nn.functional.relu(x_267, inplace=True)
        x_267 = None
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_268 = l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_ = None
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_269 = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_ = None
        x_271 = torch.nn.functional.relu(x_270, inplace=True)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = None
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_272 = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = None
        x_se_96 = x_273.mean((2, 3), keepdim=True)
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
        x_274 = x_273 * sigmoid_24
        x_273 = sigmoid_24 = None
        x_274 += x_265
        x_275 = x_274
        x_274 = x_265 = None
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = None
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = None
        x_279 = torch.nn.functional.relu(x_278, inplace=True)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_ = None
        x_281 = torch.nn.functional.batch_norm(
            x_280,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_280 = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_ = None
        x_282 = torch.nn.functional.relu(x_281, inplace=True)
        x_281 = None
        x_283 = torch.conv2d(
            x_282,
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_282 = l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = None
        x_284 = torch.nn.functional.batch_norm(
            x_283,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_283 = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = None
        x_se_100 = x_284.mean((2, 3), keepdim=True)
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
        x_285 = x_284 * sigmoid_25
        x_284 = sigmoid_25 = None
        x_285 += x_276
        x_286 = x_285
        x_285 = x_276 = None
        x_287 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = None
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_288 = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = None
        x_290 = torch.nn.functional.relu(x_289, inplace=True)
        x_289 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_290 = l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_ = None
        x_292 = torch.nn.functional.batch_norm(
            x_291,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_291 = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_ = None
        x_293 = torch.nn.functional.relu(x_292, inplace=True)
        x_292 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = None
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = None
        x_se_104 = x_295.mean((2, 3), keepdim=True)
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
        x_296 = x_295 * sigmoid_26
        x_295 = sigmoid_26 = None
        x_296 += x_287
        x_297 = x_296
        x_296 = x_287 = None
        x_298 = torch.nn.functional.relu(x_297, inplace=True)
        x_297 = None
        x_299 = torch.conv2d(
            x_298,
            l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = None
        x_300 = torch.nn.functional.batch_norm(
            x_299,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_299 = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = None
        x_301 = torch.nn.functional.relu(x_300, inplace=True)
        x_300 = None
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_ = None
        x_303 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_302 = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_ = None
        x_304 = torch.nn.functional.relu(x_303, inplace=True)
        x_303 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_304 = l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = None
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = None
        x_se_108 = x_306.mean((2, 3), keepdim=True)
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
        x_307 = x_306 * sigmoid_27
        x_306 = sigmoid_27 = None
        x_307 += x_298
        x_308 = x_307
        x_307 = x_298 = None
        x_309 = torch.nn.functional.relu(x_308, inplace=True)
        x_308 = None
        x_310 = torch.conv2d(
            x_309,
            l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = None
        x_311 = torch.nn.functional.batch_norm(
            x_310,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_310 = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = None
        x_312 = torch.nn.functional.relu(x_311, inplace=True)
        x_311 = None
        x_313 = torch.conv2d(
            x_312,
            l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_312 = l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_ = None
        x_314 = torch.nn.functional.batch_norm(
            x_313,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_313 = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_ = None
        x_315 = torch.nn.functional.relu(x_314, inplace=True)
        x_314 = None
        x_316 = torch.conv2d(
            x_315,
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_315 = l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = None
        x_317 = torch.nn.functional.batch_norm(
            x_316,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_316 = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = None
        x_se_112 = x_317.mean((2, 3), keepdim=True)
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
        x_318 = x_317 * sigmoid_28
        x_317 = sigmoid_28 = None
        x_318 += x_309
        x_319 = x_318
        x_318 = x_309 = None
        x_320 = torch.nn.functional.relu(x_319, inplace=True)
        x_319 = None
        x_321 = torch.conv2d(
            x_320,
            l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = None
        x_322 = torch.nn.functional.batch_norm(
            x_321,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_321 = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = None
        x_323 = torch.nn.functional.relu(x_322, inplace=True)
        x_322 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_323 = l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_ = None
        x_325 = torch.nn.functional.batch_norm(
            x_324,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_324 = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_ = None
        x_326 = torch.nn.functional.relu(x_325, inplace=True)
        x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_326 = l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = None
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_327 = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = None
        x_se_116 = x_328.mean((2, 3), keepdim=True)
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
        x_329 = x_328 * sigmoid_29
        x_328 = sigmoid_29 = None
        x_329 += x_320
        x_330 = x_329
        x_329 = x_320 = None
        x_331 = torch.nn.functional.relu(x_330, inplace=True)
        x_330 = None
        x_332 = torch.conv2d(
            x_331,
            l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = None
        x_333 = torch.nn.functional.batch_norm(
            x_332,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_332 = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = None
        x_334 = torch.nn.functional.relu(x_333, inplace=True)
        x_333 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_ = None
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_335 = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_ = None
        x_337 = torch.nn.functional.relu(x_336, inplace=True)
        x_336 = None
        x_338 = torch.conv2d(
            x_337,
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_337 = l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = None
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = None
        x_se_120 = x_339.mean((2, 3), keepdim=True)
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
        x_340 = x_339 * sigmoid_30
        x_339 = sigmoid_30 = None
        x_340 += x_331
        x_341 = x_340
        x_340 = x_331 = None
        x_342 = torch.nn.functional.relu(x_341, inplace=True)
        x_341 = None
        x_343 = torch.conv2d(
            x_342,
            l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = None
        x_344 = torch.nn.functional.batch_norm(
            x_343,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_343 = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = None
        x_345 = torch.nn.functional.relu(x_344, inplace=True)
        x_344 = None
        x_346 = torch.conv2d(
            x_345,
            l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_345 = l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_ = None
        x_347 = torch.nn.functional.batch_norm(
            x_346,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_346 = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_ = None
        x_348 = torch.nn.functional.relu(x_347, inplace=True)
        x_347 = None
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_348 = l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = None
        x_350 = torch.nn.functional.batch_norm(
            x_349,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_349 = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = None
        x_se_124 = x_350.mean((2, 3), keepdim=True)
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
        x_351 = x_350 * sigmoid_31
        x_350 = sigmoid_31 = None
        x_351 += x_342
        x_352 = x_351
        x_351 = x_342 = None
        x_353 = torch.nn.functional.relu(x_352, inplace=True)
        x_352 = None
        x_354 = torch.conv2d(
            x_353,
            l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = None
        x_355 = torch.nn.functional.batch_norm(
            x_354,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_354 = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = None
        x_356 = torch.nn.functional.relu(x_355, inplace=True)
        x_355 = None
        x_357 = torch.conv2d(
            x_356,
            l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_356 = l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_ = None
        x_358 = torch.nn.functional.batch_norm(
            x_357,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_357 = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_ = None
        x_359 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        x_360 = torch.conv2d(
            x_359,
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_359 = l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = None
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_360 = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = None
        x_se_128 = x_361.mean((2, 3), keepdim=True)
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
        x_362 = x_361 * sigmoid_32
        x_361 = sigmoid_32 = None
        x_362 += x_353
        x_363 = x_362
        x_362 = x_353 = None
        x_364 = torch.nn.functional.relu(x_363, inplace=True)
        x_363 = None
        x_365 = torch.conv2d(
            x_364,
            l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = None
        x_366 = torch.nn.functional.batch_norm(
            x_365,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_365 = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = None
        x_367 = torch.nn.functional.relu(x_366, inplace=True)
        x_366 = None
        x_368 = torch.conv2d(
            x_367,
            l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_367 = l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_ = None
        x_369 = torch.nn.functional.batch_norm(
            x_368,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_368 = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_ = None
        x_370 = torch.nn.functional.relu(x_369, inplace=True)
        x_369 = None
        x_371 = torch.conv2d(
            x_370,
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_370 = l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = None
        x_372 = torch.nn.functional.batch_norm(
            x_371,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_371 = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = None
        x_se_132 = x_372.mean((2, 3), keepdim=True)
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
        x_373 = x_372 * sigmoid_33
        x_372 = sigmoid_33 = None
        x_373 += x_364
        x_374 = x_373
        x_373 = x_364 = None
        x_375 = torch.nn.functional.relu(x_374, inplace=True)
        x_374 = None
        x_376 = torch.conv2d(
            x_375,
            l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_ = None
        x_377 = torch.nn.functional.batch_norm(
            x_376,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_376 = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_ = None
        x_378 = torch.nn.functional.relu(x_377, inplace=True)
        x_377 = None
        x_379 = torch.conv2d(
            x_378,
            l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_378 = l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_ = None
        x_380 = torch.nn.functional.batch_norm(
            x_379,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_379 = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_ = None
        x_381 = torch.nn.functional.relu(x_380, inplace=True)
        x_380 = None
        x_382 = torch.conv2d(
            x_381,
            l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_381 = l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_ = None
        x_383 = torch.nn.functional.batch_norm(
            x_382,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_382 = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_ = None
        x_se_136 = x_383.mean((2, 3), keepdim=True)
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
        x_384 = x_383 * sigmoid_34
        x_383 = sigmoid_34 = None
        x_384 += x_375
        x_385 = x_384
        x_384 = x_375 = None
        x_386 = torch.nn.functional.relu(x_385, inplace=True)
        x_385 = None
        x_387 = torch.conv2d(
            x_386,
            l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_ = None
        x_388 = torch.nn.functional.batch_norm(
            x_387,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_387 = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_ = None
        x_389 = torch.nn.functional.relu(x_388, inplace=True)
        x_388 = None
        x_390 = torch.conv2d(
            x_389,
            l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_389 = l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_ = None
        x_391 = torch.nn.functional.batch_norm(
            x_390,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_390 = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_ = None
        x_392 = torch.nn.functional.relu(x_391, inplace=True)
        x_391 = None
        x_393 = torch.conv2d(
            x_392,
            l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_392 = l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_ = None
        x_394 = torch.nn.functional.batch_norm(
            x_393,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_393 = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_ = None
        x_se_140 = x_394.mean((2, 3), keepdim=True)
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
        x_395 = x_394 * sigmoid_35
        x_394 = sigmoid_35 = None
        x_395 += x_386
        x_396 = x_395
        x_395 = x_386 = None
        x_397 = torch.nn.functional.relu(x_396, inplace=True)
        x_396 = None
        x_398 = torch.conv2d(
            x_397,
            l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_ = None
        x_399 = torch.nn.functional.batch_norm(
            x_398,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_398 = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_ = None
        x_400 = torch.nn.functional.relu(x_399, inplace=True)
        x_399 = None
        x_401 = torch.conv2d(
            x_400,
            l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_400 = l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_ = None
        x_402 = torch.nn.functional.batch_norm(
            x_401,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_401 = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_ = None
        x_403 = torch.nn.functional.relu(x_402, inplace=True)
        x_402 = None
        x_404 = torch.conv2d(
            x_403,
            l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_403 = l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_ = None
        x_405 = torch.nn.functional.batch_norm(
            x_404,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_404 = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_ = None
        x_se_144 = x_405.mean((2, 3), keepdim=True)
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
        x_406 = x_405 * sigmoid_36
        x_405 = sigmoid_36 = None
        x_406 += x_397
        x_407 = x_406
        x_406 = x_397 = None
        x_408 = torch.nn.functional.relu(x_407, inplace=True)
        x_407 = None
        x_409 = torch.conv2d(
            x_408,
            l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_ = None
        x_410 = torch.nn.functional.batch_norm(
            x_409,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_409 = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_ = None
        x_411 = torch.nn.functional.relu(x_410, inplace=True)
        x_410 = None
        x_412 = torch.conv2d(
            x_411,
            l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_411 = l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_ = None
        x_413 = torch.nn.functional.batch_norm(
            x_412,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_412 = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_ = None
        x_414 = torch.nn.functional.relu(x_413, inplace=True)
        x_413 = None
        x_415 = torch.conv2d(
            x_414,
            l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_414 = l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_ = None
        x_416 = torch.nn.functional.batch_norm(
            x_415,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_415 = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_ = None
        x_se_148 = x_416.mean((2, 3), keepdim=True)
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
        x_417 = x_416 * sigmoid_37
        x_416 = sigmoid_37 = None
        x_417 += x_408
        x_418 = x_417
        x_417 = x_408 = None
        x_419 = torch.nn.functional.relu(x_418, inplace=True)
        x_418 = None
        x_420 = torch.conv2d(
            x_419,
            l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_ = None
        x_421 = torch.nn.functional.batch_norm(
            x_420,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_420 = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_ = None
        x_422 = torch.nn.functional.relu(x_421, inplace=True)
        x_421 = None
        x_423 = torch.conv2d(
            x_422,
            l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_422 = l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_ = None
        x_424 = torch.nn.functional.batch_norm(
            x_423,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_423 = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_ = None
        x_425 = torch.nn.functional.relu(x_424, inplace=True)
        x_424 = None
        x_426 = torch.conv2d(
            x_425,
            l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_425 = l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_ = None
        x_427 = torch.nn.functional.batch_norm(
            x_426,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_426 = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_ = None
        x_se_152 = x_427.mean((2, 3), keepdim=True)
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
        x_428 = x_427 * sigmoid_38
        x_427 = sigmoid_38 = None
        x_428 += x_419
        x_429 = x_428
        x_428 = x_419 = None
        x_430 = torch.nn.functional.relu(x_429, inplace=True)
        x_429 = None
        x_431 = torch.conv2d(
            x_430,
            l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_ = None
        x_432 = torch.nn.functional.batch_norm(
            x_431,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_431 = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_ = None
        x_433 = torch.nn.functional.relu(x_432, inplace=True)
        x_432 = None
        x_434 = torch.conv2d(
            x_433,
            l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_433 = l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_ = None
        x_435 = torch.nn.functional.batch_norm(
            x_434,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_434 = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_ = None
        x_436 = torch.nn.functional.relu(x_435, inplace=True)
        x_435 = None
        x_437 = torch.conv2d(
            x_436,
            l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_436 = l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_ = None
        x_438 = torch.nn.functional.batch_norm(
            x_437,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_437 = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_ = None
        x_se_156 = x_438.mean((2, 3), keepdim=True)
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
        x_439 = x_438 * sigmoid_39
        x_438 = sigmoid_39 = None
        x_439 += x_430
        x_440 = x_439
        x_439 = x_430 = None
        x_441 = torch.nn.functional.relu(x_440, inplace=True)
        x_440 = None
        x_442 = torch.conv2d(
            x_441,
            l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_ = None
        x_443 = torch.nn.functional.batch_norm(
            x_442,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_442 = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_ = None
        x_444 = torch.nn.functional.relu(x_443, inplace=True)
        x_443 = None
        x_445 = torch.conv2d(
            x_444,
            l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_444 = l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_ = None
        x_446 = torch.nn.functional.batch_norm(
            x_445,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_445 = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_ = None
        x_447 = torch.nn.functional.relu(x_446, inplace=True)
        x_446 = None
        x_448 = torch.conv2d(
            x_447,
            l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_447 = l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_ = None
        x_449 = torch.nn.functional.batch_norm(
            x_448,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_448 = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_ = None
        x_se_160 = x_449.mean((2, 3), keepdim=True)
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
        x_450 = x_449 * sigmoid_40
        x_449 = sigmoid_40 = None
        x_450 += x_441
        x_451 = x_450
        x_450 = x_441 = None
        x_452 = torch.nn.functional.relu(x_451, inplace=True)
        x_451 = None
        x_453 = torch.conv2d(
            x_452,
            l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_ = None
        x_454 = torch.nn.functional.batch_norm(
            x_453,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_453 = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_ = None
        x_455 = torch.nn.functional.relu(x_454, inplace=True)
        x_454 = None
        x_456 = torch.conv2d(
            x_455,
            l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_455 = l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_ = None
        x_457 = torch.nn.functional.batch_norm(
            x_456,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_456 = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_ = None
        x_458 = torch.nn.functional.relu(x_457, inplace=True)
        x_457 = None
        x_459 = torch.conv2d(
            x_458,
            l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_458 = l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_ = None
        x_460 = torch.nn.functional.batch_norm(
            x_459,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_459 = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_ = None
        x_se_164 = x_460.mean((2, 3), keepdim=True)
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
        x_461 = x_460 * sigmoid_41
        x_460 = sigmoid_41 = None
        x_461 += x_452
        x_462 = x_461
        x_461 = x_452 = None
        x_463 = torch.nn.functional.relu(x_462, inplace=True)
        x_462 = None
        x_464 = torch.conv2d(
            x_463,
            l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_ = None
        x_465 = torch.nn.functional.batch_norm(
            x_464,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_464 = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_ = None
        x_466 = torch.nn.functional.relu(x_465, inplace=True)
        x_465 = None
        x_467 = torch.conv2d(
            x_466,
            l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_466 = l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_ = None
        x_468 = torch.nn.functional.batch_norm(
            x_467,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_467 = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_ = None
        x_469 = torch.nn.functional.relu(x_468, inplace=True)
        x_468 = None
        x_470 = torch.conv2d(
            x_469,
            l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_469 = l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_ = None
        x_471 = torch.nn.functional.batch_norm(
            x_470,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_470 = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_ = None
        x_se_168 = x_471.mean((2, 3), keepdim=True)
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
        x_472 = x_471 * sigmoid_42
        x_471 = sigmoid_42 = None
        x_472 += x_463
        x_473 = x_472
        x_472 = x_463 = None
        x_474 = torch.nn.functional.relu(x_473, inplace=True)
        x_473 = None
        x_475 = torch.conv2d(
            x_474,
            l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_ = None
        x_476 = torch.nn.functional.batch_norm(
            x_475,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_475 = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_ = None
        x_477 = torch.nn.functional.relu(x_476, inplace=True)
        x_476 = None
        x_478 = torch.conv2d(
            x_477,
            l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_477 = l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_ = None
        x_479 = torch.nn.functional.batch_norm(
            x_478,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_478 = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_ = None
        x_480 = torch.nn.functional.relu(x_479, inplace=True)
        x_479 = None
        x_481 = torch.conv2d(
            x_480,
            l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_480 = l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_ = None
        x_482 = torch.nn.functional.batch_norm(
            x_481,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_481 = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_ = None
        x_se_172 = x_482.mean((2, 3), keepdim=True)
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
        x_483 = x_482 * sigmoid_43
        x_482 = sigmoid_43 = None
        x_483 += x_474
        x_484 = x_483
        x_483 = x_474 = None
        x_485 = torch.nn.functional.relu(x_484, inplace=True)
        x_484 = None
        x_486 = torch.conv2d(
            x_485,
            l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_ = None
        x_487 = torch.nn.functional.batch_norm(
            x_486,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_486 = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_ = None
        x_488 = torch.nn.functional.relu(x_487, inplace=True)
        x_487 = None
        x_489 = torch.conv2d(
            x_488,
            l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_488 = l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_ = None
        x_490 = torch.nn.functional.batch_norm(
            x_489,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_489 = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_ = None
        x_491 = torch.nn.functional.relu(x_490, inplace=True)
        x_490 = None
        x_492 = torch.conv2d(
            x_491,
            l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_491 = l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_ = None
        x_493 = torch.nn.functional.batch_norm(
            x_492,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_492 = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_ = None
        x_se_176 = x_493.mean((2, 3), keepdim=True)
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
        x_494 = x_493 * sigmoid_44
        x_493 = sigmoid_44 = None
        x_494 += x_485
        x_495 = x_494
        x_494 = x_485 = None
        x_496 = torch.nn.functional.relu(x_495, inplace=True)
        x_495 = None
        x_497 = torch.conv2d(
            x_496,
            l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_ = None
        x_498 = torch.nn.functional.batch_norm(
            x_497,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_497 = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_ = None
        x_499 = torch.nn.functional.relu(x_498, inplace=True)
        x_498 = None
        x_500 = torch.conv2d(
            x_499,
            l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_499 = l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_ = None
        x_501 = torch.nn.functional.batch_norm(
            x_500,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_500 = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_ = None
        x_502 = torch.nn.functional.relu(x_501, inplace=True)
        x_501 = None
        x_503 = torch.conv2d(
            x_502,
            l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_502 = l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_ = None
        x_504 = torch.nn.functional.batch_norm(
            x_503,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_503 = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_ = None
        x_se_180 = x_504.mean((2, 3), keepdim=True)
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
        x_505 = x_504 * sigmoid_45
        x_504 = sigmoid_45 = None
        x_505 += x_496
        x_506 = x_505
        x_505 = x_496 = None
        x_507 = torch.nn.functional.relu(x_506, inplace=True)
        x_506 = None
        x_508 = torch.conv2d(
            x_507,
            l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_ = None
        x_509 = torch.nn.functional.batch_norm(
            x_508,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_508 = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_ = None
        x_510 = torch.nn.functional.relu(x_509, inplace=True)
        x_509 = None
        x_511 = torch.conv2d(
            x_510,
            l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_510 = l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_ = None
        x_512 = torch.nn.functional.batch_norm(
            x_511,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_511 = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_ = None
        x_513 = torch.nn.functional.relu(x_512, inplace=True)
        x_512 = None
        x_514 = torch.conv2d(
            x_513,
            l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_513 = l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_ = None
        x_515 = torch.nn.functional.batch_norm(
            x_514,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_514 = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_ = None
        x_se_184 = x_515.mean((2, 3), keepdim=True)
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
        x_516 = x_515 * sigmoid_46
        x_515 = sigmoid_46 = None
        x_516 += x_507
        x_517 = x_516
        x_516 = x_507 = None
        x_518 = torch.nn.functional.relu(x_517, inplace=True)
        x_517 = None
        x_519 = torch.conv2d(
            x_518,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_520 = torch.nn.functional.batch_norm(
            x_519,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_519 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_521 = torch.nn.functional.relu(x_520, inplace=True)
        x_520 = None
        x_522 = torch.conv2d(
            x_521,
            l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_521 = l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_523 = torch.nn.functional.batch_norm(
            x_522,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_522 = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        x_524 = torch.nn.functional.relu(x_523, inplace=True)
        x_523 = None
        x_525 = torch.conv2d(
            x_524,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_524 = l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = None
        x_526 = torch.nn.functional.batch_norm(
            x_525,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_525 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        x_se_188 = x_526.mean((2, 3), keepdim=True)
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
        x_527 = x_526 * sigmoid_47
        x_526 = sigmoid_47 = None
        input_19 = torch._C._nn.avg_pool2d(x_518, 2, 2, 0, True, False, None)
        x_518 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_527 += input_21
        x_528 = x_527
        x_527 = input_21 = None
        x_529 = torch.nn.functional.relu(x_528, inplace=True)
        x_528 = None
        x_530 = torch.conv2d(
            x_529,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_531 = torch.nn.functional.batch_norm(
            x_530,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_530 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_532 = torch.nn.functional.relu(x_531, inplace=True)
        x_531 = None
        x_533 = torch.conv2d(
            x_532,
            l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_532 = l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_534 = torch.nn.functional.batch_norm(
            x_533,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_533 = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_535 = torch.nn.functional.relu(x_534, inplace=True)
        x_534 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_535 = l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = None
        x_537 = torch.nn.functional.batch_norm(
            x_536,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_536 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        x_se_192 = x_537.mean((2, 3), keepdim=True)
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
        x_538 = x_537 * sigmoid_48
        x_537 = sigmoid_48 = None
        x_538 += x_529
        x_539 = x_538
        x_538 = x_529 = None
        x_540 = torch.nn.functional.relu(x_539, inplace=True)
        x_539 = None
        x_541 = torch.conv2d(
            x_540,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        x_542 = torch.nn.functional.batch_norm(
            x_541,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_541 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        x_543 = torch.nn.functional.relu(x_542, inplace=True)
        x_542 = None
        x_544 = torch.conv2d(
            x_543,
            l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_543 = l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_ = None
        x_545 = torch.nn.functional.batch_norm(
            x_544,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_544 = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_ = None
        x_546 = torch.nn.functional.relu(x_545, inplace=True)
        x_545 = None
        x_547 = torch.conv2d(
            x_546,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_546 = l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = None
        x_548 = torch.nn.functional.batch_norm(
            x_547,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_547 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        x_se_196 = x_548.mean((2, 3), keepdim=True)
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
        x_549 = x_548 * sigmoid_49
        x_548 = sigmoid_49 = None
        x_549 += x_540
        x_550 = x_549
        x_549 = x_540 = None
        x_551 = torch.nn.functional.relu(x_550, inplace=True)
        x_550 = None
        x_552 = torch.nn.functional.adaptive_avg_pool2d(x_551, 1)
        x_551 = None
        x_553 = x_552.flatten(1, -1)
        x_552 = None
        x_554 = torch._C._nn.linear(
            x_553,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_553 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_554,)
