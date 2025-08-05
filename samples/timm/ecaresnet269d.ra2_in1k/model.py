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
        L_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_2_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_2_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_3_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_4_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_5_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_6_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_7_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_8_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_9_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_10_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_11_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_12_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_13_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_14_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_15_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_16_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_17_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_18_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_19_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_20_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_21_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_22_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_23_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_24_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_24_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_24_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_24_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_24_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_24_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_24_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_25_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_25_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_25_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_25_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_25_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_25_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_25_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_26_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_26_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_26_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_26_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_26_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_26_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_26_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_27_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_27_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_27_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_27_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_27_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_27_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_27_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_28_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_28_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_28_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_28_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_28_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_28_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_28_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_29_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_29_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_29_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_29_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_29_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_29_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_29_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_2_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_3_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_4_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_5_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_6_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_7_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_8_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_9_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_10_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_11_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_12_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_13_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_14_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_15_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_16_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_17_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_18_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_19_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_20_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_21_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_22_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_23_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_24_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_25_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_26_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_27_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_28_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_29_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_30_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_31_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_32_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_33_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_34_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_35_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_36_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_36_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_36_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_36_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_36_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_36_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_36_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_37_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_37_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_37_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_37_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_37_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_37_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_37_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_38_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_38_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_38_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_38_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_38_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_38_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_38_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_39_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_39_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_39_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_39_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_39_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_39_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_39_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_40_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_40_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_40_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_40_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_40_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_40_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_40_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_41_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_41_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_41_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_41_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_41_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_41_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_41_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_42_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_42_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_42_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_42_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_42_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_42_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_42_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_43_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_43_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_43_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_43_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_43_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_43_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_43_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_44_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_44_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_44_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_44_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_44_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_44_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_44_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_45_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_45_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_45_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_45_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_45_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_45_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_45_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_46_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_46_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_46_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_46_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_46_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_46_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_46_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_47_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_47_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_47_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_47_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_47_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_47_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_47_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_2_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_3_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_4_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_5_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_6_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_7_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer1_modules_2_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_2_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_3_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_4_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_4_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_5_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_5_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_6_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_6_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_7_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_7_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_8_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_8_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_9_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_9_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_10_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_10_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_11_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_11_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_12_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_12_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_13_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_13_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_14_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_14_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_15_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_15_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_16_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_16_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_17_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_17_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_18_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_18_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_19_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_19_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_20_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_20_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_21_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_21_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_22_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_22_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer2_modules_23_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_23_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_24_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_24_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_24_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_24_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_24_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_24_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_24_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_24_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_24_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_24_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_24_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_24_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_24_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_24_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_24_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_24_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_24_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_24_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_24_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_24_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_25_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_25_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_25_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_25_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_25_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_25_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_25_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_25_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_25_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_25_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_25_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_25_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_25_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_25_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_25_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_25_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_25_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_25_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_25_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_25_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_26_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_26_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_26_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_26_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_26_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_26_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_26_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_26_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_26_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_26_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_26_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_26_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_26_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_26_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_26_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_26_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_26_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_26_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_26_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_26_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_27_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_27_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_27_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_27_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_27_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_27_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_27_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_27_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_27_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_27_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_27_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_27_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_27_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_27_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_27_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_27_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_27_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_27_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_27_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_27_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_28_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_28_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_28_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_28_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_28_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_28_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_28_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_28_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_28_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_28_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_28_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_28_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_28_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_28_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_28_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_28_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_28_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_28_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_28_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_28_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_29_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_29_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_29_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_29_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_29_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_29_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_29_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_29_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_29_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_29_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_29_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_29_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_29_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_29_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_29_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_29_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_29_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_29_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_29_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_29_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_2_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_3_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_4_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_5_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_6_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_7_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_8_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_9_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_10_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_11_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_12_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_13_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_14_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_15_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_16_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_17_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_18_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_19_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_20_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_21_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_22_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_23_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_23_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_24_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_24_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_25_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_25_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_26_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_26_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_27_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_27_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_28_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_28_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_29_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_29_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_30_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_30_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_31_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_31_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_32_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_32_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_33_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_33_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_34_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_34_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer3_modules_35_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_35_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_36_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_36_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_36_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_36_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_36_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_36_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_36_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_36_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_36_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_36_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_36_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_36_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_36_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_36_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_36_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_36_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_36_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_36_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_36_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_36_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_37_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_37_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_37_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_37_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_37_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_37_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_37_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_37_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_37_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_37_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_37_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_37_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_37_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_37_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_37_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_37_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_37_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_37_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_37_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_37_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_38_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_38_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_38_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_38_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_38_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_38_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_38_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_38_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_38_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_38_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_38_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_38_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_38_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_38_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_38_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_38_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_38_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_38_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_38_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_38_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_39_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_39_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_39_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_39_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_39_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_39_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_39_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_39_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_39_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_39_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_39_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_39_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_39_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_39_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_39_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_39_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_39_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_39_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_39_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_39_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_40_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_40_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_40_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_40_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_40_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_40_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_40_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_40_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_40_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_40_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_40_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_40_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_40_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_40_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_40_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_40_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_40_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_40_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_40_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_40_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_41_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_41_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_41_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_41_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_41_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_41_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_41_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_41_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_41_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_41_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_41_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_41_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_41_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_41_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_41_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_41_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_41_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_41_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_41_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_41_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_42_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_42_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_42_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_42_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_42_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_42_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_42_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_42_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_42_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_42_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_42_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_42_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_42_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_42_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_42_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_42_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_42_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_42_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_42_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_42_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_43_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_43_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_43_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_43_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_43_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_43_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_43_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_43_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_43_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_43_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_43_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_43_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_43_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_43_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_43_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_43_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_43_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_43_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_43_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_43_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_44_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_44_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_44_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_44_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_44_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_44_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_44_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_44_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_44_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_44_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_44_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_44_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_44_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_44_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_44_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_44_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_44_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_44_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_44_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_44_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_45_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_45_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_45_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_45_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_45_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_45_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_45_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_45_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_45_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_45_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_45_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_45_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_45_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_45_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_45_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_45_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_45_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_45_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_45_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_45_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_46_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_46_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_46_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_46_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_46_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_46_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_46_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_46_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_46_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_46_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_46_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_46_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_46_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_46_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_46_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_46_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_46_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_46_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_46_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_46_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_47_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_47_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_47_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_47_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_47_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_47_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_47_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_47_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_47_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_47_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_47_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_47_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_47_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_47_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_47_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_47_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_47_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_47_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_47_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_47_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer4_modules_2_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_se_modules_conv_parameters_weight_
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
        l_self_modules_layer4_modules_3_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_3_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_4_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_4_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_4_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_5_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_5_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_5_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_5_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_6_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_6_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_6_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_6_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_6_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_6_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_6_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_6_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_7_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_7_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_7_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_7_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_7_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_7_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_7_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_7_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_7_modules_se_modules_conv_parameters_weight_
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
        mean = x_10.mean((2, 3))
        sym_sum = torch.sym_sum([-1, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        y = mean.view(1, 1, -1)
        mean = None
        y_1 = torch.conv1d(
            y,
            l_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y = (
            l_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid = y_1.sigmoid()
        y_1 = None
        y_2 = sigmoid.view(1, -1, 1, 1)
        sigmoid = None
        expand_as = y_2.expand_as(x_10)
        y_2 = None
        x_11 = x_10 * expand_as
        x_10 = expand_as = None
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
        mean_1 = x_21.mean((2, 3))
        y_3 = mean_1.view(1, 1, -1)
        mean_1 = None
        y_4 = torch.conv1d(
            y_3,
            l_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_3 = (
            l_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_1 = y_4.sigmoid()
        y_4 = None
        y_5 = sigmoid_1.view(1, -1, 1, 1)
        sigmoid_1 = None
        expand_as_1 = y_5.expand_as(x_21)
        y_5 = None
        x_22 = x_21 * expand_as_1
        x_21 = expand_as_1 = None
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
        mean_2 = x_32.mean((2, 3))
        y_6 = mean_2.view(1, 1, -1)
        mean_2 = None
        y_7 = torch.conv1d(
            y_6,
            l_self_modules_layer1_modules_2_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_6 = (
            l_self_modules_layer1_modules_2_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_2 = y_7.sigmoid()
        y_7 = None
        y_8 = sigmoid_2.view(1, -1, 1, 1)
        sigmoid_2 = None
        expand_as_2 = y_8.expand_as(x_32)
        y_8 = None
        x_33 = x_32 * expand_as_2
        x_32 = expand_as_2 = None
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
        mean_3 = x_43.mean((2, 3))
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        y_9 = mean_3.view(1, 1, -1)
        mean_3 = None
        y_10 = torch.conv1d(
            y_9,
            l_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_9 = (
            l_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_3 = y_10.sigmoid()
        y_10 = None
        y_11 = sigmoid_3.view(1, -1, 1, 1)
        sigmoid_3 = None
        expand_as_3 = y_11.expand_as(x_43)
        y_11 = None
        x_44 = x_43 * expand_as_3
        x_43 = expand_as_3 = None
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
        mean_4 = x_54.mean((2, 3))
        y_12 = mean_4.view(1, 1, -1)
        mean_4 = None
        y_13 = torch.conv1d(
            y_12,
            l_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_12 = (
            l_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_4 = y_13.sigmoid()
        y_13 = None
        y_14 = sigmoid_4.view(1, -1, 1, 1)
        sigmoid_4 = None
        expand_as_4 = y_14.expand_as(x_54)
        y_14 = None
        x_55 = x_54 * expand_as_4
        x_54 = expand_as_4 = None
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
        mean_5 = x_65.mean((2, 3))
        y_15 = mean_5.view(1, 1, -1)
        mean_5 = None
        y_16 = torch.conv1d(
            y_15,
            l_self_modules_layer2_modules_2_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_15 = (
            l_self_modules_layer2_modules_2_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_5 = y_16.sigmoid()
        y_16 = None
        y_17 = sigmoid_5.view(1, -1, 1, 1)
        sigmoid_5 = None
        expand_as_5 = y_17.expand_as(x_65)
        y_17 = None
        x_66 = x_65 * expand_as_5
        x_65 = expand_as_5 = None
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
        mean_6 = x_76.mean((2, 3))
        y_18 = mean_6.view(1, 1, -1)
        mean_6 = None
        y_19 = torch.conv1d(
            y_18,
            l_self_modules_layer2_modules_3_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_18 = (
            l_self_modules_layer2_modules_3_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_6 = y_19.sigmoid()
        y_19 = None
        y_20 = sigmoid_6.view(1, -1, 1, 1)
        sigmoid_6 = None
        expand_as_6 = y_20.expand_as(x_76)
        y_20 = None
        x_77 = x_76 * expand_as_6
        x_76 = expand_as_6 = None
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
        mean_7 = x_87.mean((2, 3))
        y_21 = mean_7.view(1, 1, -1)
        mean_7 = None
        y_22 = torch.conv1d(
            y_21,
            l_self_modules_layer2_modules_4_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_21 = (
            l_self_modules_layer2_modules_4_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_7 = y_22.sigmoid()
        y_22 = None
        y_23 = sigmoid_7.view(1, -1, 1, 1)
        sigmoid_7 = None
        expand_as_7 = y_23.expand_as(x_87)
        y_23 = None
        x_88 = x_87 * expand_as_7
        x_87 = expand_as_7 = None
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
        mean_8 = x_98.mean((2, 3))
        y_24 = mean_8.view(1, 1, -1)
        mean_8 = None
        y_25 = torch.conv1d(
            y_24,
            l_self_modules_layer2_modules_5_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_24 = (
            l_self_modules_layer2_modules_5_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_8 = y_25.sigmoid()
        y_25 = None
        y_26 = sigmoid_8.view(1, -1, 1, 1)
        sigmoid_8 = None
        expand_as_8 = y_26.expand_as(x_98)
        y_26 = None
        x_99 = x_98 * expand_as_8
        x_98 = expand_as_8 = None
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
        mean_9 = x_109.mean((2, 3))
        y_27 = mean_9.view(1, 1, -1)
        mean_9 = None
        y_28 = torch.conv1d(
            y_27,
            l_self_modules_layer2_modules_6_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_27 = (
            l_self_modules_layer2_modules_6_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_9 = y_28.sigmoid()
        y_28 = None
        y_29 = sigmoid_9.view(1, -1, 1, 1)
        sigmoid_9 = None
        expand_as_9 = y_29.expand_as(x_109)
        y_29 = None
        x_110 = x_109 * expand_as_9
        x_109 = expand_as_9 = None
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
        mean_10 = x_120.mean((2, 3))
        y_30 = mean_10.view(1, 1, -1)
        mean_10 = None
        y_31 = torch.conv1d(
            y_30,
            l_self_modules_layer2_modules_7_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_30 = (
            l_self_modules_layer2_modules_7_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_10 = y_31.sigmoid()
        y_31 = None
        y_32 = sigmoid_10.view(1, -1, 1, 1)
        sigmoid_10 = None
        expand_as_10 = y_32.expand_as(x_120)
        y_32 = None
        x_121 = x_120 * expand_as_10
        x_120 = expand_as_10 = None
        x_121 += x_112
        x_122 = x_121
        x_121 = x_112 = None
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_layer2_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_8_modules_conv1_parameters_weight_ = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_8_modules_bn1_parameters_bias_ = None
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_layer2_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_layer2_modules_8_modules_conv2_parameters_weight_ = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_8_modules_bn2_parameters_bias_ = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_layer2_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_layer2_modules_8_modules_conv3_parameters_weight_ = None
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_8_modules_bn3_parameters_bias_ = None
        mean_11 = x_131.mean((2, 3))
        y_33 = mean_11.view(1, 1, -1)
        mean_11 = None
        y_34 = torch.conv1d(
            y_33,
            l_self_modules_layer2_modules_8_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_33 = (
            l_self_modules_layer2_modules_8_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_11 = y_34.sigmoid()
        y_34 = None
        y_35 = sigmoid_11.view(1, -1, 1, 1)
        sigmoid_11 = None
        expand_as_11 = y_35.expand_as(x_131)
        y_35 = None
        x_132 = x_131 * expand_as_11
        x_131 = expand_as_11 = None
        x_132 += x_123
        x_133 = x_132
        x_132 = x_123 = None
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_layer2_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_9_modules_conv1_parameters_weight_ = None
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_9_modules_bn1_parameters_bias_ = None
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_layer2_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_layer2_modules_9_modules_conv2_parameters_weight_ = None
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = (
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_9_modules_bn2_parameters_bias_ = None
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_layer2_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_layer2_modules_9_modules_conv3_parameters_weight_ = None
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_9_modules_bn3_parameters_bias_ = None
        mean_12 = x_142.mean((2, 3))
        y_36 = mean_12.view(1, 1, -1)
        mean_12 = None
        y_37 = torch.conv1d(
            y_36,
            l_self_modules_layer2_modules_9_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_36 = (
            l_self_modules_layer2_modules_9_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_12 = y_37.sigmoid()
        y_37 = None
        y_38 = sigmoid_12.view(1, -1, 1, 1)
        sigmoid_12 = None
        expand_as_12 = y_38.expand_as(x_142)
        y_38 = None
        x_143 = x_142 * expand_as_12
        x_142 = expand_as_12 = None
        x_143 += x_134
        x_144 = x_143
        x_143 = x_134 = None
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_layer2_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_10_modules_conv1_parameters_weight_ = None
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = (
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_10_modules_bn1_parameters_bias_ = None
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_layer2_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_layer2_modules_10_modules_conv2_parameters_weight_ = None
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = (
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_10_modules_bn2_parameters_bias_ = None
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_layer2_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_layer2_modules_10_modules_conv3_parameters_weight_ = None
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_10_modules_bn3_parameters_bias_ = None
        mean_13 = x_153.mean((2, 3))
        y_39 = mean_13.view(1, 1, -1)
        mean_13 = None
        y_40 = torch.conv1d(
            y_39,
            l_self_modules_layer2_modules_10_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_39 = (
            l_self_modules_layer2_modules_10_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_13 = y_40.sigmoid()
        y_40 = None
        y_41 = sigmoid_13.view(1, -1, 1, 1)
        sigmoid_13 = None
        expand_as_13 = y_41.expand_as(x_153)
        y_41 = None
        x_154 = x_153 * expand_as_13
        x_153 = expand_as_13 = None
        x_154 += x_145
        x_155 = x_154
        x_154 = x_145 = None
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_layer2_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_11_modules_conv1_parameters_weight_ = None
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = (
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_11_modules_bn1_parameters_bias_ = None
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_layer2_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_layer2_modules_11_modules_conv2_parameters_weight_ = None
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = (
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_11_modules_bn2_parameters_bias_ = None
        x_162 = torch.nn.functional.relu(x_161, inplace=True)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_layer2_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_layer2_modules_11_modules_conv3_parameters_weight_ = None
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = (
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_11_modules_bn3_parameters_bias_ = None
        mean_14 = x_164.mean((2, 3))
        y_42 = mean_14.view(1, 1, -1)
        mean_14 = None
        y_43 = torch.conv1d(
            y_42,
            l_self_modules_layer2_modules_11_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_42 = (
            l_self_modules_layer2_modules_11_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_14 = y_43.sigmoid()
        y_43 = None
        y_44 = sigmoid_14.view(1, -1, 1, 1)
        sigmoid_14 = None
        expand_as_14 = y_44.expand_as(x_164)
        y_44 = None
        x_165 = x_164 * expand_as_14
        x_164 = expand_as_14 = None
        x_165 += x_156
        x_166 = x_165
        x_165 = x_156 = None
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_layer2_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_12_modules_conv1_parameters_weight_ = None
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = (
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_12_modules_bn1_parameters_bias_ = None
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_layer2_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_layer2_modules_12_modules_conv2_parameters_weight_ = None
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = (
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_12_modules_bn2_parameters_bias_ = None
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_layer2_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_layer2_modules_12_modules_conv3_parameters_weight_ = None
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = (
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_12_modules_bn3_parameters_bias_ = None
        mean_15 = x_175.mean((2, 3))
        y_45 = mean_15.view(1, 1, -1)
        mean_15 = None
        y_46 = torch.conv1d(
            y_45,
            l_self_modules_layer2_modules_12_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_45 = (
            l_self_modules_layer2_modules_12_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_15 = y_46.sigmoid()
        y_46 = None
        y_47 = sigmoid_15.view(1, -1, 1, 1)
        sigmoid_15 = None
        expand_as_15 = y_47.expand_as(x_175)
        y_47 = None
        x_176 = x_175 * expand_as_15
        x_175 = expand_as_15 = None
        x_176 += x_167
        x_177 = x_176
        x_176 = x_167 = None
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_layer2_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_13_modules_conv1_parameters_weight_ = None
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = (
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_13_modules_bn1_parameters_bias_ = None
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_layer2_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_layer2_modules_13_modules_conv2_parameters_weight_ = None
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = (
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_13_modules_bn2_parameters_bias_ = None
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_layer2_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_layer2_modules_13_modules_conv3_parameters_weight_ = None
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = (
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_13_modules_bn3_parameters_bias_ = None
        mean_16 = x_186.mean((2, 3))
        y_48 = mean_16.view(1, 1, -1)
        mean_16 = None
        y_49 = torch.conv1d(
            y_48,
            l_self_modules_layer2_modules_13_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_48 = (
            l_self_modules_layer2_modules_13_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_16 = y_49.sigmoid()
        y_49 = None
        y_50 = sigmoid_16.view(1, -1, 1, 1)
        sigmoid_16 = None
        expand_as_16 = y_50.expand_as(x_186)
        y_50 = None
        x_187 = x_186 * expand_as_16
        x_186 = expand_as_16 = None
        x_187 += x_178
        x_188 = x_187
        x_187 = x_178 = None
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_layer2_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_14_modules_conv1_parameters_weight_ = None
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = (
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_14_modules_bn1_parameters_bias_ = None
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_layer2_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_layer2_modules_14_modules_conv2_parameters_weight_ = None
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = (
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_14_modules_bn2_parameters_bias_ = None
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_layer2_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_layer2_modules_14_modules_conv3_parameters_weight_ = None
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = (
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_14_modules_bn3_parameters_bias_ = None
        mean_17 = x_197.mean((2, 3))
        y_51 = mean_17.view(1, 1, -1)
        mean_17 = None
        y_52 = torch.conv1d(
            y_51,
            l_self_modules_layer2_modules_14_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_51 = (
            l_self_modules_layer2_modules_14_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_17 = y_52.sigmoid()
        y_52 = None
        y_53 = sigmoid_17.view(1, -1, 1, 1)
        sigmoid_17 = None
        expand_as_17 = y_53.expand_as(x_197)
        y_53 = None
        x_198 = x_197 * expand_as_17
        x_197 = expand_as_17 = None
        x_198 += x_189
        x_199 = x_198
        x_198 = x_189 = None
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_layer2_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_15_modules_conv1_parameters_weight_ = None
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = (
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_15_modules_bn1_parameters_bias_ = None
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_layer2_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_layer2_modules_15_modules_conv2_parameters_weight_ = None
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = (
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_15_modules_bn2_parameters_bias_ = None
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_layer2_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_layer2_modules_15_modules_conv3_parameters_weight_ = None
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = (
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_15_modules_bn3_parameters_bias_ = None
        mean_18 = x_208.mean((2, 3))
        y_54 = mean_18.view(1, 1, -1)
        mean_18 = None
        y_55 = torch.conv1d(
            y_54,
            l_self_modules_layer2_modules_15_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_54 = (
            l_self_modules_layer2_modules_15_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_18 = y_55.sigmoid()
        y_55 = None
        y_56 = sigmoid_18.view(1, -1, 1, 1)
        sigmoid_18 = None
        expand_as_18 = y_56.expand_as(x_208)
        y_56 = None
        x_209 = x_208 * expand_as_18
        x_208 = expand_as_18 = None
        x_209 += x_200
        x_210 = x_209
        x_209 = x_200 = None
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_layer2_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_16_modules_conv1_parameters_weight_ = None
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_16_modules_bn1_parameters_bias_ = None
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_layer2_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_layer2_modules_16_modules_conv2_parameters_weight_ = None
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_215 = (
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_16_modules_bn2_parameters_bias_ = None
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_layer2_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_layer2_modules_16_modules_conv3_parameters_weight_ = None
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = (
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_16_modules_bn3_parameters_bias_ = None
        mean_19 = x_219.mean((2, 3))
        y_57 = mean_19.view(1, 1, -1)
        mean_19 = None
        y_58 = torch.conv1d(
            y_57,
            l_self_modules_layer2_modules_16_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_57 = (
            l_self_modules_layer2_modules_16_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_19 = y_58.sigmoid()
        y_58 = None
        y_59 = sigmoid_19.view(1, -1, 1, 1)
        sigmoid_19 = None
        expand_as_19 = y_59.expand_as(x_219)
        y_59 = None
        x_220 = x_219 * expand_as_19
        x_219 = expand_as_19 = None
        x_220 += x_211
        x_221 = x_220
        x_220 = x_211 = None
        x_222 = torch.nn.functional.relu(x_221, inplace=True)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_layer2_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_17_modules_conv1_parameters_weight_ = None
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = (
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_17_modules_bn1_parameters_bias_ = None
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_layer2_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_layer2_modules_17_modules_conv2_parameters_weight_ = None
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = (
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_17_modules_bn2_parameters_bias_ = None
        x_228 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_layer2_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_layer2_modules_17_modules_conv3_parameters_weight_ = None
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = (
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_17_modules_bn3_parameters_bias_ = None
        mean_20 = x_230.mean((2, 3))
        y_60 = mean_20.view(1, 1, -1)
        mean_20 = None
        y_61 = torch.conv1d(
            y_60,
            l_self_modules_layer2_modules_17_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_60 = (
            l_self_modules_layer2_modules_17_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_20 = y_61.sigmoid()
        y_61 = None
        y_62 = sigmoid_20.view(1, -1, 1, 1)
        sigmoid_20 = None
        expand_as_20 = y_62.expand_as(x_230)
        y_62 = None
        x_231 = x_230 * expand_as_20
        x_230 = expand_as_20 = None
        x_231 += x_222
        x_232 = x_231
        x_231 = x_222 = None
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_layer2_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_18_modules_conv1_parameters_weight_ = None
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = (
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_18_modules_bn1_parameters_bias_ = None
        x_236 = torch.nn.functional.relu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_layer2_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_layer2_modules_18_modules_conv2_parameters_weight_ = None
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = (
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_18_modules_bn2_parameters_bias_ = None
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_layer2_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_layer2_modules_18_modules_conv3_parameters_weight_ = None
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = (
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_18_modules_bn3_parameters_bias_ = None
        mean_21 = x_241.mean((2, 3))
        y_63 = mean_21.view(1, 1, -1)
        mean_21 = None
        y_64 = torch.conv1d(
            y_63,
            l_self_modules_layer2_modules_18_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_63 = (
            l_self_modules_layer2_modules_18_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_21 = y_64.sigmoid()
        y_64 = None
        y_65 = sigmoid_21.view(1, -1, 1, 1)
        sigmoid_21 = None
        expand_as_21 = y_65.expand_as(x_241)
        y_65 = None
        x_242 = x_241 * expand_as_21
        x_241 = expand_as_21 = None
        x_242 += x_233
        x_243 = x_242
        x_242 = x_233 = None
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_layer2_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_19_modules_conv1_parameters_weight_ = None
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = (
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_19_modules_bn1_parameters_bias_ = None
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_layer2_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_layer2_modules_19_modules_conv2_parameters_weight_ = None
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = (
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_19_modules_bn2_parameters_bias_ = None
        x_250 = torch.nn.functional.relu(x_249, inplace=True)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_layer2_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_layer2_modules_19_modules_conv3_parameters_weight_ = None
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = (
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_19_modules_bn3_parameters_bias_ = None
        mean_22 = x_252.mean((2, 3))
        y_66 = mean_22.view(1, 1, -1)
        mean_22 = None
        y_67 = torch.conv1d(
            y_66,
            l_self_modules_layer2_modules_19_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_66 = (
            l_self_modules_layer2_modules_19_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_22 = y_67.sigmoid()
        y_67 = None
        y_68 = sigmoid_22.view(1, -1, 1, 1)
        sigmoid_22 = None
        expand_as_22 = y_68.expand_as(x_252)
        y_68 = None
        x_253 = x_252 * expand_as_22
        x_252 = expand_as_22 = None
        x_253 += x_244
        x_254 = x_253
        x_253 = x_244 = None
        x_255 = torch.nn.functional.relu(x_254, inplace=True)
        x_254 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_layer2_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_20_modules_conv1_parameters_weight_ = None
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = (
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_20_modules_bn1_parameters_bias_ = None
        x_258 = torch.nn.functional.relu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_layer2_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_layer2_modules_20_modules_conv2_parameters_weight_ = None
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = (
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_20_modules_bn2_parameters_bias_ = None
        x_261 = torch.nn.functional.relu(x_260, inplace=True)
        x_260 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_layer2_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_261 = l_self_modules_layer2_modules_20_modules_conv3_parameters_weight_ = None
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = (
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_20_modules_bn3_parameters_bias_ = None
        mean_23 = x_263.mean((2, 3))
        y_69 = mean_23.view(1, 1, -1)
        mean_23 = None
        y_70 = torch.conv1d(
            y_69,
            l_self_modules_layer2_modules_20_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_69 = (
            l_self_modules_layer2_modules_20_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_23 = y_70.sigmoid()
        y_70 = None
        y_71 = sigmoid_23.view(1, -1, 1, 1)
        sigmoid_23 = None
        expand_as_23 = y_71.expand_as(x_263)
        y_71 = None
        x_264 = x_263 * expand_as_23
        x_263 = expand_as_23 = None
        x_264 += x_255
        x_265 = x_264
        x_264 = x_255 = None
        x_266 = torch.nn.functional.relu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_layer2_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_21_modules_conv1_parameters_weight_ = None
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = (
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_21_modules_bn1_parameters_bias_ = None
        x_269 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_layer2_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_layer2_modules_21_modules_conv2_parameters_weight_ = None
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = (
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_21_modules_bn2_parameters_bias_ = None
        x_272 = torch.nn.functional.relu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_layer2_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_layer2_modules_21_modules_conv3_parameters_weight_ = None
        x_274 = torch.nn.functional.batch_norm(
            x_273,
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_273 = (
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_21_modules_bn3_parameters_bias_ = None
        mean_24 = x_274.mean((2, 3))
        y_72 = mean_24.view(1, 1, -1)
        mean_24 = None
        y_73 = torch.conv1d(
            y_72,
            l_self_modules_layer2_modules_21_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_72 = (
            l_self_modules_layer2_modules_21_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_24 = y_73.sigmoid()
        y_73 = None
        y_74 = sigmoid_24.view(1, -1, 1, 1)
        sigmoid_24 = None
        expand_as_24 = y_74.expand_as(x_274)
        y_74 = None
        x_275 = x_274 * expand_as_24
        x_274 = expand_as_24 = None
        x_275 += x_266
        x_276 = x_275
        x_275 = x_266 = None
        x_277 = torch.nn.functional.relu(x_276, inplace=True)
        x_276 = None
        x_278 = torch.conv2d(
            x_277,
            l_self_modules_layer2_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_22_modules_conv1_parameters_weight_ = None
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = (
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_22_modules_bn1_parameters_bias_ = None
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_layer2_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_layer2_modules_22_modules_conv2_parameters_weight_ = None
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = (
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_22_modules_bn2_parameters_bias_ = None
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_layer2_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_layer2_modules_22_modules_conv3_parameters_weight_ = None
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = (
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_22_modules_bn3_parameters_bias_ = None
        mean_25 = x_285.mean((2, 3))
        y_75 = mean_25.view(1, 1, -1)
        mean_25 = None
        y_76 = torch.conv1d(
            y_75,
            l_self_modules_layer2_modules_22_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_75 = (
            l_self_modules_layer2_modules_22_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_25 = y_76.sigmoid()
        y_76 = None
        y_77 = sigmoid_25.view(1, -1, 1, 1)
        sigmoid_25 = None
        expand_as_25 = y_77.expand_as(x_285)
        y_77 = None
        x_286 = x_285 * expand_as_25
        x_285 = expand_as_25 = None
        x_286 += x_277
        x_287 = x_286
        x_286 = x_277 = None
        x_288 = torch.nn.functional.relu(x_287, inplace=True)
        x_287 = None
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_layer2_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_23_modules_conv1_parameters_weight_ = None
        x_290 = torch.nn.functional.batch_norm(
            x_289,
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_289 = (
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_23_modules_bn1_parameters_bias_ = None
        x_291 = torch.nn.functional.relu(x_290, inplace=True)
        x_290 = None
        x_292 = torch.conv2d(
            x_291,
            l_self_modules_layer2_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_291 = l_self_modules_layer2_modules_23_modules_conv2_parameters_weight_ = None
        x_293 = torch.nn.functional.batch_norm(
            x_292,
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_292 = (
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_23_modules_bn2_parameters_bias_ = None
        x_294 = torch.nn.functional.relu(x_293, inplace=True)
        x_293 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_layer2_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_layer2_modules_23_modules_conv3_parameters_weight_ = None
        x_296 = torch.nn.functional.batch_norm(
            x_295,
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_295 = (
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_23_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_23_modules_bn3_parameters_bias_ = None
        mean_26 = x_296.mean((2, 3))
        y_78 = mean_26.view(1, 1, -1)
        mean_26 = None
        y_79 = torch.conv1d(
            y_78,
            l_self_modules_layer2_modules_23_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_78 = (
            l_self_modules_layer2_modules_23_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_26 = y_79.sigmoid()
        y_79 = None
        y_80 = sigmoid_26.view(1, -1, 1, 1)
        sigmoid_26 = None
        expand_as_26 = y_80.expand_as(x_296)
        y_80 = None
        x_297 = x_296 * expand_as_26
        x_296 = expand_as_26 = None
        x_297 += x_288
        x_298 = x_297
        x_297 = x_288 = None
        x_299 = torch.nn.functional.relu(x_298, inplace=True)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_layer2_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_24_modules_conv1_parameters_weight_ = None
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_layer2_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_300 = (
            l_self_modules_layer2_modules_24_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_24_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_24_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_24_modules_bn1_parameters_bias_ = None
        x_302 = torch.nn.functional.relu(x_301, inplace=True)
        x_301 = None
        x_303 = torch.conv2d(
            x_302,
            l_self_modules_layer2_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_302 = l_self_modules_layer2_modules_24_modules_conv2_parameters_weight_ = None
        x_304 = torch.nn.functional.batch_norm(
            x_303,
            l_self_modules_layer2_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_303 = (
            l_self_modules_layer2_modules_24_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_24_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_24_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_24_modules_bn2_parameters_bias_ = None
        x_305 = torch.nn.functional.relu(x_304, inplace=True)
        x_304 = None
        x_306 = torch.conv2d(
            x_305,
            l_self_modules_layer2_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_305 = l_self_modules_layer2_modules_24_modules_conv3_parameters_weight_ = None
        x_307 = torch.nn.functional.batch_norm(
            x_306,
            l_self_modules_layer2_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_306 = (
            l_self_modules_layer2_modules_24_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_24_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_24_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_24_modules_bn3_parameters_bias_ = None
        mean_27 = x_307.mean((2, 3))
        y_81 = mean_27.view(1, 1, -1)
        mean_27 = None
        y_82 = torch.conv1d(
            y_81,
            l_self_modules_layer2_modules_24_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_81 = (
            l_self_modules_layer2_modules_24_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_27 = y_82.sigmoid()
        y_82 = None
        y_83 = sigmoid_27.view(1, -1, 1, 1)
        sigmoid_27 = None
        expand_as_27 = y_83.expand_as(x_307)
        y_83 = None
        x_308 = x_307 * expand_as_27
        x_307 = expand_as_27 = None
        x_308 += x_299
        x_309 = x_308
        x_308 = x_299 = None
        x_310 = torch.nn.functional.relu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_layer2_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_25_modules_conv1_parameters_weight_ = None
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_layer2_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = (
            l_self_modules_layer2_modules_25_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_25_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_25_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_25_modules_bn1_parameters_bias_ = None
        x_313 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_layer2_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_layer2_modules_25_modules_conv2_parameters_weight_ = None
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_layer2_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = (
            l_self_modules_layer2_modules_25_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_25_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_25_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_25_modules_bn2_parameters_bias_ = None
        x_316 = torch.nn.functional.relu(x_315, inplace=True)
        x_315 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_layer2_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_316 = l_self_modules_layer2_modules_25_modules_conv3_parameters_weight_ = None
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_layer2_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_317 = (
            l_self_modules_layer2_modules_25_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_25_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_25_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_25_modules_bn3_parameters_bias_ = None
        mean_28 = x_318.mean((2, 3))
        y_84 = mean_28.view(1, 1, -1)
        mean_28 = None
        y_85 = torch.conv1d(
            y_84,
            l_self_modules_layer2_modules_25_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_84 = (
            l_self_modules_layer2_modules_25_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_28 = y_85.sigmoid()
        y_85 = None
        y_86 = sigmoid_28.view(1, -1, 1, 1)
        sigmoid_28 = None
        expand_as_28 = y_86.expand_as(x_318)
        y_86 = None
        x_319 = x_318 * expand_as_28
        x_318 = expand_as_28 = None
        x_319 += x_310
        x_320 = x_319
        x_319 = x_310 = None
        x_321 = torch.nn.functional.relu(x_320, inplace=True)
        x_320 = None
        x_322 = torch.conv2d(
            x_321,
            l_self_modules_layer2_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_26_modules_conv1_parameters_weight_ = None
        x_323 = torch.nn.functional.batch_norm(
            x_322,
            l_self_modules_layer2_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_322 = (
            l_self_modules_layer2_modules_26_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_26_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_26_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_26_modules_bn1_parameters_bias_ = None
        x_324 = torch.nn.functional.relu(x_323, inplace=True)
        x_323 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_layer2_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_324 = l_self_modules_layer2_modules_26_modules_conv2_parameters_weight_ = None
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_layer2_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = (
            l_self_modules_layer2_modules_26_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_26_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_26_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_26_modules_bn2_parameters_bias_ = None
        x_327 = torch.nn.functional.relu(x_326, inplace=True)
        x_326 = None
        x_328 = torch.conv2d(
            x_327,
            l_self_modules_layer2_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_327 = l_self_modules_layer2_modules_26_modules_conv3_parameters_weight_ = None
        x_329 = torch.nn.functional.batch_norm(
            x_328,
            l_self_modules_layer2_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_328 = (
            l_self_modules_layer2_modules_26_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_26_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_26_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_26_modules_bn3_parameters_bias_ = None
        mean_29 = x_329.mean((2, 3))
        y_87 = mean_29.view(1, 1, -1)
        mean_29 = None
        y_88 = torch.conv1d(
            y_87,
            l_self_modules_layer2_modules_26_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_87 = (
            l_self_modules_layer2_modules_26_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_29 = y_88.sigmoid()
        y_88 = None
        y_89 = sigmoid_29.view(1, -1, 1, 1)
        sigmoid_29 = None
        expand_as_29 = y_89.expand_as(x_329)
        y_89 = None
        x_330 = x_329 * expand_as_29
        x_329 = expand_as_29 = None
        x_330 += x_321
        x_331 = x_330
        x_330 = x_321 = None
        x_332 = torch.nn.functional.relu(x_331, inplace=True)
        x_331 = None
        x_333 = torch.conv2d(
            x_332,
            l_self_modules_layer2_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_27_modules_conv1_parameters_weight_ = None
        x_334 = torch.nn.functional.batch_norm(
            x_333,
            l_self_modules_layer2_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_333 = (
            l_self_modules_layer2_modules_27_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_27_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_27_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_27_modules_bn1_parameters_bias_ = None
        x_335 = torch.nn.functional.relu(x_334, inplace=True)
        x_334 = None
        x_336 = torch.conv2d(
            x_335,
            l_self_modules_layer2_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_335 = l_self_modules_layer2_modules_27_modules_conv2_parameters_weight_ = None
        x_337 = torch.nn.functional.batch_norm(
            x_336,
            l_self_modules_layer2_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_336 = (
            l_self_modules_layer2_modules_27_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_27_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_27_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_27_modules_bn2_parameters_bias_ = None
        x_338 = torch.nn.functional.relu(x_337, inplace=True)
        x_337 = None
        x_339 = torch.conv2d(
            x_338,
            l_self_modules_layer2_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_338 = l_self_modules_layer2_modules_27_modules_conv3_parameters_weight_ = None
        x_340 = torch.nn.functional.batch_norm(
            x_339,
            l_self_modules_layer2_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_339 = (
            l_self_modules_layer2_modules_27_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_27_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_27_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_27_modules_bn3_parameters_bias_ = None
        mean_30 = x_340.mean((2, 3))
        y_90 = mean_30.view(1, 1, -1)
        mean_30 = None
        y_91 = torch.conv1d(
            y_90,
            l_self_modules_layer2_modules_27_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_90 = (
            l_self_modules_layer2_modules_27_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_30 = y_91.sigmoid()
        y_91 = None
        y_92 = sigmoid_30.view(1, -1, 1, 1)
        sigmoid_30 = None
        expand_as_30 = y_92.expand_as(x_340)
        y_92 = None
        x_341 = x_340 * expand_as_30
        x_340 = expand_as_30 = None
        x_341 += x_332
        x_342 = x_341
        x_341 = x_332 = None
        x_343 = torch.nn.functional.relu(x_342, inplace=True)
        x_342 = None
        x_344 = torch.conv2d(
            x_343,
            l_self_modules_layer2_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_28_modules_conv1_parameters_weight_ = None
        x_345 = torch.nn.functional.batch_norm(
            x_344,
            l_self_modules_layer2_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_344 = (
            l_self_modules_layer2_modules_28_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_28_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_28_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_28_modules_bn1_parameters_bias_ = None
        x_346 = torch.nn.functional.relu(x_345, inplace=True)
        x_345 = None
        x_347 = torch.conv2d(
            x_346,
            l_self_modules_layer2_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_346 = l_self_modules_layer2_modules_28_modules_conv2_parameters_weight_ = None
        x_348 = torch.nn.functional.batch_norm(
            x_347,
            l_self_modules_layer2_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_347 = (
            l_self_modules_layer2_modules_28_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_28_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_28_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_28_modules_bn2_parameters_bias_ = None
        x_349 = torch.nn.functional.relu(x_348, inplace=True)
        x_348 = None
        x_350 = torch.conv2d(
            x_349,
            l_self_modules_layer2_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_349 = l_self_modules_layer2_modules_28_modules_conv3_parameters_weight_ = None
        x_351 = torch.nn.functional.batch_norm(
            x_350,
            l_self_modules_layer2_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_350 = (
            l_self_modules_layer2_modules_28_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_28_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_28_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_28_modules_bn3_parameters_bias_ = None
        mean_31 = x_351.mean((2, 3))
        y_93 = mean_31.view(1, 1, -1)
        mean_31 = None
        y_94 = torch.conv1d(
            y_93,
            l_self_modules_layer2_modules_28_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_93 = (
            l_self_modules_layer2_modules_28_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_31 = y_94.sigmoid()
        y_94 = None
        y_95 = sigmoid_31.view(1, -1, 1, 1)
        sigmoid_31 = None
        expand_as_31 = y_95.expand_as(x_351)
        y_95 = None
        x_352 = x_351 * expand_as_31
        x_351 = expand_as_31 = None
        x_352 += x_343
        x_353 = x_352
        x_352 = x_343 = None
        x_354 = torch.nn.functional.relu(x_353, inplace=True)
        x_353 = None
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_layer2_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_29_modules_conv1_parameters_weight_ = None
        x_356 = torch.nn.functional.batch_norm(
            x_355,
            l_self_modules_layer2_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_355 = (
            l_self_modules_layer2_modules_29_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_29_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_29_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_29_modules_bn1_parameters_bias_ = None
        x_357 = torch.nn.functional.relu(x_356, inplace=True)
        x_356 = None
        x_358 = torch.conv2d(
            x_357,
            l_self_modules_layer2_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_357 = l_self_modules_layer2_modules_29_modules_conv2_parameters_weight_ = None
        x_359 = torch.nn.functional.batch_norm(
            x_358,
            l_self_modules_layer2_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_358 = (
            l_self_modules_layer2_modules_29_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_29_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_29_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_29_modules_bn2_parameters_bias_ = None
        x_360 = torch.nn.functional.relu(x_359, inplace=True)
        x_359 = None
        x_361 = torch.conv2d(
            x_360,
            l_self_modules_layer2_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_360 = l_self_modules_layer2_modules_29_modules_conv3_parameters_weight_ = None
        x_362 = torch.nn.functional.batch_norm(
            x_361,
            l_self_modules_layer2_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_361 = (
            l_self_modules_layer2_modules_29_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_29_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_29_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_29_modules_bn3_parameters_bias_ = None
        mean_32 = x_362.mean((2, 3))
        y_96 = mean_32.view(1, 1, -1)
        mean_32 = None
        y_97 = torch.conv1d(
            y_96,
            l_self_modules_layer2_modules_29_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_96 = (
            l_self_modules_layer2_modules_29_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_32 = y_97.sigmoid()
        y_97 = None
        y_98 = sigmoid_32.view(1, -1, 1, 1)
        sigmoid_32 = None
        expand_as_32 = y_98.expand_as(x_362)
        y_98 = None
        x_363 = x_362 * expand_as_32
        x_362 = expand_as_32 = None
        x_363 += x_354
        x_364 = x_363
        x_363 = x_354 = None
        x_365 = torch.nn.functional.relu(x_364, inplace=True)
        x_364 = None
        x_366 = torch.conv2d(
            x_365,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_367 = torch.nn.functional.batch_norm(
            x_366,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_366 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_368 = torch.nn.functional.relu(x_367, inplace=True)
        x_367 = None
        x_369 = torch.conv2d(
            x_368,
            l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_368 = l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        x_371 = torch.nn.functional.relu(x_370, inplace=True)
        x_370 = None
        x_372 = torch.conv2d(
            x_371,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_371 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        x_373 = torch.nn.functional.batch_norm(
            x_372,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_372 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        mean_33 = x_373.mean((2, 3))
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        y_99 = mean_33.view(1, 1, -1)
        mean_33 = None
        y_100 = torch.conv1d(
            y_99,
            l_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_99 = (
            l_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_33 = y_100.sigmoid()
        y_100 = None
        y_101 = sigmoid_33.view(1, -1, 1, 1)
        sigmoid_33 = None
        expand_as_33 = y_101.expand_as(x_373)
        y_101 = None
        x_374 = x_373 * expand_as_33
        x_373 = expand_as_33 = None
        input_13 = torch._C._nn.avg_pool2d(x_365, 2, 2, 0, True, False, None)
        x_365 = None
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
        x_374 += input_15
        x_375 = x_374
        x_374 = input_15 = None
        x_376 = torch.nn.functional.relu(x_375, inplace=True)
        x_375 = None
        x_377 = torch.conv2d(
            x_376,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_378 = torch.nn.functional.batch_norm(
            x_377,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_377 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_379 = torch.nn.functional.relu(x_378, inplace=True)
        x_378 = None
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_381 = torch.nn.functional.batch_norm(
            x_380,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_380 = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_382 = torch.nn.functional.relu(x_381, inplace=True)
        x_381 = None
        x_383 = torch.conv2d(
            x_382,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_382 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_383 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        mean_34 = x_384.mean((2, 3))
        y_102 = mean_34.view(1, 1, -1)
        mean_34 = None
        y_103 = torch.conv1d(
            y_102,
            l_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_102 = (
            l_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_34 = y_103.sigmoid()
        y_103 = None
        y_104 = sigmoid_34.view(1, -1, 1, 1)
        sigmoid_34 = None
        expand_as_34 = y_104.expand_as(x_384)
        y_104 = None
        x_385 = x_384 * expand_as_34
        x_384 = expand_as_34 = None
        x_385 += x_376
        x_386 = x_385
        x_385 = x_376 = None
        x_387 = torch.nn.functional.relu(x_386, inplace=True)
        x_386 = None
        x_388 = torch.conv2d(
            x_387,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        x_389 = torch.nn.functional.batch_norm(
            x_388,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_388 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        x_390 = torch.nn.functional.relu(x_389, inplace=True)
        x_389 = None
        x_391 = torch.conv2d(
            x_390,
            l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_390 = l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_ = None
        x_392 = torch.nn.functional.batch_norm(
            x_391,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_391 = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_ = None
        x_393 = torch.nn.functional.relu(x_392, inplace=True)
        x_392 = None
        x_394 = torch.conv2d(
            x_393,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_393 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        x_395 = torch.nn.functional.batch_norm(
            x_394,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_394 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        mean_35 = x_395.mean((2, 3))
        y_105 = mean_35.view(1, 1, -1)
        mean_35 = None
        y_106 = torch.conv1d(
            y_105,
            l_self_modules_layer3_modules_2_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_105 = (
            l_self_modules_layer3_modules_2_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_35 = y_106.sigmoid()
        y_106 = None
        y_107 = sigmoid_35.view(1, -1, 1, 1)
        sigmoid_35 = None
        expand_as_35 = y_107.expand_as(x_395)
        y_107 = None
        x_396 = x_395 * expand_as_35
        x_395 = expand_as_35 = None
        x_396 += x_387
        x_397 = x_396
        x_396 = x_387 = None
        x_398 = torch.nn.functional.relu(x_397, inplace=True)
        x_397 = None
        x_399 = torch.conv2d(
            x_398,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        x_400 = torch.nn.functional.batch_norm(
            x_399,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_399 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        x_401 = torch.nn.functional.relu(x_400, inplace=True)
        x_400 = None
        x_402 = torch.conv2d(
            x_401,
            l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_401 = l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_ = None
        x_403 = torch.nn.functional.batch_norm(
            x_402,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_402 = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_ = None
        x_404 = torch.nn.functional.relu(x_403, inplace=True)
        x_403 = None
        x_405 = torch.conv2d(
            x_404,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_404 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        x_406 = torch.nn.functional.batch_norm(
            x_405,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_405 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        mean_36 = x_406.mean((2, 3))
        y_108 = mean_36.view(1, 1, -1)
        mean_36 = None
        y_109 = torch.conv1d(
            y_108,
            l_self_modules_layer3_modules_3_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_108 = (
            l_self_modules_layer3_modules_3_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_36 = y_109.sigmoid()
        y_109 = None
        y_110 = sigmoid_36.view(1, -1, 1, 1)
        sigmoid_36 = None
        expand_as_36 = y_110.expand_as(x_406)
        y_110 = None
        x_407 = x_406 * expand_as_36
        x_406 = expand_as_36 = None
        x_407 += x_398
        x_408 = x_407
        x_407 = x_398 = None
        x_409 = torch.nn.functional.relu(x_408, inplace=True)
        x_408 = None
        x_410 = torch.conv2d(
            x_409,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        x_411 = torch.nn.functional.batch_norm(
            x_410,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_410 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        x_412 = torch.nn.functional.relu(x_411, inplace=True)
        x_411 = None
        x_413 = torch.conv2d(
            x_412,
            l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_412 = l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_ = None
        x_414 = torch.nn.functional.batch_norm(
            x_413,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_413 = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_ = None
        x_415 = torch.nn.functional.relu(x_414, inplace=True)
        x_414 = None
        x_416 = torch.conv2d(
            x_415,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_415 = l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = None
        x_417 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_416 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        mean_37 = x_417.mean((2, 3))
        y_111 = mean_37.view(1, 1, -1)
        mean_37 = None
        y_112 = torch.conv1d(
            y_111,
            l_self_modules_layer3_modules_4_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_111 = (
            l_self_modules_layer3_modules_4_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_37 = y_112.sigmoid()
        y_112 = None
        y_113 = sigmoid_37.view(1, -1, 1, 1)
        sigmoid_37 = None
        expand_as_37 = y_113.expand_as(x_417)
        y_113 = None
        x_418 = x_417 * expand_as_37
        x_417 = expand_as_37 = None
        x_418 += x_409
        x_419 = x_418
        x_418 = x_409 = None
        x_420 = torch.nn.functional.relu(x_419, inplace=True)
        x_419 = None
        x_421 = torch.conv2d(
            x_420,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        x_422 = torch.nn.functional.batch_norm(
            x_421,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_421 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        x_423 = torch.nn.functional.relu(x_422, inplace=True)
        x_422 = None
        x_424 = torch.conv2d(
            x_423,
            l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_423 = l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_ = None
        x_425 = torch.nn.functional.batch_norm(
            x_424,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_424 = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_ = None
        x_426 = torch.nn.functional.relu(x_425, inplace=True)
        x_425 = None
        x_427 = torch.conv2d(
            x_426,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_426 = l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = None
        x_428 = torch.nn.functional.batch_norm(
            x_427,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_427 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        mean_38 = x_428.mean((2, 3))
        y_114 = mean_38.view(1, 1, -1)
        mean_38 = None
        y_115 = torch.conv1d(
            y_114,
            l_self_modules_layer3_modules_5_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_114 = (
            l_self_modules_layer3_modules_5_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_38 = y_115.sigmoid()
        y_115 = None
        y_116 = sigmoid_38.view(1, -1, 1, 1)
        sigmoid_38 = None
        expand_as_38 = y_116.expand_as(x_428)
        y_116 = None
        x_429 = x_428 * expand_as_38
        x_428 = expand_as_38 = None
        x_429 += x_420
        x_430 = x_429
        x_429 = x_420 = None
        x_431 = torch.nn.functional.relu(x_430, inplace=True)
        x_430 = None
        x_432 = torch.conv2d(
            x_431,
            l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = None
        x_433 = torch.nn.functional.batch_norm(
            x_432,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_432 = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = None
        x_434 = torch.nn.functional.relu(x_433, inplace=True)
        x_433 = None
        x_435 = torch.conv2d(
            x_434,
            l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_434 = l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_ = None
        x_436 = torch.nn.functional.batch_norm(
            x_435,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_435 = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_ = None
        x_437 = torch.nn.functional.relu(x_436, inplace=True)
        x_436 = None
        x_438 = torch.conv2d(
            x_437,
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_437 = l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = None
        x_439 = torch.nn.functional.batch_norm(
            x_438,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_438 = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = None
        mean_39 = x_439.mean((2, 3))
        y_117 = mean_39.view(1, 1, -1)
        mean_39 = None
        y_118 = torch.conv1d(
            y_117,
            l_self_modules_layer3_modules_6_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_117 = (
            l_self_modules_layer3_modules_6_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_39 = y_118.sigmoid()
        y_118 = None
        y_119 = sigmoid_39.view(1, -1, 1, 1)
        sigmoid_39 = None
        expand_as_39 = y_119.expand_as(x_439)
        y_119 = None
        x_440 = x_439 * expand_as_39
        x_439 = expand_as_39 = None
        x_440 += x_431
        x_441 = x_440
        x_440 = x_431 = None
        x_442 = torch.nn.functional.relu(x_441, inplace=True)
        x_441 = None
        x_443 = torch.conv2d(
            x_442,
            l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = None
        x_444 = torch.nn.functional.batch_norm(
            x_443,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_443 = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = None
        x_445 = torch.nn.functional.relu(x_444, inplace=True)
        x_444 = None
        x_446 = torch.conv2d(
            x_445,
            l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_445 = l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_ = None
        x_447 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_446 = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_ = None
        x_448 = torch.nn.functional.relu(x_447, inplace=True)
        x_447 = None
        x_449 = torch.conv2d(
            x_448,
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_448 = l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = None
        x_450 = torch.nn.functional.batch_norm(
            x_449,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_449 = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = None
        mean_40 = x_450.mean((2, 3))
        y_120 = mean_40.view(1, 1, -1)
        mean_40 = None
        y_121 = torch.conv1d(
            y_120,
            l_self_modules_layer3_modules_7_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_120 = (
            l_self_modules_layer3_modules_7_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_40 = y_121.sigmoid()
        y_121 = None
        y_122 = sigmoid_40.view(1, -1, 1, 1)
        sigmoid_40 = None
        expand_as_40 = y_122.expand_as(x_450)
        y_122 = None
        x_451 = x_450 * expand_as_40
        x_450 = expand_as_40 = None
        x_451 += x_442
        x_452 = x_451
        x_451 = x_442 = None
        x_453 = torch.nn.functional.relu(x_452, inplace=True)
        x_452 = None
        x_454 = torch.conv2d(
            x_453,
            l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = None
        x_455 = torch.nn.functional.batch_norm(
            x_454,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_454 = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = None
        x_456 = torch.nn.functional.relu(x_455, inplace=True)
        x_455 = None
        x_457 = torch.conv2d(
            x_456,
            l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_456 = l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_ = None
        x_458 = torch.nn.functional.batch_norm(
            x_457,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_457 = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_ = None
        x_459 = torch.nn.functional.relu(x_458, inplace=True)
        x_458 = None
        x_460 = torch.conv2d(
            x_459,
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_459 = l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = None
        x_461 = torch.nn.functional.batch_norm(
            x_460,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_460 = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = None
        mean_41 = x_461.mean((2, 3))
        y_123 = mean_41.view(1, 1, -1)
        mean_41 = None
        y_124 = torch.conv1d(
            y_123,
            l_self_modules_layer3_modules_8_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_123 = (
            l_self_modules_layer3_modules_8_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_41 = y_124.sigmoid()
        y_124 = None
        y_125 = sigmoid_41.view(1, -1, 1, 1)
        sigmoid_41 = None
        expand_as_41 = y_125.expand_as(x_461)
        y_125 = None
        x_462 = x_461 * expand_as_41
        x_461 = expand_as_41 = None
        x_462 += x_453
        x_463 = x_462
        x_462 = x_453 = None
        x_464 = torch.nn.functional.relu(x_463, inplace=True)
        x_463 = None
        x_465 = torch.conv2d(
            x_464,
            l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = None
        x_466 = torch.nn.functional.batch_norm(
            x_465,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_465 = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = None
        x_467 = torch.nn.functional.relu(x_466, inplace=True)
        x_466 = None
        x_468 = torch.conv2d(
            x_467,
            l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_467 = l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_ = None
        x_469 = torch.nn.functional.batch_norm(
            x_468,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_468 = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_ = None
        x_470 = torch.nn.functional.relu(x_469, inplace=True)
        x_469 = None
        x_471 = torch.conv2d(
            x_470,
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_470 = l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = None
        x_472 = torch.nn.functional.batch_norm(
            x_471,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_471 = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = None
        mean_42 = x_472.mean((2, 3))
        y_126 = mean_42.view(1, 1, -1)
        mean_42 = None
        y_127 = torch.conv1d(
            y_126,
            l_self_modules_layer3_modules_9_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_126 = (
            l_self_modules_layer3_modules_9_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_42 = y_127.sigmoid()
        y_127 = None
        y_128 = sigmoid_42.view(1, -1, 1, 1)
        sigmoid_42 = None
        expand_as_42 = y_128.expand_as(x_472)
        y_128 = None
        x_473 = x_472 * expand_as_42
        x_472 = expand_as_42 = None
        x_473 += x_464
        x_474 = x_473
        x_473 = x_464 = None
        x_475 = torch.nn.functional.relu(x_474, inplace=True)
        x_474 = None
        x_476 = torch.conv2d(
            x_475,
            l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = None
        x_477 = torch.nn.functional.batch_norm(
            x_476,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_476 = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = None
        x_478 = torch.nn.functional.relu(x_477, inplace=True)
        x_477 = None
        x_479 = torch.conv2d(
            x_478,
            l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_478 = l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_ = None
        x_480 = torch.nn.functional.batch_norm(
            x_479,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_479 = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_ = None
        x_481 = torch.nn.functional.relu(x_480, inplace=True)
        x_480 = None
        x_482 = torch.conv2d(
            x_481,
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_481 = l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = None
        x_483 = torch.nn.functional.batch_norm(
            x_482,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_482 = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = None
        mean_43 = x_483.mean((2, 3))
        y_129 = mean_43.view(1, 1, -1)
        mean_43 = None
        y_130 = torch.conv1d(
            y_129,
            l_self_modules_layer3_modules_10_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_129 = (
            l_self_modules_layer3_modules_10_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_43 = y_130.sigmoid()
        y_130 = None
        y_131 = sigmoid_43.view(1, -1, 1, 1)
        sigmoid_43 = None
        expand_as_43 = y_131.expand_as(x_483)
        y_131 = None
        x_484 = x_483 * expand_as_43
        x_483 = expand_as_43 = None
        x_484 += x_475
        x_485 = x_484
        x_484 = x_475 = None
        x_486 = torch.nn.functional.relu(x_485, inplace=True)
        x_485 = None
        x_487 = torch.conv2d(
            x_486,
            l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = None
        x_488 = torch.nn.functional.batch_norm(
            x_487,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_487 = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = None
        x_489 = torch.nn.functional.relu(x_488, inplace=True)
        x_488 = None
        x_490 = torch.conv2d(
            x_489,
            l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_489 = l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_ = None
        x_491 = torch.nn.functional.batch_norm(
            x_490,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_490 = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_ = None
        x_492 = torch.nn.functional.relu(x_491, inplace=True)
        x_491 = None
        x_493 = torch.conv2d(
            x_492,
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_492 = l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = None
        x_494 = torch.nn.functional.batch_norm(
            x_493,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_493 = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = None
        mean_44 = x_494.mean((2, 3))
        y_132 = mean_44.view(1, 1, -1)
        mean_44 = None
        y_133 = torch.conv1d(
            y_132,
            l_self_modules_layer3_modules_11_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_132 = (
            l_self_modules_layer3_modules_11_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_44 = y_133.sigmoid()
        y_133 = None
        y_134 = sigmoid_44.view(1, -1, 1, 1)
        sigmoid_44 = None
        expand_as_44 = y_134.expand_as(x_494)
        y_134 = None
        x_495 = x_494 * expand_as_44
        x_494 = expand_as_44 = None
        x_495 += x_486
        x_496 = x_495
        x_495 = x_486 = None
        x_497 = torch.nn.functional.relu(x_496, inplace=True)
        x_496 = None
        x_498 = torch.conv2d(
            x_497,
            l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = None
        x_499 = torch.nn.functional.batch_norm(
            x_498,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_498 = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = None
        x_500 = torch.nn.functional.relu(x_499, inplace=True)
        x_499 = None
        x_501 = torch.conv2d(
            x_500,
            l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_500 = l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_ = None
        x_502 = torch.nn.functional.batch_norm(
            x_501,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_501 = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_ = None
        x_503 = torch.nn.functional.relu(x_502, inplace=True)
        x_502 = None
        x_504 = torch.conv2d(
            x_503,
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_503 = l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = None
        x_505 = torch.nn.functional.batch_norm(
            x_504,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_504 = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = None
        mean_45 = x_505.mean((2, 3))
        y_135 = mean_45.view(1, 1, -1)
        mean_45 = None
        y_136 = torch.conv1d(
            y_135,
            l_self_modules_layer3_modules_12_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_135 = (
            l_self_modules_layer3_modules_12_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_45 = y_136.sigmoid()
        y_136 = None
        y_137 = sigmoid_45.view(1, -1, 1, 1)
        sigmoid_45 = None
        expand_as_45 = y_137.expand_as(x_505)
        y_137 = None
        x_506 = x_505 * expand_as_45
        x_505 = expand_as_45 = None
        x_506 += x_497
        x_507 = x_506
        x_506 = x_497 = None
        x_508 = torch.nn.functional.relu(x_507, inplace=True)
        x_507 = None
        x_509 = torch.conv2d(
            x_508,
            l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = None
        x_510 = torch.nn.functional.batch_norm(
            x_509,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_509 = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = None
        x_511 = torch.nn.functional.relu(x_510, inplace=True)
        x_510 = None
        x_512 = torch.conv2d(
            x_511,
            l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_511 = l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_ = None
        x_513 = torch.nn.functional.batch_norm(
            x_512,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_512 = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_ = None
        x_514 = torch.nn.functional.relu(x_513, inplace=True)
        x_513 = None
        x_515 = torch.conv2d(
            x_514,
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_514 = l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = None
        x_516 = torch.nn.functional.batch_norm(
            x_515,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_515 = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = None
        mean_46 = x_516.mean((2, 3))
        y_138 = mean_46.view(1, 1, -1)
        mean_46 = None
        y_139 = torch.conv1d(
            y_138,
            l_self_modules_layer3_modules_13_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_138 = (
            l_self_modules_layer3_modules_13_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_46 = y_139.sigmoid()
        y_139 = None
        y_140 = sigmoid_46.view(1, -1, 1, 1)
        sigmoid_46 = None
        expand_as_46 = y_140.expand_as(x_516)
        y_140 = None
        x_517 = x_516 * expand_as_46
        x_516 = expand_as_46 = None
        x_517 += x_508
        x_518 = x_517
        x_517 = x_508 = None
        x_519 = torch.nn.functional.relu(x_518, inplace=True)
        x_518 = None
        x_520 = torch.conv2d(
            x_519,
            l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = None
        x_521 = torch.nn.functional.batch_norm(
            x_520,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_520 = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = None
        x_522 = torch.nn.functional.relu(x_521, inplace=True)
        x_521 = None
        x_523 = torch.conv2d(
            x_522,
            l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_522 = l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_ = None
        x_524 = torch.nn.functional.batch_norm(
            x_523,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_523 = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_ = None
        x_525 = torch.nn.functional.relu(x_524, inplace=True)
        x_524 = None
        x_526 = torch.conv2d(
            x_525,
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_525 = l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = None
        x_527 = torch.nn.functional.batch_norm(
            x_526,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_526 = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = None
        mean_47 = x_527.mean((2, 3))
        y_141 = mean_47.view(1, 1, -1)
        mean_47 = None
        y_142 = torch.conv1d(
            y_141,
            l_self_modules_layer3_modules_14_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_141 = (
            l_self_modules_layer3_modules_14_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_47 = y_142.sigmoid()
        y_142 = None
        y_143 = sigmoid_47.view(1, -1, 1, 1)
        sigmoid_47 = None
        expand_as_47 = y_143.expand_as(x_527)
        y_143 = None
        x_528 = x_527 * expand_as_47
        x_527 = expand_as_47 = None
        x_528 += x_519
        x_529 = x_528
        x_528 = x_519 = None
        x_530 = torch.nn.functional.relu(x_529, inplace=True)
        x_529 = None
        x_531 = torch.conv2d(
            x_530,
            l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = None
        x_532 = torch.nn.functional.batch_norm(
            x_531,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_531 = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = None
        x_533 = torch.nn.functional.relu(x_532, inplace=True)
        x_532 = None
        x_534 = torch.conv2d(
            x_533,
            l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_533 = l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_ = None
        x_535 = torch.nn.functional.batch_norm(
            x_534,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_534 = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_ = None
        x_536 = torch.nn.functional.relu(x_535, inplace=True)
        x_535 = None
        x_537 = torch.conv2d(
            x_536,
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_536 = l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = None
        x_538 = torch.nn.functional.batch_norm(
            x_537,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_537 = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = None
        mean_48 = x_538.mean((2, 3))
        y_144 = mean_48.view(1, 1, -1)
        mean_48 = None
        y_145 = torch.conv1d(
            y_144,
            l_self_modules_layer3_modules_15_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_144 = (
            l_self_modules_layer3_modules_15_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_48 = y_145.sigmoid()
        y_145 = None
        y_146 = sigmoid_48.view(1, -1, 1, 1)
        sigmoid_48 = None
        expand_as_48 = y_146.expand_as(x_538)
        y_146 = None
        x_539 = x_538 * expand_as_48
        x_538 = expand_as_48 = None
        x_539 += x_530
        x_540 = x_539
        x_539 = x_530 = None
        x_541 = torch.nn.functional.relu(x_540, inplace=True)
        x_540 = None
        x_542 = torch.conv2d(
            x_541,
            l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = None
        x_543 = torch.nn.functional.batch_norm(
            x_542,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_542 = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = None
        x_544 = torch.nn.functional.relu(x_543, inplace=True)
        x_543 = None
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_ = None
        x_546 = torch.nn.functional.batch_norm(
            x_545,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_545 = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_ = None
        x_547 = torch.nn.functional.relu(x_546, inplace=True)
        x_546 = None
        x_548 = torch.conv2d(
            x_547,
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_547 = l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = None
        x_549 = torch.nn.functional.batch_norm(
            x_548,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_548 = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = None
        mean_49 = x_549.mean((2, 3))
        y_147 = mean_49.view(1, 1, -1)
        mean_49 = None
        y_148 = torch.conv1d(
            y_147,
            l_self_modules_layer3_modules_16_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_147 = (
            l_self_modules_layer3_modules_16_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_49 = y_148.sigmoid()
        y_148 = None
        y_149 = sigmoid_49.view(1, -1, 1, 1)
        sigmoid_49 = None
        expand_as_49 = y_149.expand_as(x_549)
        y_149 = None
        x_550 = x_549 * expand_as_49
        x_549 = expand_as_49 = None
        x_550 += x_541
        x_551 = x_550
        x_550 = x_541 = None
        x_552 = torch.nn.functional.relu(x_551, inplace=True)
        x_551 = None
        x_553 = torch.conv2d(
            x_552,
            l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = None
        x_554 = torch.nn.functional.batch_norm(
            x_553,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_553 = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = None
        x_555 = torch.nn.functional.relu(x_554, inplace=True)
        x_554 = None
        x_556 = torch.conv2d(
            x_555,
            l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_555 = l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_ = None
        x_557 = torch.nn.functional.batch_norm(
            x_556,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_556 = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_ = None
        x_558 = torch.nn.functional.relu(x_557, inplace=True)
        x_557 = None
        x_559 = torch.conv2d(
            x_558,
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_558 = l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = None
        x_560 = torch.nn.functional.batch_norm(
            x_559,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_559 = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = None
        mean_50 = x_560.mean((2, 3))
        y_150 = mean_50.view(1, 1, -1)
        mean_50 = None
        y_151 = torch.conv1d(
            y_150,
            l_self_modules_layer3_modules_17_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_150 = (
            l_self_modules_layer3_modules_17_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_50 = y_151.sigmoid()
        y_151 = None
        y_152 = sigmoid_50.view(1, -1, 1, 1)
        sigmoid_50 = None
        expand_as_50 = y_152.expand_as(x_560)
        y_152 = None
        x_561 = x_560 * expand_as_50
        x_560 = expand_as_50 = None
        x_561 += x_552
        x_562 = x_561
        x_561 = x_552 = None
        x_563 = torch.nn.functional.relu(x_562, inplace=True)
        x_562 = None
        x_564 = torch.conv2d(
            x_563,
            l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = None
        x_565 = torch.nn.functional.batch_norm(
            x_564,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_564 = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = None
        x_566 = torch.nn.functional.relu(x_565, inplace=True)
        x_565 = None
        x_567 = torch.conv2d(
            x_566,
            l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_566 = l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_ = None
        x_568 = torch.nn.functional.batch_norm(
            x_567,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_567 = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_ = None
        x_569 = torch.nn.functional.relu(x_568, inplace=True)
        x_568 = None
        x_570 = torch.conv2d(
            x_569,
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_569 = l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = None
        x_571 = torch.nn.functional.batch_norm(
            x_570,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_570 = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = None
        mean_51 = x_571.mean((2, 3))
        y_153 = mean_51.view(1, 1, -1)
        mean_51 = None
        y_154 = torch.conv1d(
            y_153,
            l_self_modules_layer3_modules_18_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_153 = (
            l_self_modules_layer3_modules_18_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_51 = y_154.sigmoid()
        y_154 = None
        y_155 = sigmoid_51.view(1, -1, 1, 1)
        sigmoid_51 = None
        expand_as_51 = y_155.expand_as(x_571)
        y_155 = None
        x_572 = x_571 * expand_as_51
        x_571 = expand_as_51 = None
        x_572 += x_563
        x_573 = x_572
        x_572 = x_563 = None
        x_574 = torch.nn.functional.relu(x_573, inplace=True)
        x_573 = None
        x_575 = torch.conv2d(
            x_574,
            l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = None
        x_576 = torch.nn.functional.batch_norm(
            x_575,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_575 = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = None
        x_577 = torch.nn.functional.relu(x_576, inplace=True)
        x_576 = None
        x_578 = torch.conv2d(
            x_577,
            l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_577 = l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_ = None
        x_579 = torch.nn.functional.batch_norm(
            x_578,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_578 = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_ = None
        x_580 = torch.nn.functional.relu(x_579, inplace=True)
        x_579 = None
        x_581 = torch.conv2d(
            x_580,
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_580 = l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = None
        x_582 = torch.nn.functional.batch_norm(
            x_581,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_581 = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = None
        mean_52 = x_582.mean((2, 3))
        y_156 = mean_52.view(1, 1, -1)
        mean_52 = None
        y_157 = torch.conv1d(
            y_156,
            l_self_modules_layer3_modules_19_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_156 = (
            l_self_modules_layer3_modules_19_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_52 = y_157.sigmoid()
        y_157 = None
        y_158 = sigmoid_52.view(1, -1, 1, 1)
        sigmoid_52 = None
        expand_as_52 = y_158.expand_as(x_582)
        y_158 = None
        x_583 = x_582 * expand_as_52
        x_582 = expand_as_52 = None
        x_583 += x_574
        x_584 = x_583
        x_583 = x_574 = None
        x_585 = torch.nn.functional.relu(x_584, inplace=True)
        x_584 = None
        x_586 = torch.conv2d(
            x_585,
            l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = None
        x_587 = torch.nn.functional.batch_norm(
            x_586,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_586 = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = None
        x_588 = torch.nn.functional.relu(x_587, inplace=True)
        x_587 = None
        x_589 = torch.conv2d(
            x_588,
            l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_588 = l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_ = None
        x_590 = torch.nn.functional.batch_norm(
            x_589,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_589 = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_ = None
        x_591 = torch.nn.functional.relu(x_590, inplace=True)
        x_590 = None
        x_592 = torch.conv2d(
            x_591,
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_591 = l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = None
        x_593 = torch.nn.functional.batch_norm(
            x_592,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_592 = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = None
        mean_53 = x_593.mean((2, 3))
        y_159 = mean_53.view(1, 1, -1)
        mean_53 = None
        y_160 = torch.conv1d(
            y_159,
            l_self_modules_layer3_modules_20_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_159 = (
            l_self_modules_layer3_modules_20_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_53 = y_160.sigmoid()
        y_160 = None
        y_161 = sigmoid_53.view(1, -1, 1, 1)
        sigmoid_53 = None
        expand_as_53 = y_161.expand_as(x_593)
        y_161 = None
        x_594 = x_593 * expand_as_53
        x_593 = expand_as_53 = None
        x_594 += x_585
        x_595 = x_594
        x_594 = x_585 = None
        x_596 = torch.nn.functional.relu(x_595, inplace=True)
        x_595 = None
        x_597 = torch.conv2d(
            x_596,
            l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = None
        x_598 = torch.nn.functional.batch_norm(
            x_597,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_597 = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = None
        x_599 = torch.nn.functional.relu(x_598, inplace=True)
        x_598 = None
        x_600 = torch.conv2d(
            x_599,
            l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_599 = l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_ = None
        x_601 = torch.nn.functional.batch_norm(
            x_600,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_600 = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_ = None
        x_602 = torch.nn.functional.relu(x_601, inplace=True)
        x_601 = None
        x_603 = torch.conv2d(
            x_602,
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_602 = l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = None
        x_604 = torch.nn.functional.batch_norm(
            x_603,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_603 = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = None
        mean_54 = x_604.mean((2, 3))
        y_162 = mean_54.view(1, 1, -1)
        mean_54 = None
        y_163 = torch.conv1d(
            y_162,
            l_self_modules_layer3_modules_21_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_162 = (
            l_self_modules_layer3_modules_21_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_54 = y_163.sigmoid()
        y_163 = None
        y_164 = sigmoid_54.view(1, -1, 1, 1)
        sigmoid_54 = None
        expand_as_54 = y_164.expand_as(x_604)
        y_164 = None
        x_605 = x_604 * expand_as_54
        x_604 = expand_as_54 = None
        x_605 += x_596
        x_606 = x_605
        x_605 = x_596 = None
        x_607 = torch.nn.functional.relu(x_606, inplace=True)
        x_606 = None
        x_608 = torch.conv2d(
            x_607,
            l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = None
        x_609 = torch.nn.functional.batch_norm(
            x_608,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_608 = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = None
        x_610 = torch.nn.functional.relu(x_609, inplace=True)
        x_609 = None
        x_611 = torch.conv2d(
            x_610,
            l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_610 = l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_ = None
        x_612 = torch.nn.functional.batch_norm(
            x_611,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_611 = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_ = None
        x_613 = torch.nn.functional.relu(x_612, inplace=True)
        x_612 = None
        x_614 = torch.conv2d(
            x_613,
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_613 = l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = None
        x_615 = torch.nn.functional.batch_norm(
            x_614,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_614 = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = None
        mean_55 = x_615.mean((2, 3))
        y_165 = mean_55.view(1, 1, -1)
        mean_55 = None
        y_166 = torch.conv1d(
            y_165,
            l_self_modules_layer3_modules_22_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_165 = (
            l_self_modules_layer3_modules_22_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_55 = y_166.sigmoid()
        y_166 = None
        y_167 = sigmoid_55.view(1, -1, 1, 1)
        sigmoid_55 = None
        expand_as_55 = y_167.expand_as(x_615)
        y_167 = None
        x_616 = x_615 * expand_as_55
        x_615 = expand_as_55 = None
        x_616 += x_607
        x_617 = x_616
        x_616 = x_607 = None
        x_618 = torch.nn.functional.relu(x_617, inplace=True)
        x_617 = None
        x_619 = torch.conv2d(
            x_618,
            l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_23_modules_conv1_parameters_weight_ = None
        x_620 = torch.nn.functional.batch_norm(
            x_619,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_619 = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn1_parameters_bias_ = None
        x_621 = torch.nn.functional.relu(x_620, inplace=True)
        x_620 = None
        x_622 = torch.conv2d(
            x_621,
            l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_621 = l_self_modules_layer3_modules_23_modules_conv2_parameters_weight_ = None
        x_623 = torch.nn.functional.batch_norm(
            x_622,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_622 = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn2_parameters_bias_ = None
        x_624 = torch.nn.functional.relu(x_623, inplace=True)
        x_623 = None
        x_625 = torch.conv2d(
            x_624,
            l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_624 = l_self_modules_layer3_modules_23_modules_conv3_parameters_weight_ = None
        x_626 = torch.nn.functional.batch_norm(
            x_625,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_625 = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_23_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_23_modules_bn3_parameters_bias_ = None
        mean_56 = x_626.mean((2, 3))
        y_168 = mean_56.view(1, 1, -1)
        mean_56 = None
        y_169 = torch.conv1d(
            y_168,
            l_self_modules_layer3_modules_23_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_168 = (
            l_self_modules_layer3_modules_23_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_56 = y_169.sigmoid()
        y_169 = None
        y_170 = sigmoid_56.view(1, -1, 1, 1)
        sigmoid_56 = None
        expand_as_56 = y_170.expand_as(x_626)
        y_170 = None
        x_627 = x_626 * expand_as_56
        x_626 = expand_as_56 = None
        x_627 += x_618
        x_628 = x_627
        x_627 = x_618 = None
        x_629 = torch.nn.functional.relu(x_628, inplace=True)
        x_628 = None
        x_630 = torch.conv2d(
            x_629,
            l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_24_modules_conv1_parameters_weight_ = None
        x_631 = torch.nn.functional.batch_norm(
            x_630,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_630 = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn1_parameters_bias_ = None
        x_632 = torch.nn.functional.relu(x_631, inplace=True)
        x_631 = None
        x_633 = torch.conv2d(
            x_632,
            l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_632 = l_self_modules_layer3_modules_24_modules_conv2_parameters_weight_ = None
        x_634 = torch.nn.functional.batch_norm(
            x_633,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_633 = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn2_parameters_bias_ = None
        x_635 = torch.nn.functional.relu(x_634, inplace=True)
        x_634 = None
        x_636 = torch.conv2d(
            x_635,
            l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_635 = l_self_modules_layer3_modules_24_modules_conv3_parameters_weight_ = None
        x_637 = torch.nn.functional.batch_norm(
            x_636,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_636 = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_24_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_24_modules_bn3_parameters_bias_ = None
        mean_57 = x_637.mean((2, 3))
        y_171 = mean_57.view(1, 1, -1)
        mean_57 = None
        y_172 = torch.conv1d(
            y_171,
            l_self_modules_layer3_modules_24_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_171 = (
            l_self_modules_layer3_modules_24_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_57 = y_172.sigmoid()
        y_172 = None
        y_173 = sigmoid_57.view(1, -1, 1, 1)
        sigmoid_57 = None
        expand_as_57 = y_173.expand_as(x_637)
        y_173 = None
        x_638 = x_637 * expand_as_57
        x_637 = expand_as_57 = None
        x_638 += x_629
        x_639 = x_638
        x_638 = x_629 = None
        x_640 = torch.nn.functional.relu(x_639, inplace=True)
        x_639 = None
        x_641 = torch.conv2d(
            x_640,
            l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_25_modules_conv1_parameters_weight_ = None
        x_642 = torch.nn.functional.batch_norm(
            x_641,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_641 = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn1_parameters_bias_ = None
        x_643 = torch.nn.functional.relu(x_642, inplace=True)
        x_642 = None
        x_644 = torch.conv2d(
            x_643,
            l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_643 = l_self_modules_layer3_modules_25_modules_conv2_parameters_weight_ = None
        x_645 = torch.nn.functional.batch_norm(
            x_644,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_644 = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn2_parameters_bias_ = None
        x_646 = torch.nn.functional.relu(x_645, inplace=True)
        x_645 = None
        x_647 = torch.conv2d(
            x_646,
            l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_646 = l_self_modules_layer3_modules_25_modules_conv3_parameters_weight_ = None
        x_648 = torch.nn.functional.batch_norm(
            x_647,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_647 = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_25_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_25_modules_bn3_parameters_bias_ = None
        mean_58 = x_648.mean((2, 3))
        y_174 = mean_58.view(1, 1, -1)
        mean_58 = None
        y_175 = torch.conv1d(
            y_174,
            l_self_modules_layer3_modules_25_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_174 = (
            l_self_modules_layer3_modules_25_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_58 = y_175.sigmoid()
        y_175 = None
        y_176 = sigmoid_58.view(1, -1, 1, 1)
        sigmoid_58 = None
        expand_as_58 = y_176.expand_as(x_648)
        y_176 = None
        x_649 = x_648 * expand_as_58
        x_648 = expand_as_58 = None
        x_649 += x_640
        x_650 = x_649
        x_649 = x_640 = None
        x_651 = torch.nn.functional.relu(x_650, inplace=True)
        x_650 = None
        x_652 = torch.conv2d(
            x_651,
            l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_26_modules_conv1_parameters_weight_ = None
        x_653 = torch.nn.functional.batch_norm(
            x_652,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_652 = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn1_parameters_bias_ = None
        x_654 = torch.nn.functional.relu(x_653, inplace=True)
        x_653 = None
        x_655 = torch.conv2d(
            x_654,
            l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_654 = l_self_modules_layer3_modules_26_modules_conv2_parameters_weight_ = None
        x_656 = torch.nn.functional.batch_norm(
            x_655,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_655 = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn2_parameters_bias_ = None
        x_657 = torch.nn.functional.relu(x_656, inplace=True)
        x_656 = None
        x_658 = torch.conv2d(
            x_657,
            l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_657 = l_self_modules_layer3_modules_26_modules_conv3_parameters_weight_ = None
        x_659 = torch.nn.functional.batch_norm(
            x_658,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_658 = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_26_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_26_modules_bn3_parameters_bias_ = None
        mean_59 = x_659.mean((2, 3))
        y_177 = mean_59.view(1, 1, -1)
        mean_59 = None
        y_178 = torch.conv1d(
            y_177,
            l_self_modules_layer3_modules_26_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_177 = (
            l_self_modules_layer3_modules_26_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_59 = y_178.sigmoid()
        y_178 = None
        y_179 = sigmoid_59.view(1, -1, 1, 1)
        sigmoid_59 = None
        expand_as_59 = y_179.expand_as(x_659)
        y_179 = None
        x_660 = x_659 * expand_as_59
        x_659 = expand_as_59 = None
        x_660 += x_651
        x_661 = x_660
        x_660 = x_651 = None
        x_662 = torch.nn.functional.relu(x_661, inplace=True)
        x_661 = None
        x_663 = torch.conv2d(
            x_662,
            l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_27_modules_conv1_parameters_weight_ = None
        x_664 = torch.nn.functional.batch_norm(
            x_663,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_663 = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn1_parameters_bias_ = None
        x_665 = torch.nn.functional.relu(x_664, inplace=True)
        x_664 = None
        x_666 = torch.conv2d(
            x_665,
            l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_665 = l_self_modules_layer3_modules_27_modules_conv2_parameters_weight_ = None
        x_667 = torch.nn.functional.batch_norm(
            x_666,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_666 = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn2_parameters_bias_ = None
        x_668 = torch.nn.functional.relu(x_667, inplace=True)
        x_667 = None
        x_669 = torch.conv2d(
            x_668,
            l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_668 = l_self_modules_layer3_modules_27_modules_conv3_parameters_weight_ = None
        x_670 = torch.nn.functional.batch_norm(
            x_669,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_669 = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_27_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_27_modules_bn3_parameters_bias_ = None
        mean_60 = x_670.mean((2, 3))
        y_180 = mean_60.view(1, 1, -1)
        mean_60 = None
        y_181 = torch.conv1d(
            y_180,
            l_self_modules_layer3_modules_27_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_180 = (
            l_self_modules_layer3_modules_27_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_60 = y_181.sigmoid()
        y_181 = None
        y_182 = sigmoid_60.view(1, -1, 1, 1)
        sigmoid_60 = None
        expand_as_60 = y_182.expand_as(x_670)
        y_182 = None
        x_671 = x_670 * expand_as_60
        x_670 = expand_as_60 = None
        x_671 += x_662
        x_672 = x_671
        x_671 = x_662 = None
        x_673 = torch.nn.functional.relu(x_672, inplace=True)
        x_672 = None
        x_674 = torch.conv2d(
            x_673,
            l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_28_modules_conv1_parameters_weight_ = None
        x_675 = torch.nn.functional.batch_norm(
            x_674,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_674 = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn1_parameters_bias_ = None
        x_676 = torch.nn.functional.relu(x_675, inplace=True)
        x_675 = None
        x_677 = torch.conv2d(
            x_676,
            l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_676 = l_self_modules_layer3_modules_28_modules_conv2_parameters_weight_ = None
        x_678 = torch.nn.functional.batch_norm(
            x_677,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_677 = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn2_parameters_bias_ = None
        x_679 = torch.nn.functional.relu(x_678, inplace=True)
        x_678 = None
        x_680 = torch.conv2d(
            x_679,
            l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_679 = l_self_modules_layer3_modules_28_modules_conv3_parameters_weight_ = None
        x_681 = torch.nn.functional.batch_norm(
            x_680,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_680 = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_28_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_28_modules_bn3_parameters_bias_ = None
        mean_61 = x_681.mean((2, 3))
        y_183 = mean_61.view(1, 1, -1)
        mean_61 = None
        y_184 = torch.conv1d(
            y_183,
            l_self_modules_layer3_modules_28_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_183 = (
            l_self_modules_layer3_modules_28_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_61 = y_184.sigmoid()
        y_184 = None
        y_185 = sigmoid_61.view(1, -1, 1, 1)
        sigmoid_61 = None
        expand_as_61 = y_185.expand_as(x_681)
        y_185 = None
        x_682 = x_681 * expand_as_61
        x_681 = expand_as_61 = None
        x_682 += x_673
        x_683 = x_682
        x_682 = x_673 = None
        x_684 = torch.nn.functional.relu(x_683, inplace=True)
        x_683 = None
        x_685 = torch.conv2d(
            x_684,
            l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_29_modules_conv1_parameters_weight_ = None
        x_686 = torch.nn.functional.batch_norm(
            x_685,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_685 = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn1_parameters_bias_ = None
        x_687 = torch.nn.functional.relu(x_686, inplace=True)
        x_686 = None
        x_688 = torch.conv2d(
            x_687,
            l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_687 = l_self_modules_layer3_modules_29_modules_conv2_parameters_weight_ = None
        x_689 = torch.nn.functional.batch_norm(
            x_688,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_688 = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn2_parameters_bias_ = None
        x_690 = torch.nn.functional.relu(x_689, inplace=True)
        x_689 = None
        x_691 = torch.conv2d(
            x_690,
            l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_690 = l_self_modules_layer3_modules_29_modules_conv3_parameters_weight_ = None
        x_692 = torch.nn.functional.batch_norm(
            x_691,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_691 = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_29_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_29_modules_bn3_parameters_bias_ = None
        mean_62 = x_692.mean((2, 3))
        y_186 = mean_62.view(1, 1, -1)
        mean_62 = None
        y_187 = torch.conv1d(
            y_186,
            l_self_modules_layer3_modules_29_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_186 = (
            l_self_modules_layer3_modules_29_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_62 = y_187.sigmoid()
        y_187 = None
        y_188 = sigmoid_62.view(1, -1, 1, 1)
        sigmoid_62 = None
        expand_as_62 = y_188.expand_as(x_692)
        y_188 = None
        x_693 = x_692 * expand_as_62
        x_692 = expand_as_62 = None
        x_693 += x_684
        x_694 = x_693
        x_693 = x_684 = None
        x_695 = torch.nn.functional.relu(x_694, inplace=True)
        x_694 = None
        x_696 = torch.conv2d(
            x_695,
            l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_30_modules_conv1_parameters_weight_ = None
        x_697 = torch.nn.functional.batch_norm(
            x_696,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_696 = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn1_parameters_bias_ = None
        x_698 = torch.nn.functional.relu(x_697, inplace=True)
        x_697 = None
        x_699 = torch.conv2d(
            x_698,
            l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_698 = l_self_modules_layer3_modules_30_modules_conv2_parameters_weight_ = None
        x_700 = torch.nn.functional.batch_norm(
            x_699,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_699 = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn2_parameters_bias_ = None
        x_701 = torch.nn.functional.relu(x_700, inplace=True)
        x_700 = None
        x_702 = torch.conv2d(
            x_701,
            l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_701 = l_self_modules_layer3_modules_30_modules_conv3_parameters_weight_ = None
        x_703 = torch.nn.functional.batch_norm(
            x_702,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_702 = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_30_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_30_modules_bn3_parameters_bias_ = None
        mean_63 = x_703.mean((2, 3))
        y_189 = mean_63.view(1, 1, -1)
        mean_63 = None
        y_190 = torch.conv1d(
            y_189,
            l_self_modules_layer3_modules_30_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_189 = (
            l_self_modules_layer3_modules_30_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_63 = y_190.sigmoid()
        y_190 = None
        y_191 = sigmoid_63.view(1, -1, 1, 1)
        sigmoid_63 = None
        expand_as_63 = y_191.expand_as(x_703)
        y_191 = None
        x_704 = x_703 * expand_as_63
        x_703 = expand_as_63 = None
        x_704 += x_695
        x_705 = x_704
        x_704 = x_695 = None
        x_706 = torch.nn.functional.relu(x_705, inplace=True)
        x_705 = None
        x_707 = torch.conv2d(
            x_706,
            l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_31_modules_conv1_parameters_weight_ = None
        x_708 = torch.nn.functional.batch_norm(
            x_707,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_707 = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn1_parameters_bias_ = None
        x_709 = torch.nn.functional.relu(x_708, inplace=True)
        x_708 = None
        x_710 = torch.conv2d(
            x_709,
            l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_709 = l_self_modules_layer3_modules_31_modules_conv2_parameters_weight_ = None
        x_711 = torch.nn.functional.batch_norm(
            x_710,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_710 = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn2_parameters_bias_ = None
        x_712 = torch.nn.functional.relu(x_711, inplace=True)
        x_711 = None
        x_713 = torch.conv2d(
            x_712,
            l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_712 = l_self_modules_layer3_modules_31_modules_conv3_parameters_weight_ = None
        x_714 = torch.nn.functional.batch_norm(
            x_713,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_713 = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_31_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_31_modules_bn3_parameters_bias_ = None
        mean_64 = x_714.mean((2, 3))
        y_192 = mean_64.view(1, 1, -1)
        mean_64 = None
        y_193 = torch.conv1d(
            y_192,
            l_self_modules_layer3_modules_31_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_192 = (
            l_self_modules_layer3_modules_31_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_64 = y_193.sigmoid()
        y_193 = None
        y_194 = sigmoid_64.view(1, -1, 1, 1)
        sigmoid_64 = None
        expand_as_64 = y_194.expand_as(x_714)
        y_194 = None
        x_715 = x_714 * expand_as_64
        x_714 = expand_as_64 = None
        x_715 += x_706
        x_716 = x_715
        x_715 = x_706 = None
        x_717 = torch.nn.functional.relu(x_716, inplace=True)
        x_716 = None
        x_718 = torch.conv2d(
            x_717,
            l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_32_modules_conv1_parameters_weight_ = None
        x_719 = torch.nn.functional.batch_norm(
            x_718,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_718 = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn1_parameters_bias_ = None
        x_720 = torch.nn.functional.relu(x_719, inplace=True)
        x_719 = None
        x_721 = torch.conv2d(
            x_720,
            l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_720 = l_self_modules_layer3_modules_32_modules_conv2_parameters_weight_ = None
        x_722 = torch.nn.functional.batch_norm(
            x_721,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_721 = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn2_parameters_bias_ = None
        x_723 = torch.nn.functional.relu(x_722, inplace=True)
        x_722 = None
        x_724 = torch.conv2d(
            x_723,
            l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_723 = l_self_modules_layer3_modules_32_modules_conv3_parameters_weight_ = None
        x_725 = torch.nn.functional.batch_norm(
            x_724,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_724 = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_32_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_32_modules_bn3_parameters_bias_ = None
        mean_65 = x_725.mean((2, 3))
        y_195 = mean_65.view(1, 1, -1)
        mean_65 = None
        y_196 = torch.conv1d(
            y_195,
            l_self_modules_layer3_modules_32_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_195 = (
            l_self_modules_layer3_modules_32_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_65 = y_196.sigmoid()
        y_196 = None
        y_197 = sigmoid_65.view(1, -1, 1, 1)
        sigmoid_65 = None
        expand_as_65 = y_197.expand_as(x_725)
        y_197 = None
        x_726 = x_725 * expand_as_65
        x_725 = expand_as_65 = None
        x_726 += x_717
        x_727 = x_726
        x_726 = x_717 = None
        x_728 = torch.nn.functional.relu(x_727, inplace=True)
        x_727 = None
        x_729 = torch.conv2d(
            x_728,
            l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_33_modules_conv1_parameters_weight_ = None
        x_730 = torch.nn.functional.batch_norm(
            x_729,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_729 = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn1_parameters_bias_ = None
        x_731 = torch.nn.functional.relu(x_730, inplace=True)
        x_730 = None
        x_732 = torch.conv2d(
            x_731,
            l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_731 = l_self_modules_layer3_modules_33_modules_conv2_parameters_weight_ = None
        x_733 = torch.nn.functional.batch_norm(
            x_732,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_732 = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn2_parameters_bias_ = None
        x_734 = torch.nn.functional.relu(x_733, inplace=True)
        x_733 = None
        x_735 = torch.conv2d(
            x_734,
            l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_734 = l_self_modules_layer3_modules_33_modules_conv3_parameters_weight_ = None
        x_736 = torch.nn.functional.batch_norm(
            x_735,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_735 = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_33_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_33_modules_bn3_parameters_bias_ = None
        mean_66 = x_736.mean((2, 3))
        y_198 = mean_66.view(1, 1, -1)
        mean_66 = None
        y_199 = torch.conv1d(
            y_198,
            l_self_modules_layer3_modules_33_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_198 = (
            l_self_modules_layer3_modules_33_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_66 = y_199.sigmoid()
        y_199 = None
        y_200 = sigmoid_66.view(1, -1, 1, 1)
        sigmoid_66 = None
        expand_as_66 = y_200.expand_as(x_736)
        y_200 = None
        x_737 = x_736 * expand_as_66
        x_736 = expand_as_66 = None
        x_737 += x_728
        x_738 = x_737
        x_737 = x_728 = None
        x_739 = torch.nn.functional.relu(x_738, inplace=True)
        x_738 = None
        x_740 = torch.conv2d(
            x_739,
            l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_34_modules_conv1_parameters_weight_ = None
        x_741 = torch.nn.functional.batch_norm(
            x_740,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_740 = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn1_parameters_bias_ = None
        x_742 = torch.nn.functional.relu(x_741, inplace=True)
        x_741 = None
        x_743 = torch.conv2d(
            x_742,
            l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_742 = l_self_modules_layer3_modules_34_modules_conv2_parameters_weight_ = None
        x_744 = torch.nn.functional.batch_norm(
            x_743,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_743 = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn2_parameters_bias_ = None
        x_745 = torch.nn.functional.relu(x_744, inplace=True)
        x_744 = None
        x_746 = torch.conv2d(
            x_745,
            l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_745 = l_self_modules_layer3_modules_34_modules_conv3_parameters_weight_ = None
        x_747 = torch.nn.functional.batch_norm(
            x_746,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_746 = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_34_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_34_modules_bn3_parameters_bias_ = None
        mean_67 = x_747.mean((2, 3))
        y_201 = mean_67.view(1, 1, -1)
        mean_67 = None
        y_202 = torch.conv1d(
            y_201,
            l_self_modules_layer3_modules_34_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_201 = (
            l_self_modules_layer3_modules_34_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_67 = y_202.sigmoid()
        y_202 = None
        y_203 = sigmoid_67.view(1, -1, 1, 1)
        sigmoid_67 = None
        expand_as_67 = y_203.expand_as(x_747)
        y_203 = None
        x_748 = x_747 * expand_as_67
        x_747 = expand_as_67 = None
        x_748 += x_739
        x_749 = x_748
        x_748 = x_739 = None
        x_750 = torch.nn.functional.relu(x_749, inplace=True)
        x_749 = None
        x_751 = torch.conv2d(
            x_750,
            l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_35_modules_conv1_parameters_weight_ = None
        x_752 = torch.nn.functional.batch_norm(
            x_751,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_751 = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn1_parameters_bias_ = None
        x_753 = torch.nn.functional.relu(x_752, inplace=True)
        x_752 = None
        x_754 = torch.conv2d(
            x_753,
            l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_753 = l_self_modules_layer3_modules_35_modules_conv2_parameters_weight_ = None
        x_755 = torch.nn.functional.batch_norm(
            x_754,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_754 = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn2_parameters_bias_ = None
        x_756 = torch.nn.functional.relu(x_755, inplace=True)
        x_755 = None
        x_757 = torch.conv2d(
            x_756,
            l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_756 = l_self_modules_layer3_modules_35_modules_conv3_parameters_weight_ = None
        x_758 = torch.nn.functional.batch_norm(
            x_757,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_757 = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_35_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_35_modules_bn3_parameters_bias_ = None
        mean_68 = x_758.mean((2, 3))
        y_204 = mean_68.view(1, 1, -1)
        mean_68 = None
        y_205 = torch.conv1d(
            y_204,
            l_self_modules_layer3_modules_35_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_204 = (
            l_self_modules_layer3_modules_35_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_68 = y_205.sigmoid()
        y_205 = None
        y_206 = sigmoid_68.view(1, -1, 1, 1)
        sigmoid_68 = None
        expand_as_68 = y_206.expand_as(x_758)
        y_206 = None
        x_759 = x_758 * expand_as_68
        x_758 = expand_as_68 = None
        x_759 += x_750
        x_760 = x_759
        x_759 = x_750 = None
        x_761 = torch.nn.functional.relu(x_760, inplace=True)
        x_760 = None
        x_762 = torch.conv2d(
            x_761,
            l_self_modules_layer3_modules_36_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_36_modules_conv1_parameters_weight_ = None
        x_763 = torch.nn.functional.batch_norm(
            x_762,
            l_self_modules_layer3_modules_36_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_36_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_36_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_36_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_762 = (
            l_self_modules_layer3_modules_36_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_36_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_36_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_36_modules_bn1_parameters_bias_ = None
        x_764 = torch.nn.functional.relu(x_763, inplace=True)
        x_763 = None
        x_765 = torch.conv2d(
            x_764,
            l_self_modules_layer3_modules_36_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_764 = l_self_modules_layer3_modules_36_modules_conv2_parameters_weight_ = None
        x_766 = torch.nn.functional.batch_norm(
            x_765,
            l_self_modules_layer3_modules_36_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_36_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_36_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_36_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_765 = (
            l_self_modules_layer3_modules_36_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_36_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_36_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_36_modules_bn2_parameters_bias_ = None
        x_767 = torch.nn.functional.relu(x_766, inplace=True)
        x_766 = None
        x_768 = torch.conv2d(
            x_767,
            l_self_modules_layer3_modules_36_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_767 = l_self_modules_layer3_modules_36_modules_conv3_parameters_weight_ = None
        x_769 = torch.nn.functional.batch_norm(
            x_768,
            l_self_modules_layer3_modules_36_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_36_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_36_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_36_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_768 = (
            l_self_modules_layer3_modules_36_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_36_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_36_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_36_modules_bn3_parameters_bias_ = None
        mean_69 = x_769.mean((2, 3))
        y_207 = mean_69.view(1, 1, -1)
        mean_69 = None
        y_208 = torch.conv1d(
            y_207,
            l_self_modules_layer3_modules_36_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_207 = (
            l_self_modules_layer3_modules_36_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_69 = y_208.sigmoid()
        y_208 = None
        y_209 = sigmoid_69.view(1, -1, 1, 1)
        sigmoid_69 = None
        expand_as_69 = y_209.expand_as(x_769)
        y_209 = None
        x_770 = x_769 * expand_as_69
        x_769 = expand_as_69 = None
        x_770 += x_761
        x_771 = x_770
        x_770 = x_761 = None
        x_772 = torch.nn.functional.relu(x_771, inplace=True)
        x_771 = None
        x_773 = torch.conv2d(
            x_772,
            l_self_modules_layer3_modules_37_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_37_modules_conv1_parameters_weight_ = None
        x_774 = torch.nn.functional.batch_norm(
            x_773,
            l_self_modules_layer3_modules_37_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_37_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_37_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_37_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_773 = (
            l_self_modules_layer3_modules_37_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_37_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_37_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_37_modules_bn1_parameters_bias_ = None
        x_775 = torch.nn.functional.relu(x_774, inplace=True)
        x_774 = None
        x_776 = torch.conv2d(
            x_775,
            l_self_modules_layer3_modules_37_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_775 = l_self_modules_layer3_modules_37_modules_conv2_parameters_weight_ = None
        x_777 = torch.nn.functional.batch_norm(
            x_776,
            l_self_modules_layer3_modules_37_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_37_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_37_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_37_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_776 = (
            l_self_modules_layer3_modules_37_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_37_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_37_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_37_modules_bn2_parameters_bias_ = None
        x_778 = torch.nn.functional.relu(x_777, inplace=True)
        x_777 = None
        x_779 = torch.conv2d(
            x_778,
            l_self_modules_layer3_modules_37_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_778 = l_self_modules_layer3_modules_37_modules_conv3_parameters_weight_ = None
        x_780 = torch.nn.functional.batch_norm(
            x_779,
            l_self_modules_layer3_modules_37_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_37_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_37_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_37_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_779 = (
            l_self_modules_layer3_modules_37_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_37_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_37_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_37_modules_bn3_parameters_bias_ = None
        mean_70 = x_780.mean((2, 3))
        y_210 = mean_70.view(1, 1, -1)
        mean_70 = None
        y_211 = torch.conv1d(
            y_210,
            l_self_modules_layer3_modules_37_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_210 = (
            l_self_modules_layer3_modules_37_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_70 = y_211.sigmoid()
        y_211 = None
        y_212 = sigmoid_70.view(1, -1, 1, 1)
        sigmoid_70 = None
        expand_as_70 = y_212.expand_as(x_780)
        y_212 = None
        x_781 = x_780 * expand_as_70
        x_780 = expand_as_70 = None
        x_781 += x_772
        x_782 = x_781
        x_781 = x_772 = None
        x_783 = torch.nn.functional.relu(x_782, inplace=True)
        x_782 = None
        x_784 = torch.conv2d(
            x_783,
            l_self_modules_layer3_modules_38_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_38_modules_conv1_parameters_weight_ = None
        x_785 = torch.nn.functional.batch_norm(
            x_784,
            l_self_modules_layer3_modules_38_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_38_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_38_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_38_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_784 = (
            l_self_modules_layer3_modules_38_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_38_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_38_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_38_modules_bn1_parameters_bias_ = None
        x_786 = torch.nn.functional.relu(x_785, inplace=True)
        x_785 = None
        x_787 = torch.conv2d(
            x_786,
            l_self_modules_layer3_modules_38_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_786 = l_self_modules_layer3_modules_38_modules_conv2_parameters_weight_ = None
        x_788 = torch.nn.functional.batch_norm(
            x_787,
            l_self_modules_layer3_modules_38_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_38_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_38_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_38_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_787 = (
            l_self_modules_layer3_modules_38_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_38_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_38_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_38_modules_bn2_parameters_bias_ = None
        x_789 = torch.nn.functional.relu(x_788, inplace=True)
        x_788 = None
        x_790 = torch.conv2d(
            x_789,
            l_self_modules_layer3_modules_38_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_789 = l_self_modules_layer3_modules_38_modules_conv3_parameters_weight_ = None
        x_791 = torch.nn.functional.batch_norm(
            x_790,
            l_self_modules_layer3_modules_38_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_38_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_38_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_38_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_790 = (
            l_self_modules_layer3_modules_38_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_38_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_38_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_38_modules_bn3_parameters_bias_ = None
        mean_71 = x_791.mean((2, 3))
        y_213 = mean_71.view(1, 1, -1)
        mean_71 = None
        y_214 = torch.conv1d(
            y_213,
            l_self_modules_layer3_modules_38_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_213 = (
            l_self_modules_layer3_modules_38_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_71 = y_214.sigmoid()
        y_214 = None
        y_215 = sigmoid_71.view(1, -1, 1, 1)
        sigmoid_71 = None
        expand_as_71 = y_215.expand_as(x_791)
        y_215 = None
        x_792 = x_791 * expand_as_71
        x_791 = expand_as_71 = None
        x_792 += x_783
        x_793 = x_792
        x_792 = x_783 = None
        x_794 = torch.nn.functional.relu(x_793, inplace=True)
        x_793 = None
        x_795 = torch.conv2d(
            x_794,
            l_self_modules_layer3_modules_39_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_39_modules_conv1_parameters_weight_ = None
        x_796 = torch.nn.functional.batch_norm(
            x_795,
            l_self_modules_layer3_modules_39_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_39_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_39_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_39_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_795 = (
            l_self_modules_layer3_modules_39_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_39_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_39_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_39_modules_bn1_parameters_bias_ = None
        x_797 = torch.nn.functional.relu(x_796, inplace=True)
        x_796 = None
        x_798 = torch.conv2d(
            x_797,
            l_self_modules_layer3_modules_39_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_797 = l_self_modules_layer3_modules_39_modules_conv2_parameters_weight_ = None
        x_799 = torch.nn.functional.batch_norm(
            x_798,
            l_self_modules_layer3_modules_39_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_39_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_39_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_39_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_798 = (
            l_self_modules_layer3_modules_39_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_39_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_39_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_39_modules_bn2_parameters_bias_ = None
        x_800 = torch.nn.functional.relu(x_799, inplace=True)
        x_799 = None
        x_801 = torch.conv2d(
            x_800,
            l_self_modules_layer3_modules_39_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_800 = l_self_modules_layer3_modules_39_modules_conv3_parameters_weight_ = None
        x_802 = torch.nn.functional.batch_norm(
            x_801,
            l_self_modules_layer3_modules_39_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_39_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_39_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_39_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_801 = (
            l_self_modules_layer3_modules_39_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_39_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_39_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_39_modules_bn3_parameters_bias_ = None
        mean_72 = x_802.mean((2, 3))
        y_216 = mean_72.view(1, 1, -1)
        mean_72 = None
        y_217 = torch.conv1d(
            y_216,
            l_self_modules_layer3_modules_39_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_216 = (
            l_self_modules_layer3_modules_39_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_72 = y_217.sigmoid()
        y_217 = None
        y_218 = sigmoid_72.view(1, -1, 1, 1)
        sigmoid_72 = None
        expand_as_72 = y_218.expand_as(x_802)
        y_218 = None
        x_803 = x_802 * expand_as_72
        x_802 = expand_as_72 = None
        x_803 += x_794
        x_804 = x_803
        x_803 = x_794 = None
        x_805 = torch.nn.functional.relu(x_804, inplace=True)
        x_804 = None
        x_806 = torch.conv2d(
            x_805,
            l_self_modules_layer3_modules_40_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_40_modules_conv1_parameters_weight_ = None
        x_807 = torch.nn.functional.batch_norm(
            x_806,
            l_self_modules_layer3_modules_40_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_40_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_40_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_40_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_806 = (
            l_self_modules_layer3_modules_40_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_40_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_40_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_40_modules_bn1_parameters_bias_ = None
        x_808 = torch.nn.functional.relu(x_807, inplace=True)
        x_807 = None
        x_809 = torch.conv2d(
            x_808,
            l_self_modules_layer3_modules_40_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_808 = l_self_modules_layer3_modules_40_modules_conv2_parameters_weight_ = None
        x_810 = torch.nn.functional.batch_norm(
            x_809,
            l_self_modules_layer3_modules_40_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_40_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_40_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_40_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_809 = (
            l_self_modules_layer3_modules_40_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_40_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_40_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_40_modules_bn2_parameters_bias_ = None
        x_811 = torch.nn.functional.relu(x_810, inplace=True)
        x_810 = None
        x_812 = torch.conv2d(
            x_811,
            l_self_modules_layer3_modules_40_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_811 = l_self_modules_layer3_modules_40_modules_conv3_parameters_weight_ = None
        x_813 = torch.nn.functional.batch_norm(
            x_812,
            l_self_modules_layer3_modules_40_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_40_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_40_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_40_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_812 = (
            l_self_modules_layer3_modules_40_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_40_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_40_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_40_modules_bn3_parameters_bias_ = None
        mean_73 = x_813.mean((2, 3))
        y_219 = mean_73.view(1, 1, -1)
        mean_73 = None
        y_220 = torch.conv1d(
            y_219,
            l_self_modules_layer3_modules_40_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_219 = (
            l_self_modules_layer3_modules_40_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_73 = y_220.sigmoid()
        y_220 = None
        y_221 = sigmoid_73.view(1, -1, 1, 1)
        sigmoid_73 = None
        expand_as_73 = y_221.expand_as(x_813)
        y_221 = None
        x_814 = x_813 * expand_as_73
        x_813 = expand_as_73 = None
        x_814 += x_805
        x_815 = x_814
        x_814 = x_805 = None
        x_816 = torch.nn.functional.relu(x_815, inplace=True)
        x_815 = None
        x_817 = torch.conv2d(
            x_816,
            l_self_modules_layer3_modules_41_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_41_modules_conv1_parameters_weight_ = None
        x_818 = torch.nn.functional.batch_norm(
            x_817,
            l_self_modules_layer3_modules_41_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_41_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_41_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_41_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_817 = (
            l_self_modules_layer3_modules_41_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_41_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_41_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_41_modules_bn1_parameters_bias_ = None
        x_819 = torch.nn.functional.relu(x_818, inplace=True)
        x_818 = None
        x_820 = torch.conv2d(
            x_819,
            l_self_modules_layer3_modules_41_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_819 = l_self_modules_layer3_modules_41_modules_conv2_parameters_weight_ = None
        x_821 = torch.nn.functional.batch_norm(
            x_820,
            l_self_modules_layer3_modules_41_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_41_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_41_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_41_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_820 = (
            l_self_modules_layer3_modules_41_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_41_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_41_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_41_modules_bn2_parameters_bias_ = None
        x_822 = torch.nn.functional.relu(x_821, inplace=True)
        x_821 = None
        x_823 = torch.conv2d(
            x_822,
            l_self_modules_layer3_modules_41_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_822 = l_self_modules_layer3_modules_41_modules_conv3_parameters_weight_ = None
        x_824 = torch.nn.functional.batch_norm(
            x_823,
            l_self_modules_layer3_modules_41_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_41_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_41_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_41_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_823 = (
            l_self_modules_layer3_modules_41_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_41_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_41_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_41_modules_bn3_parameters_bias_ = None
        mean_74 = x_824.mean((2, 3))
        y_222 = mean_74.view(1, 1, -1)
        mean_74 = None
        y_223 = torch.conv1d(
            y_222,
            l_self_modules_layer3_modules_41_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_222 = (
            l_self_modules_layer3_modules_41_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_74 = y_223.sigmoid()
        y_223 = None
        y_224 = sigmoid_74.view(1, -1, 1, 1)
        sigmoid_74 = None
        expand_as_74 = y_224.expand_as(x_824)
        y_224 = None
        x_825 = x_824 * expand_as_74
        x_824 = expand_as_74 = None
        x_825 += x_816
        x_826 = x_825
        x_825 = x_816 = None
        x_827 = torch.nn.functional.relu(x_826, inplace=True)
        x_826 = None
        x_828 = torch.conv2d(
            x_827,
            l_self_modules_layer3_modules_42_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_42_modules_conv1_parameters_weight_ = None
        x_829 = torch.nn.functional.batch_norm(
            x_828,
            l_self_modules_layer3_modules_42_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_42_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_42_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_42_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_828 = (
            l_self_modules_layer3_modules_42_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_42_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_42_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_42_modules_bn1_parameters_bias_ = None
        x_830 = torch.nn.functional.relu(x_829, inplace=True)
        x_829 = None
        x_831 = torch.conv2d(
            x_830,
            l_self_modules_layer3_modules_42_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_830 = l_self_modules_layer3_modules_42_modules_conv2_parameters_weight_ = None
        x_832 = torch.nn.functional.batch_norm(
            x_831,
            l_self_modules_layer3_modules_42_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_42_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_42_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_42_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_831 = (
            l_self_modules_layer3_modules_42_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_42_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_42_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_42_modules_bn2_parameters_bias_ = None
        x_833 = torch.nn.functional.relu(x_832, inplace=True)
        x_832 = None
        x_834 = torch.conv2d(
            x_833,
            l_self_modules_layer3_modules_42_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_833 = l_self_modules_layer3_modules_42_modules_conv3_parameters_weight_ = None
        x_835 = torch.nn.functional.batch_norm(
            x_834,
            l_self_modules_layer3_modules_42_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_42_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_42_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_42_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_834 = (
            l_self_modules_layer3_modules_42_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_42_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_42_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_42_modules_bn3_parameters_bias_ = None
        mean_75 = x_835.mean((2, 3))
        y_225 = mean_75.view(1, 1, -1)
        mean_75 = None
        y_226 = torch.conv1d(
            y_225,
            l_self_modules_layer3_modules_42_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_225 = (
            l_self_modules_layer3_modules_42_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_75 = y_226.sigmoid()
        y_226 = None
        y_227 = sigmoid_75.view(1, -1, 1, 1)
        sigmoid_75 = None
        expand_as_75 = y_227.expand_as(x_835)
        y_227 = None
        x_836 = x_835 * expand_as_75
        x_835 = expand_as_75 = None
        x_836 += x_827
        x_837 = x_836
        x_836 = x_827 = None
        x_838 = torch.nn.functional.relu(x_837, inplace=True)
        x_837 = None
        x_839 = torch.conv2d(
            x_838,
            l_self_modules_layer3_modules_43_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_43_modules_conv1_parameters_weight_ = None
        x_840 = torch.nn.functional.batch_norm(
            x_839,
            l_self_modules_layer3_modules_43_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_43_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_43_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_43_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_839 = (
            l_self_modules_layer3_modules_43_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_43_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_43_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_43_modules_bn1_parameters_bias_ = None
        x_841 = torch.nn.functional.relu(x_840, inplace=True)
        x_840 = None
        x_842 = torch.conv2d(
            x_841,
            l_self_modules_layer3_modules_43_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_841 = l_self_modules_layer3_modules_43_modules_conv2_parameters_weight_ = None
        x_843 = torch.nn.functional.batch_norm(
            x_842,
            l_self_modules_layer3_modules_43_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_43_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_43_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_43_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_842 = (
            l_self_modules_layer3_modules_43_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_43_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_43_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_43_modules_bn2_parameters_bias_ = None
        x_844 = torch.nn.functional.relu(x_843, inplace=True)
        x_843 = None
        x_845 = torch.conv2d(
            x_844,
            l_self_modules_layer3_modules_43_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_844 = l_self_modules_layer3_modules_43_modules_conv3_parameters_weight_ = None
        x_846 = torch.nn.functional.batch_norm(
            x_845,
            l_self_modules_layer3_modules_43_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_43_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_43_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_43_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_845 = (
            l_self_modules_layer3_modules_43_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_43_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_43_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_43_modules_bn3_parameters_bias_ = None
        mean_76 = x_846.mean((2, 3))
        y_228 = mean_76.view(1, 1, -1)
        mean_76 = None
        y_229 = torch.conv1d(
            y_228,
            l_self_modules_layer3_modules_43_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_228 = (
            l_self_modules_layer3_modules_43_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_76 = y_229.sigmoid()
        y_229 = None
        y_230 = sigmoid_76.view(1, -1, 1, 1)
        sigmoid_76 = None
        expand_as_76 = y_230.expand_as(x_846)
        y_230 = None
        x_847 = x_846 * expand_as_76
        x_846 = expand_as_76 = None
        x_847 += x_838
        x_848 = x_847
        x_847 = x_838 = None
        x_849 = torch.nn.functional.relu(x_848, inplace=True)
        x_848 = None
        x_850 = torch.conv2d(
            x_849,
            l_self_modules_layer3_modules_44_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_44_modules_conv1_parameters_weight_ = None
        x_851 = torch.nn.functional.batch_norm(
            x_850,
            l_self_modules_layer3_modules_44_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_44_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_44_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_44_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_850 = (
            l_self_modules_layer3_modules_44_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_44_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_44_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_44_modules_bn1_parameters_bias_ = None
        x_852 = torch.nn.functional.relu(x_851, inplace=True)
        x_851 = None
        x_853 = torch.conv2d(
            x_852,
            l_self_modules_layer3_modules_44_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_852 = l_self_modules_layer3_modules_44_modules_conv2_parameters_weight_ = None
        x_854 = torch.nn.functional.batch_norm(
            x_853,
            l_self_modules_layer3_modules_44_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_44_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_44_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_44_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_853 = (
            l_self_modules_layer3_modules_44_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_44_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_44_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_44_modules_bn2_parameters_bias_ = None
        x_855 = torch.nn.functional.relu(x_854, inplace=True)
        x_854 = None
        x_856 = torch.conv2d(
            x_855,
            l_self_modules_layer3_modules_44_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_855 = l_self_modules_layer3_modules_44_modules_conv3_parameters_weight_ = None
        x_857 = torch.nn.functional.batch_norm(
            x_856,
            l_self_modules_layer3_modules_44_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_44_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_44_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_44_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_856 = (
            l_self_modules_layer3_modules_44_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_44_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_44_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_44_modules_bn3_parameters_bias_ = None
        mean_77 = x_857.mean((2, 3))
        y_231 = mean_77.view(1, 1, -1)
        mean_77 = None
        y_232 = torch.conv1d(
            y_231,
            l_self_modules_layer3_modules_44_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_231 = (
            l_self_modules_layer3_modules_44_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_77 = y_232.sigmoid()
        y_232 = None
        y_233 = sigmoid_77.view(1, -1, 1, 1)
        sigmoid_77 = None
        expand_as_77 = y_233.expand_as(x_857)
        y_233 = None
        x_858 = x_857 * expand_as_77
        x_857 = expand_as_77 = None
        x_858 += x_849
        x_859 = x_858
        x_858 = x_849 = None
        x_860 = torch.nn.functional.relu(x_859, inplace=True)
        x_859 = None
        x_861 = torch.conv2d(
            x_860,
            l_self_modules_layer3_modules_45_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_45_modules_conv1_parameters_weight_ = None
        x_862 = torch.nn.functional.batch_norm(
            x_861,
            l_self_modules_layer3_modules_45_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_45_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_45_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_45_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_861 = (
            l_self_modules_layer3_modules_45_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_45_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_45_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_45_modules_bn1_parameters_bias_ = None
        x_863 = torch.nn.functional.relu(x_862, inplace=True)
        x_862 = None
        x_864 = torch.conv2d(
            x_863,
            l_self_modules_layer3_modules_45_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_863 = l_self_modules_layer3_modules_45_modules_conv2_parameters_weight_ = None
        x_865 = torch.nn.functional.batch_norm(
            x_864,
            l_self_modules_layer3_modules_45_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_45_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_45_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_45_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_864 = (
            l_self_modules_layer3_modules_45_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_45_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_45_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_45_modules_bn2_parameters_bias_ = None
        x_866 = torch.nn.functional.relu(x_865, inplace=True)
        x_865 = None
        x_867 = torch.conv2d(
            x_866,
            l_self_modules_layer3_modules_45_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_866 = l_self_modules_layer3_modules_45_modules_conv3_parameters_weight_ = None
        x_868 = torch.nn.functional.batch_norm(
            x_867,
            l_self_modules_layer3_modules_45_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_45_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_45_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_45_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_867 = (
            l_self_modules_layer3_modules_45_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_45_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_45_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_45_modules_bn3_parameters_bias_ = None
        mean_78 = x_868.mean((2, 3))
        y_234 = mean_78.view(1, 1, -1)
        mean_78 = None
        y_235 = torch.conv1d(
            y_234,
            l_self_modules_layer3_modules_45_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_234 = (
            l_self_modules_layer3_modules_45_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_78 = y_235.sigmoid()
        y_235 = None
        y_236 = sigmoid_78.view(1, -1, 1, 1)
        sigmoid_78 = None
        expand_as_78 = y_236.expand_as(x_868)
        y_236 = None
        x_869 = x_868 * expand_as_78
        x_868 = expand_as_78 = None
        x_869 += x_860
        x_870 = x_869
        x_869 = x_860 = None
        x_871 = torch.nn.functional.relu(x_870, inplace=True)
        x_870 = None
        x_872 = torch.conv2d(
            x_871,
            l_self_modules_layer3_modules_46_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_46_modules_conv1_parameters_weight_ = None
        x_873 = torch.nn.functional.batch_norm(
            x_872,
            l_self_modules_layer3_modules_46_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_46_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_46_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_46_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_872 = (
            l_self_modules_layer3_modules_46_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_46_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_46_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_46_modules_bn1_parameters_bias_ = None
        x_874 = torch.nn.functional.relu(x_873, inplace=True)
        x_873 = None
        x_875 = torch.conv2d(
            x_874,
            l_self_modules_layer3_modules_46_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_874 = l_self_modules_layer3_modules_46_modules_conv2_parameters_weight_ = None
        x_876 = torch.nn.functional.batch_norm(
            x_875,
            l_self_modules_layer3_modules_46_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_46_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_46_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_46_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_875 = (
            l_self_modules_layer3_modules_46_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_46_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_46_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_46_modules_bn2_parameters_bias_ = None
        x_877 = torch.nn.functional.relu(x_876, inplace=True)
        x_876 = None
        x_878 = torch.conv2d(
            x_877,
            l_self_modules_layer3_modules_46_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_877 = l_self_modules_layer3_modules_46_modules_conv3_parameters_weight_ = None
        x_879 = torch.nn.functional.batch_norm(
            x_878,
            l_self_modules_layer3_modules_46_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_46_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_46_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_46_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_878 = (
            l_self_modules_layer3_modules_46_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_46_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_46_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_46_modules_bn3_parameters_bias_ = None
        mean_79 = x_879.mean((2, 3))
        y_237 = mean_79.view(1, 1, -1)
        mean_79 = None
        y_238 = torch.conv1d(
            y_237,
            l_self_modules_layer3_modules_46_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_237 = (
            l_self_modules_layer3_modules_46_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_79 = y_238.sigmoid()
        y_238 = None
        y_239 = sigmoid_79.view(1, -1, 1, 1)
        sigmoid_79 = None
        expand_as_79 = y_239.expand_as(x_879)
        y_239 = None
        x_880 = x_879 * expand_as_79
        x_879 = expand_as_79 = None
        x_880 += x_871
        x_881 = x_880
        x_880 = x_871 = None
        x_882 = torch.nn.functional.relu(x_881, inplace=True)
        x_881 = None
        x_883 = torch.conv2d(
            x_882,
            l_self_modules_layer3_modules_47_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_47_modules_conv1_parameters_weight_ = None
        x_884 = torch.nn.functional.batch_norm(
            x_883,
            l_self_modules_layer3_modules_47_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_47_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_47_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_47_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_883 = (
            l_self_modules_layer3_modules_47_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_47_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_47_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_47_modules_bn1_parameters_bias_ = None
        x_885 = torch.nn.functional.relu(x_884, inplace=True)
        x_884 = None
        x_886 = torch.conv2d(
            x_885,
            l_self_modules_layer3_modules_47_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_885 = l_self_modules_layer3_modules_47_modules_conv2_parameters_weight_ = None
        x_887 = torch.nn.functional.batch_norm(
            x_886,
            l_self_modules_layer3_modules_47_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_47_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_47_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_47_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_886 = (
            l_self_modules_layer3_modules_47_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_47_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_47_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_47_modules_bn2_parameters_bias_ = None
        x_888 = torch.nn.functional.relu(x_887, inplace=True)
        x_887 = None
        x_889 = torch.conv2d(
            x_888,
            l_self_modules_layer3_modules_47_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_888 = l_self_modules_layer3_modules_47_modules_conv3_parameters_weight_ = None
        x_890 = torch.nn.functional.batch_norm(
            x_889,
            l_self_modules_layer3_modules_47_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_47_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_47_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_47_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_889 = (
            l_self_modules_layer3_modules_47_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_47_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_47_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_47_modules_bn3_parameters_bias_ = None
        mean_80 = x_890.mean((2, 3))
        y_240 = mean_80.view(1, 1, -1)
        mean_80 = None
        y_241 = torch.conv1d(
            y_240,
            l_self_modules_layer3_modules_47_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_240 = (
            l_self_modules_layer3_modules_47_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_80 = y_241.sigmoid()
        y_241 = None
        y_242 = sigmoid_80.view(1, -1, 1, 1)
        sigmoid_80 = None
        expand_as_80 = y_242.expand_as(x_890)
        y_242 = None
        x_891 = x_890 * expand_as_80
        x_890 = expand_as_80 = None
        x_891 += x_882
        x_892 = x_891
        x_891 = x_882 = None
        x_893 = torch.nn.functional.relu(x_892, inplace=True)
        x_892 = None
        x_894 = torch.conv2d(
            x_893,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_895 = torch.nn.functional.batch_norm(
            x_894,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_894 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_896 = torch.nn.functional.relu(x_895, inplace=True)
        x_895 = None
        x_897 = torch.conv2d(
            x_896,
            l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_896 = l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_898 = torch.nn.functional.batch_norm(
            x_897,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_897 = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        x_899 = torch.nn.functional.relu(x_898, inplace=True)
        x_898 = None
        x_900 = torch.conv2d(
            x_899,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_899 = l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = None
        x_901 = torch.nn.functional.batch_norm(
            x_900,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_900 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        mean_81 = x_901.mean((2, 3))
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        y_243 = mean_81.view(1, 1, -1)
        mean_81 = None
        y_244 = torch.conv1d(
            y_243,
            l_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_243 = (
            l_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_81 = y_244.sigmoid()
        y_244 = None
        y_245 = sigmoid_81.view(1, -1, 1, 1)
        sigmoid_81 = None
        expand_as_81 = y_245.expand_as(x_901)
        y_245 = None
        x_902 = x_901 * expand_as_81
        x_901 = expand_as_81 = None
        input_16 = torch._C._nn.avg_pool2d(x_893, 2, 2, 0, True, False, None)
        x_893 = None
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
        x_902 += input_18
        x_903 = x_902
        x_902 = input_18 = None
        x_904 = torch.nn.functional.relu(x_903, inplace=True)
        x_903 = None
        x_905 = torch.conv2d(
            x_904,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_906 = torch.nn.functional.batch_norm(
            x_905,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_905 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_907 = torch.nn.functional.relu(x_906, inplace=True)
        x_906 = None
        x_908 = torch.conv2d(
            x_907,
            l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_907 = l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_909 = torch.nn.functional.batch_norm(
            x_908,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_908 = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_910 = torch.nn.functional.relu(x_909, inplace=True)
        x_909 = None
        x_911 = torch.conv2d(
            x_910,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_910 = l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = None
        x_912 = torch.nn.functional.batch_norm(
            x_911,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_911 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        mean_82 = x_912.mean((2, 3))
        y_246 = mean_82.view(1, 1, -1)
        mean_82 = None
        y_247 = torch.conv1d(
            y_246,
            l_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_246 = (
            l_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_82 = y_247.sigmoid()
        y_247 = None
        y_248 = sigmoid_82.view(1, -1, 1, 1)
        sigmoid_82 = None
        expand_as_82 = y_248.expand_as(x_912)
        y_248 = None
        x_913 = x_912 * expand_as_82
        x_912 = expand_as_82 = None
        x_913 += x_904
        x_914 = x_913
        x_913 = x_904 = None
        x_915 = torch.nn.functional.relu(x_914, inplace=True)
        x_914 = None
        x_916 = torch.conv2d(
            x_915,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        x_917 = torch.nn.functional.batch_norm(
            x_916,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_916 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        x_918 = torch.nn.functional.relu(x_917, inplace=True)
        x_917 = None
        x_919 = torch.conv2d(
            x_918,
            l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_918 = l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_ = None
        x_920 = torch.nn.functional.batch_norm(
            x_919,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_919 = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_ = None
        x_921 = torch.nn.functional.relu(x_920, inplace=True)
        x_920 = None
        x_922 = torch.conv2d(
            x_921,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_921 = l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = None
        x_923 = torch.nn.functional.batch_norm(
            x_922,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_922 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        mean_83 = x_923.mean((2, 3))
        y_249 = mean_83.view(1, 1, -1)
        mean_83 = None
        y_250 = torch.conv1d(
            y_249,
            l_self_modules_layer4_modules_2_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_249 = (
            l_self_modules_layer4_modules_2_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_83 = y_250.sigmoid()
        y_250 = None
        y_251 = sigmoid_83.view(1, -1, 1, 1)
        sigmoid_83 = None
        expand_as_83 = y_251.expand_as(x_923)
        y_251 = None
        x_924 = x_923 * expand_as_83
        x_923 = expand_as_83 = None
        x_924 += x_915
        x_925 = x_924
        x_924 = x_915 = None
        x_926 = torch.nn.functional.relu(x_925, inplace=True)
        x_925 = None
        x_927 = torch.conv2d(
            x_926,
            l_self_modules_layer4_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_3_modules_conv1_parameters_weight_ = None
        x_928 = torch.nn.functional.batch_norm(
            x_927,
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_927 = (
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_3_modules_bn1_parameters_bias_ = None
        x_929 = torch.nn.functional.relu(x_928, inplace=True)
        x_928 = None
        x_930 = torch.conv2d(
            x_929,
            l_self_modules_layer4_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_929 = l_self_modules_layer4_modules_3_modules_conv2_parameters_weight_ = None
        x_931 = torch.nn.functional.batch_norm(
            x_930,
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_930 = (
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_3_modules_bn2_parameters_bias_ = None
        x_932 = torch.nn.functional.relu(x_931, inplace=True)
        x_931 = None
        x_933 = torch.conv2d(
            x_932,
            l_self_modules_layer4_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_932 = l_self_modules_layer4_modules_3_modules_conv3_parameters_weight_ = None
        x_934 = torch.nn.functional.batch_norm(
            x_933,
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_933 = (
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_3_modules_bn3_parameters_bias_ = None
        mean_84 = x_934.mean((2, 3))
        y_252 = mean_84.view(1, 1, -1)
        mean_84 = None
        y_253 = torch.conv1d(
            y_252,
            l_self_modules_layer4_modules_3_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_252 = (
            l_self_modules_layer4_modules_3_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_84 = y_253.sigmoid()
        y_253 = None
        y_254 = sigmoid_84.view(1, -1, 1, 1)
        sigmoid_84 = None
        expand_as_84 = y_254.expand_as(x_934)
        y_254 = None
        x_935 = x_934 * expand_as_84
        x_934 = expand_as_84 = None
        x_935 += x_926
        x_936 = x_935
        x_935 = x_926 = None
        x_937 = torch.nn.functional.relu(x_936, inplace=True)
        x_936 = None
        x_938 = torch.conv2d(
            x_937,
            l_self_modules_layer4_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_4_modules_conv1_parameters_weight_ = None
        x_939 = torch.nn.functional.batch_norm(
            x_938,
            l_self_modules_layer4_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_938 = (
            l_self_modules_layer4_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_4_modules_bn1_parameters_bias_ = None
        x_940 = torch.nn.functional.relu(x_939, inplace=True)
        x_939 = None
        x_941 = torch.conv2d(
            x_940,
            l_self_modules_layer4_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_940 = l_self_modules_layer4_modules_4_modules_conv2_parameters_weight_ = None
        x_942 = torch.nn.functional.batch_norm(
            x_941,
            l_self_modules_layer4_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_941 = (
            l_self_modules_layer4_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_4_modules_bn2_parameters_bias_ = None
        x_943 = torch.nn.functional.relu(x_942, inplace=True)
        x_942 = None
        x_944 = torch.conv2d(
            x_943,
            l_self_modules_layer4_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_943 = l_self_modules_layer4_modules_4_modules_conv3_parameters_weight_ = None
        x_945 = torch.nn.functional.batch_norm(
            x_944,
            l_self_modules_layer4_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_944 = (
            l_self_modules_layer4_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_4_modules_bn3_parameters_bias_ = None
        mean_85 = x_945.mean((2, 3))
        y_255 = mean_85.view(1, 1, -1)
        mean_85 = None
        y_256 = torch.conv1d(
            y_255,
            l_self_modules_layer4_modules_4_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_255 = (
            l_self_modules_layer4_modules_4_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_85 = y_256.sigmoid()
        y_256 = None
        y_257 = sigmoid_85.view(1, -1, 1, 1)
        sigmoid_85 = None
        expand_as_85 = y_257.expand_as(x_945)
        y_257 = None
        x_946 = x_945 * expand_as_85
        x_945 = expand_as_85 = None
        x_946 += x_937
        x_947 = x_946
        x_946 = x_937 = None
        x_948 = torch.nn.functional.relu(x_947, inplace=True)
        x_947 = None
        x_949 = torch.conv2d(
            x_948,
            l_self_modules_layer4_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_5_modules_conv1_parameters_weight_ = None
        x_950 = torch.nn.functional.batch_norm(
            x_949,
            l_self_modules_layer4_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_949 = (
            l_self_modules_layer4_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_5_modules_bn1_parameters_bias_ = None
        x_951 = torch.nn.functional.relu(x_950, inplace=True)
        x_950 = None
        x_952 = torch.conv2d(
            x_951,
            l_self_modules_layer4_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_951 = l_self_modules_layer4_modules_5_modules_conv2_parameters_weight_ = None
        x_953 = torch.nn.functional.batch_norm(
            x_952,
            l_self_modules_layer4_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_952 = (
            l_self_modules_layer4_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_5_modules_bn2_parameters_bias_ = None
        x_954 = torch.nn.functional.relu(x_953, inplace=True)
        x_953 = None
        x_955 = torch.conv2d(
            x_954,
            l_self_modules_layer4_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_954 = l_self_modules_layer4_modules_5_modules_conv3_parameters_weight_ = None
        x_956 = torch.nn.functional.batch_norm(
            x_955,
            l_self_modules_layer4_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_955 = (
            l_self_modules_layer4_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_5_modules_bn3_parameters_bias_ = None
        mean_86 = x_956.mean((2, 3))
        y_258 = mean_86.view(1, 1, -1)
        mean_86 = None
        y_259 = torch.conv1d(
            y_258,
            l_self_modules_layer4_modules_5_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_258 = (
            l_self_modules_layer4_modules_5_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_86 = y_259.sigmoid()
        y_259 = None
        y_260 = sigmoid_86.view(1, -1, 1, 1)
        sigmoid_86 = None
        expand_as_86 = y_260.expand_as(x_956)
        y_260 = None
        x_957 = x_956 * expand_as_86
        x_956 = expand_as_86 = None
        x_957 += x_948
        x_958 = x_957
        x_957 = x_948 = None
        x_959 = torch.nn.functional.relu(x_958, inplace=True)
        x_958 = None
        x_960 = torch.conv2d(
            x_959,
            l_self_modules_layer4_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_6_modules_conv1_parameters_weight_ = None
        x_961 = torch.nn.functional.batch_norm(
            x_960,
            l_self_modules_layer4_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_960 = (
            l_self_modules_layer4_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_6_modules_bn1_parameters_bias_ = None
        x_962 = torch.nn.functional.relu(x_961, inplace=True)
        x_961 = None
        x_963 = torch.conv2d(
            x_962,
            l_self_modules_layer4_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_962 = l_self_modules_layer4_modules_6_modules_conv2_parameters_weight_ = None
        x_964 = torch.nn.functional.batch_norm(
            x_963,
            l_self_modules_layer4_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_963 = (
            l_self_modules_layer4_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_6_modules_bn2_parameters_bias_ = None
        x_965 = torch.nn.functional.relu(x_964, inplace=True)
        x_964 = None
        x_966 = torch.conv2d(
            x_965,
            l_self_modules_layer4_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_965 = l_self_modules_layer4_modules_6_modules_conv3_parameters_weight_ = None
        x_967 = torch.nn.functional.batch_norm(
            x_966,
            l_self_modules_layer4_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_966 = (
            l_self_modules_layer4_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_6_modules_bn3_parameters_bias_ = None
        mean_87 = x_967.mean((2, 3))
        y_261 = mean_87.view(1, 1, -1)
        mean_87 = None
        y_262 = torch.conv1d(
            y_261,
            l_self_modules_layer4_modules_6_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_261 = (
            l_self_modules_layer4_modules_6_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_87 = y_262.sigmoid()
        y_262 = None
        y_263 = sigmoid_87.view(1, -1, 1, 1)
        sigmoid_87 = None
        expand_as_87 = y_263.expand_as(x_967)
        y_263 = None
        x_968 = x_967 * expand_as_87
        x_967 = expand_as_87 = None
        x_968 += x_959
        x_969 = x_968
        x_968 = x_959 = None
        x_970 = torch.nn.functional.relu(x_969, inplace=True)
        x_969 = None
        x_971 = torch.conv2d(
            x_970,
            l_self_modules_layer4_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_7_modules_conv1_parameters_weight_ = None
        x_972 = torch.nn.functional.batch_norm(
            x_971,
            l_self_modules_layer4_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_971 = (
            l_self_modules_layer4_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_7_modules_bn1_parameters_bias_ = None
        x_973 = torch.nn.functional.relu(x_972, inplace=True)
        x_972 = None
        x_974 = torch.conv2d(
            x_973,
            l_self_modules_layer4_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_973 = l_self_modules_layer4_modules_7_modules_conv2_parameters_weight_ = None
        x_975 = torch.nn.functional.batch_norm(
            x_974,
            l_self_modules_layer4_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_974 = (
            l_self_modules_layer4_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_7_modules_bn2_parameters_bias_ = None
        x_976 = torch.nn.functional.relu(x_975, inplace=True)
        x_975 = None
        x_977 = torch.conv2d(
            x_976,
            l_self_modules_layer4_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_976 = l_self_modules_layer4_modules_7_modules_conv3_parameters_weight_ = None
        x_978 = torch.nn.functional.batch_norm(
            x_977,
            l_self_modules_layer4_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_977 = (
            l_self_modules_layer4_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_7_modules_bn3_parameters_bias_ = None
        mean_88 = x_978.mean((2, 3))
        y_264 = mean_88.view(1, 1, -1)
        mean_88 = None
        y_265 = torch.conv1d(
            y_264,
            l_self_modules_layer4_modules_7_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_264 = (
            l_self_modules_layer4_modules_7_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_88 = y_265.sigmoid()
        y_265 = None
        y_266 = sigmoid_88.view(1, -1, 1, 1)
        sigmoid_88 = None
        expand_as_88 = y_266.expand_as(x_978)
        y_266 = None
        x_979 = x_978 * expand_as_88
        x_978 = expand_as_88 = None
        x_979 += x_970
        x_980 = x_979
        x_979 = x_970 = None
        x_981 = torch.nn.functional.relu(x_980, inplace=True)
        x_980 = None
        x_982 = torch.nn.functional.adaptive_avg_pool2d(x_981, 1)
        x_981 = None
        x_983 = x_982.flatten(1, -1)
        x_982 = None
        x_984 = torch._C._nn.linear(
            x_983,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_983 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_984,)
