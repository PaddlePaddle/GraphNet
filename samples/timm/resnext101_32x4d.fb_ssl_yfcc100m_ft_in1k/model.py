import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
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
        L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv1_parameters_weight_ = (
            L_self_modules_conv1_parameters_weight_
        )
        l_x_ = L_x_
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
        l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
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
        l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
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
        l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
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
        l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
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
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_6 = l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = None
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
        input_1 = torch.conv2d(
            x_3,
            l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_11 += input_2
        x_12 = x_11
        x_11 = input_2 = None
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
        x_21 += x_13
        x_22 = x_21
        x_21 = x_13 = None
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
            32,
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
        x_31 += x_23
        x_32 = x_31
        x_31 = x_23 = None
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        x_36 = l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = None
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = None
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        input_3 = torch.conv2d(
            x_33,
            l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_41 += input_4
        x_42 = x_41
        x_41 = input_4 = None
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_46 = l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        x_51 += x_43
        x_52 = x_51
        x_51 = x_43 = None
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = None
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = None
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_56 = l_self_modules_layer2_modules_2_modules_conv2_parameters_weight_ = None
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn2_parameters_bias_ = None
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = None
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = None
        x_61 += x_53
        x_62 = x_61
        x_61 = x_53 = None
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = None
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = None
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_66 = l_self_modules_layer2_modules_3_modules_conv2_parameters_weight_ = None
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn2_parameters_bias_ = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = None
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = None
        x_71 += x_63
        x_72 = x_71
        x_71 = x_63 = None
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        x_76 = l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        input_5 = torch.conv2d(
            x_73,
            l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_81 += input_6
        x_82 = x_81
        x_81 = input_6 = None
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_86 = l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        x_91 += x_83
        x_92 = x_91
        x_91 = x_83 = None
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_96 = l_self_modules_layer3_modules_2_modules_conv2_parameters_weight_ = None
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn2_parameters_bias_ = None
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        x_101 += x_93
        x_102 = x_101
        x_101 = x_93 = None
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_106 = l_self_modules_layer3_modules_3_modules_conv2_parameters_weight_ = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn2_parameters_bias_ = None
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        x_111 += x_103
        x_112 = x_111
        x_111 = x_103 = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_116 = l_self_modules_layer3_modules_4_modules_conv2_parameters_weight_ = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn2_parameters_bias_ = None
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = None
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        x_121 += x_113
        x_122 = x_121
        x_121 = x_113 = None
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_126 = l_self_modules_layer3_modules_5_modules_conv2_parameters_weight_ = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn2_parameters_bias_ = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = None
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        x_131 += x_123
        x_132 = x_131
        x_131 = x_123 = None
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = None
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = None
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_136 = l_self_modules_layer3_modules_6_modules_conv2_parameters_weight_ = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn2_parameters_bias_ = None
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = None
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = None
        x_141 += x_133
        x_142 = x_141
        x_141 = x_133 = None
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = None
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = None
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_146 = l_self_modules_layer3_modules_7_modules_conv2_parameters_weight_ = None
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn2_parameters_bias_ = None
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = None
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = None
        x_151 += x_143
        x_152 = x_151
        x_151 = x_143 = None
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = None
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = None
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_156 = l_self_modules_layer3_modules_8_modules_conv2_parameters_weight_ = None
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn2_parameters_bias_ = None
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = None
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = None
        x_161 += x_153
        x_162 = x_161
        x_161 = x_153 = None
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = None
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = None
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_166 = l_self_modules_layer3_modules_9_modules_conv2_parameters_weight_ = None
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn2_parameters_bias_ = None
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = None
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = None
        x_171 += x_163
        x_172 = x_171
        x_171 = x_163 = None
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = None
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = None
        x_176 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_176 = l_self_modules_layer3_modules_10_modules_conv2_parameters_weight_ = None
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn2_parameters_bias_ = None
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = None
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = None
        x_181 += x_173
        x_182 = x_181
        x_181 = x_173 = None
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = None
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = None
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_186 = l_self_modules_layer3_modules_11_modules_conv2_parameters_weight_ = None
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn2_parameters_bias_ = None
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = None
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = None
        x_191 += x_183
        x_192 = x_191
        x_191 = x_183 = None
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = None
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = None
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_196 = l_self_modules_layer3_modules_12_modules_conv2_parameters_weight_ = None
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn2_parameters_bias_ = None
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = None
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = None
        x_201 += x_193
        x_202 = x_201
        x_201 = x_193 = None
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = None
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = None
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_206 = l_self_modules_layer3_modules_13_modules_conv2_parameters_weight_ = None
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn2_parameters_bias_ = None
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = None
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = None
        x_211 += x_203
        x_212 = x_211
        x_211 = x_203 = None
        x_213 = torch.nn.functional.relu(x_212, inplace=True)
        x_212 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = None
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = None
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_216 = l_self_modules_layer3_modules_14_modules_conv2_parameters_weight_ = None
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn2_parameters_bias_ = None
        x_219 = torch.nn.functional.relu(x_218, inplace=True)
        x_218 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = None
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = None
        x_221 += x_213
        x_222 = x_221
        x_221 = x_213 = None
        x_223 = torch.nn.functional.relu(x_222, inplace=True)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = None
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = None
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_226 = l_self_modules_layer3_modules_15_modules_conv2_parameters_weight_ = None
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn2_parameters_bias_ = None
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = None
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = None
        x_231 += x_223
        x_232 = x_231
        x_231 = x_223 = None
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = None
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = None
        x_236 = torch.nn.functional.relu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_236 = l_self_modules_layer3_modules_16_modules_conv2_parameters_weight_ = None
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn2_parameters_bias_ = None
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = None
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = None
        x_241 += x_233
        x_242 = x_241
        x_241 = x_233 = None
        x_243 = torch.nn.functional.relu(x_242, inplace=True)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = None
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = None
        x_246 = torch.nn.functional.relu(x_245, inplace=True)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_246 = l_self_modules_layer3_modules_17_modules_conv2_parameters_weight_ = None
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_247 = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn2_parameters_bias_ = None
        x_249 = torch.nn.functional.relu(x_248, inplace=True)
        x_248 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = None
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_250 = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = None
        x_251 += x_243
        x_252 = x_251
        x_251 = x_243 = None
        x_253 = torch.nn.functional.relu(x_252, inplace=True)
        x_252 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = None
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = None
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_256 = l_self_modules_layer3_modules_18_modules_conv2_parameters_weight_ = None
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn2_parameters_bias_ = None
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = None
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = None
        x_261 += x_253
        x_262 = x_261
        x_261 = x_253 = None
        x_263 = torch.nn.functional.relu(x_262, inplace=True)
        x_262 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = None
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = None
        x_266 = torch.nn.functional.relu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_266 = l_self_modules_layer3_modules_19_modules_conv2_parameters_weight_ = None
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn2_parameters_bias_ = None
        x_269 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = None
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = None
        x_271 += x_263
        x_272 = x_271
        x_271 = x_263 = None
        x_273 = torch.nn.functional.relu(x_272, inplace=True)
        x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = None
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = None
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_276 = l_self_modules_layer3_modules_20_modules_conv2_parameters_weight_ = None
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn2_parameters_bias_ = None
        x_279 = torch.nn.functional.relu(x_278, inplace=True)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = None
        x_281 = torch.nn.functional.batch_norm(
            x_280,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_280 = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = None
        x_281 += x_273
        x_282 = x_281
        x_281 = x_273 = None
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = None
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = None
        x_286 = torch.nn.functional.relu(x_285, inplace=True)
        x_285 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_286 = l_self_modules_layer3_modules_21_modules_conv2_parameters_weight_ = None
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn2_parameters_bias_ = None
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = None
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = None
        x_291 += x_283
        x_292 = x_291
        x_291 = x_283 = None
        x_293 = torch.nn.functional.relu(x_292, inplace=True)
        x_292 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = None
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = None
        x_296 = torch.nn.functional.relu(x_295, inplace=True)
        x_295 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_296 = l_self_modules_layer3_modules_22_modules_conv2_parameters_weight_ = None
        x_298 = torch.nn.functional.batch_norm(
            x_297,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_297 = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn2_parameters_bias_ = None
        x_299 = torch.nn.functional.relu(x_298, inplace=True)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_299 = l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = None
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_300 = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = None
        x_301 += x_293
        x_302 = x_301
        x_301 = x_293 = None
        x_303 = torch.nn.functional.relu(x_302, inplace=True)
        x_302 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_305 = torch.nn.functional.batch_norm(
            x_304,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_304 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_306 = torch.nn.functional.relu(x_305, inplace=True)
        x_305 = None
        x_307 = torch.conv2d(
            x_306,
            l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        x_306 = l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_308 = torch.nn.functional.batch_norm(
            x_307,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_307 = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        x_309 = torch.nn.functional.relu(x_308, inplace=True)
        x_308 = None
        x_310 = torch.conv2d(
            x_309,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_309 = l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = None
        x_311 = torch.nn.functional.batch_norm(
            x_310,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_310 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        input_7 = torch.conv2d(
            x_303,
            l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_303 = l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_311 += input_8
        x_312 = x_311
        x_311 = input_8 = None
        x_313 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_316 = torch.nn.functional.relu(x_315, inplace=True)
        x_315 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_316 = l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_317 = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_319 = torch.nn.functional.relu(x_318, inplace=True)
        x_318 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = None
        x_321 = torch.nn.functional.batch_norm(
            x_320,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_320 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        x_321 += x_313
        x_322 = x_321
        x_321 = x_313 = None
        x_323 = torch.nn.functional.relu(x_322, inplace=True)
        x_322 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        x_325 = torch.nn.functional.batch_norm(
            x_324,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_324 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        x_326 = torch.nn.functional.relu(x_325, inplace=True)
        x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_326 = l_self_modules_layer4_modules_2_modules_conv2_parameters_weight_ = None
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_327 = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn2_parameters_bias_ = None
        x_329 = torch.nn.functional.relu(x_328, inplace=True)
        x_328 = None
        x_330 = torch.conv2d(
            x_329,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = None
        x_331 = torch.nn.functional.batch_norm(
            x_330,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_330 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        x_331 += x_323
        x_332 = x_331
        x_331 = x_323 = None
        x_333 = torch.nn.functional.relu(x_332, inplace=True)
        x_332 = None
        x_334 = torch.nn.functional.adaptive_avg_pool2d(x_333, 1)
        x_333 = None
        x_335 = x_334.flatten(1, -1)
        x_334 = None
        x_336 = torch._C._nn.linear(
            x_335,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_335 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_336,)
