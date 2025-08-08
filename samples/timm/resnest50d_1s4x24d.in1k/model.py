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
        L_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        out = torch.conv2d(
            x_2,
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        out_1 = torch.nn.functional.batch_norm(
            out,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        out_2 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        x_3 = torch.conv2d(
            out_2,
            l_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_2 = l_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        sym_sum = torch.sym_sum([-1, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        x_gap = x_5.mean((2, 3), keepdim=True)
        x_gap_1 = torch.conv2d(
            x_gap,
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_2 = torch.nn.functional.batch_norm(
            x_gap_1,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_1 = l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_3 = torch.nn.functional.relu(x_gap_2, inplace=True)
        x_gap_2 = None
        x_attn = torch.conv2d(
            x_gap_3,
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_3 = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_6 = torch.sigmoid(x_attn)
        x_attn = None
        x_attn_1 = x_6.view(1, -1, 1, 1)
        x_6 = None
        out_3 = x_5 * x_attn_1
        x_5 = x_attn_1 = None
        out_4 = out_3.contiguous()
        out_3 = None
        out_5 = torch.conv2d(
            out_4,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_4 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        out_6 = torch.nn.functional.batch_norm(
            out_5,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_5 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
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
        out_6 += input_9
        out_7 = out_6
        out_6 = input_9 = None
        out_8 = torch.nn.functional.relu(out_7, inplace=True)
        out_7 = None
        out_9 = torch.conv2d(
            out_8,
            l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = None
        out_10 = torch.nn.functional.batch_norm(
            out_9,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_9 = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = None
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        x_7 = torch.conv2d(
            out_11,
            l_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_11 = l_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_gap_4 = x_9.mean((2, 3), keepdim=True)
        x_gap_5 = torch.conv2d(
            x_gap_4,
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_4 = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_6 = torch.nn.functional.batch_norm(
            x_gap_5,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_5 = l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_7 = torch.nn.functional.relu(x_gap_6, inplace=True)
        x_gap_6 = None
        x_attn_2 = torch.conv2d(
            x_gap_7,
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_7 = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_10 = torch.sigmoid(x_attn_2)
        x_attn_2 = None
        x_attn_3 = x_10.view(1, -1, 1, 1)
        x_10 = None
        out_12 = x_9 * x_attn_3
        x_9 = x_attn_3 = None
        out_13 = out_12.contiguous()
        out_12 = None
        out_14 = torch.conv2d(
            out_13,
            l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = None
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = None
        out_15 += out_8
        out_16 = out_15
        out_15 = out_8 = None
        out_17 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        out_18 = torch.conv2d(
            out_17,
            l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_ = None
        out_19 = torch.nn.functional.batch_norm(
            out_18,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_18 = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_ = None
        out_20 = torch.nn.functional.relu(out_19, inplace=True)
        out_19 = None
        x_11 = torch.conv2d(
            out_20,
            l_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_20 = l_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_gap_8 = x_13.mean((2, 3), keepdim=True)
        x_gap_9 = torch.conv2d(
            x_gap_8,
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_8 = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_10 = torch.nn.functional.batch_norm(
            x_gap_9,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_9 = l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_11 = torch.nn.functional.relu(x_gap_10, inplace=True)
        x_gap_10 = None
        x_attn_4 = torch.conv2d(
            x_gap_11,
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_11 = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_14 = torch.sigmoid(x_attn_4)
        x_attn_4 = None
        x_attn_5 = x_14.view(1, -1, 1, 1)
        x_14 = None
        out_21 = x_13 * x_attn_5
        x_13 = x_attn_5 = None
        out_22 = out_21.contiguous()
        out_21 = None
        out_23 = torch.conv2d(
            out_22,
            l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_22 = l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_ = None
        out_24 = torch.nn.functional.batch_norm(
            out_23,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_23 = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_ = None
        out_24 += out_17
        out_25 = out_24
        out_24 = out_17 = None
        out_26 = torch.nn.functional.relu(out_25, inplace=True)
        out_25 = None
        out_27 = torch.conv2d(
            out_26,
            l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        out_28 = torch.nn.functional.batch_norm(
            out_27,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_27 = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        out_30 = torch._C._nn.avg_pool2d(out_29, 3, 2, 1, False, True, None)
        out_29 = None
        x_15 = torch.conv2d(
            out_30,
            l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_30 = l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        x_gap_12 = x_17.mean((2, 3), keepdim=True)
        x_gap_13 = torch.conv2d(
            x_gap_12,
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_12 = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_14 = torch.nn.functional.batch_norm(
            x_gap_13,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_13 = l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_15 = torch.nn.functional.relu(x_gap_14, inplace=True)
        x_gap_14 = None
        x_attn_6 = torch.conv2d(
            x_gap_15,
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_15 = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_18 = torch.sigmoid(x_attn_6)
        x_attn_6 = None
        x_attn_7 = x_18.view(1, -1, 1, 1)
        x_18 = None
        out_31 = x_17 * x_attn_7
        x_17 = x_attn_7 = None
        out_32 = out_31.contiguous()
        out_31 = None
        out_33 = torch.conv2d(
            out_32,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_32 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        out_34 = torch.nn.functional.batch_norm(
            out_33,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_33 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        input_10 = torch._C._nn.avg_pool2d(out_26, 2, 2, 0, True, False, None)
        out_26 = None
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
        out_34 += input_12
        out_35 = out_34
        out_34 = input_12 = None
        out_36 = torch.nn.functional.relu(out_35, inplace=True)
        out_35 = None
        out_37 = torch.conv2d(
            out_36,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        out_38 = torch.nn.functional.batch_norm(
            out_37,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_37 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        out_39 = torch.nn.functional.relu(out_38, inplace=True)
        out_38 = None
        x_19 = torch.conv2d(
            out_39,
            l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_39 = l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_gap_16 = x_21.mean((2, 3), keepdim=True)
        x_gap_17 = torch.conv2d(
            x_gap_16,
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_16 = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_18 = torch.nn.functional.batch_norm(
            x_gap_17,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_17 = l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_19 = torch.nn.functional.relu(x_gap_18, inplace=True)
        x_gap_18 = None
        x_attn_8 = torch.conv2d(
            x_gap_19,
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_19 = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_22 = torch.sigmoid(x_attn_8)
        x_attn_8 = None
        x_attn_9 = x_22.view(1, -1, 1, 1)
        x_22 = None
        out_40 = x_21 * x_attn_9
        x_21 = x_attn_9 = None
        out_41 = out_40.contiguous()
        out_40 = None
        out_42 = torch.conv2d(
            out_41,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        out_43 = torch.nn.functional.batch_norm(
            out_42,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_42 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        out_43 += out_36
        out_44 = out_43
        out_43 = out_36 = None
        out_45 = torch.nn.functional.relu(out_44, inplace=True)
        out_44 = None
        out_46 = torch.conv2d(
            out_45,
            l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = None
        out_47 = torch.nn.functional.batch_norm(
            out_46,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_46 = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = None
        out_48 = torch.nn.functional.relu(out_47, inplace=True)
        out_47 = None
        x_23 = torch.conv2d(
            out_48,
            l_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_48 = l_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_gap_20 = x_25.mean((2, 3), keepdim=True)
        x_gap_21 = torch.conv2d(
            x_gap_20,
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_20 = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_22 = torch.nn.functional.batch_norm(
            x_gap_21,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_21 = l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_23 = torch.nn.functional.relu(x_gap_22, inplace=True)
        x_gap_22 = None
        x_attn_10 = torch.conv2d(
            x_gap_23,
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_23 = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_26 = torch.sigmoid(x_attn_10)
        x_attn_10 = None
        x_attn_11 = x_26.view(1, -1, 1, 1)
        x_26 = None
        out_49 = x_25 * x_attn_11
        x_25 = x_attn_11 = None
        out_50 = out_49.contiguous()
        out_49 = None
        out_51 = torch.conv2d(
            out_50,
            l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_50 = l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = None
        out_52 = torch.nn.functional.batch_norm(
            out_51,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_51 = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = None
        out_52 += out_45
        out_53 = out_52
        out_52 = out_45 = None
        out_54 = torch.nn.functional.relu(out_53, inplace=True)
        out_53 = None
        out_55 = torch.conv2d(
            out_54,
            l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = None
        out_56 = torch.nn.functional.batch_norm(
            out_55,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_55 = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = None
        out_57 = torch.nn.functional.relu(out_56, inplace=True)
        out_56 = None
        x_27 = torch.conv2d(
            out_57,
            l_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_57 = l_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_gap_24 = x_29.mean((2, 3), keepdim=True)
        x_gap_25 = torch.conv2d(
            x_gap_24,
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_24 = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_26 = torch.nn.functional.batch_norm(
            x_gap_25,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_25 = l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_27 = torch.nn.functional.relu(x_gap_26, inplace=True)
        x_gap_26 = None
        x_attn_12 = torch.conv2d(
            x_gap_27,
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_27 = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_30 = torch.sigmoid(x_attn_12)
        x_attn_12 = None
        x_attn_13 = x_30.view(1, -1, 1, 1)
        x_30 = None
        out_58 = x_29 * x_attn_13
        x_29 = x_attn_13 = None
        out_59 = out_58.contiguous()
        out_58 = None
        out_60 = torch.conv2d(
            out_59,
            l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_59 = l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = None
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = None
        out_61 += out_54
        out_62 = out_61
        out_61 = out_54 = None
        out_63 = torch.nn.functional.relu(out_62, inplace=True)
        out_62 = None
        out_64 = torch.conv2d(
            out_63,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        out_65 = torch.nn.functional.batch_norm(
            out_64,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_64 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        out_66 = torch.nn.functional.relu(out_65, inplace=True)
        out_65 = None
        out_67 = torch._C._nn.avg_pool2d(out_66, 3, 2, 1, False, True, None)
        out_66 = None
        x_31 = torch.conv2d(
            out_67,
            l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_67 = l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        x_gap_28 = x_33.mean((2, 3), keepdim=True)
        x_gap_29 = torch.conv2d(
            x_gap_28,
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_28 = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_30 = torch.nn.functional.batch_norm(
            x_gap_29,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_29 = l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_31 = torch.nn.functional.relu(x_gap_30, inplace=True)
        x_gap_30 = None
        x_attn_14 = torch.conv2d(
            x_gap_31,
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_31 = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_34 = torch.sigmoid(x_attn_14)
        x_attn_14 = None
        x_attn_15 = x_34.view(1, -1, 1, 1)
        x_34 = None
        out_68 = x_33 * x_attn_15
        x_33 = x_attn_15 = None
        out_69 = out_68.contiguous()
        out_68 = None
        out_70 = torch.conv2d(
            out_69,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_69 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        input_13 = torch._C._nn.avg_pool2d(out_63, 2, 2, 0, True, False, None)
        out_63 = None
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
        out_71 += input_15
        out_72 = out_71
        out_71 = input_15 = None
        out_73 = torch.nn.functional.relu(out_72, inplace=True)
        out_72 = None
        out_74 = torch.conv2d(
            out_73,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        out_75 = torch.nn.functional.batch_norm(
            out_74,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_74 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        out_76 = torch.nn.functional.relu(out_75, inplace=True)
        out_75 = None
        x_35 = torch.conv2d(
            out_76,
            l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_76 = l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_gap_32 = x_37.mean((2, 3), keepdim=True)
        x_gap_33 = torch.conv2d(
            x_gap_32,
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_32 = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_34 = torch.nn.functional.batch_norm(
            x_gap_33,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_33 = l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_35 = torch.nn.functional.relu(x_gap_34, inplace=True)
        x_gap_34 = None
        x_attn_16 = torch.conv2d(
            x_gap_35,
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_35 = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_38 = torch.sigmoid(x_attn_16)
        x_attn_16 = None
        x_attn_17 = x_38.view(1, -1, 1, 1)
        x_38 = None
        out_77 = x_37 * x_attn_17
        x_37 = x_attn_17 = None
        out_78 = out_77.contiguous()
        out_77 = None
        out_79 = torch.conv2d(
            out_78,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_78 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        out_80 = torch.nn.functional.batch_norm(
            out_79,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_79 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        out_80 += out_73
        out_81 = out_80
        out_80 = out_73 = None
        out_82 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        out_83 = torch.conv2d(
            out_82,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        out_84 = torch.nn.functional.batch_norm(
            out_83,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_83 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        out_85 = torch.nn.functional.relu(out_84, inplace=True)
        out_84 = None
        x_39 = torch.conv2d(
            out_85,
            l_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_85 = l_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_gap_36 = x_41.mean((2, 3), keepdim=True)
        x_gap_37 = torch.conv2d(
            x_gap_36,
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_36 = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_38 = torch.nn.functional.batch_norm(
            x_gap_37,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_37 = l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_39 = torch.nn.functional.relu(x_gap_38, inplace=True)
        x_gap_38 = None
        x_attn_18 = torch.conv2d(
            x_gap_39,
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_39 = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_42 = torch.sigmoid(x_attn_18)
        x_attn_18 = None
        x_attn_19 = x_42.view(1, -1, 1, 1)
        x_42 = None
        out_86 = x_41 * x_attn_19
        x_41 = x_attn_19 = None
        out_87 = out_86.contiguous()
        out_86 = None
        out_88 = torch.conv2d(
            out_87,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_87 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        out_89 = torch.nn.functional.batch_norm(
            out_88,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_88 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        out_89 += out_82
        out_90 = out_89
        out_89 = out_82 = None
        out_91 = torch.nn.functional.relu(out_90, inplace=True)
        out_90 = None
        out_92 = torch.conv2d(
            out_91,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        out_93 = torch.nn.functional.batch_norm(
            out_92,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_92 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        out_94 = torch.nn.functional.relu(out_93, inplace=True)
        out_93 = None
        x_43 = torch.conv2d(
            out_94,
            l_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_94 = l_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_gap_40 = x_45.mean((2, 3), keepdim=True)
        x_gap_41 = torch.conv2d(
            x_gap_40,
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_40 = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_42 = torch.nn.functional.batch_norm(
            x_gap_41,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_41 = l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_43 = torch.nn.functional.relu(x_gap_42, inplace=True)
        x_gap_42 = None
        x_attn_20 = torch.conv2d(
            x_gap_43,
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_43 = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_46 = torch.sigmoid(x_attn_20)
        x_attn_20 = None
        x_attn_21 = x_46.view(1, -1, 1, 1)
        x_46 = None
        out_95 = x_45 * x_attn_21
        x_45 = x_attn_21 = None
        out_96 = out_95.contiguous()
        out_95 = None
        out_97 = torch.conv2d(
            out_96,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_96 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        out_98 = torch.nn.functional.batch_norm(
            out_97,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_97 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        out_98 += out_91
        out_99 = out_98
        out_98 = out_91 = None
        out_100 = torch.nn.functional.relu(out_99, inplace=True)
        out_99 = None
        out_101 = torch.conv2d(
            out_100,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        out_102 = torch.nn.functional.batch_norm(
            out_101,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_101 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        out_103 = torch.nn.functional.relu(out_102, inplace=True)
        out_102 = None
        x_47 = torch.conv2d(
            out_103,
            l_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_103 = l_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_gap_44 = x_49.mean((2, 3), keepdim=True)
        x_gap_45 = torch.conv2d(
            x_gap_44,
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_44 = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_46 = torch.nn.functional.batch_norm(
            x_gap_45,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_45 = l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_47 = torch.nn.functional.relu(x_gap_46, inplace=True)
        x_gap_46 = None
        x_attn_22 = torch.conv2d(
            x_gap_47,
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_47 = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_50 = torch.sigmoid(x_attn_22)
        x_attn_22 = None
        x_attn_23 = x_50.view(1, -1, 1, 1)
        x_50 = None
        out_104 = x_49 * x_attn_23
        x_49 = x_attn_23 = None
        out_105 = out_104.contiguous()
        out_104 = None
        out_106 = torch.conv2d(
            out_105,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = (
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_
        ) = None
        out_107 = torch.nn.functional.batch_norm(
            out_106,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_106 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        out_107 += out_100
        out_108 = out_107
        out_107 = out_100 = None
        out_109 = torch.nn.functional.relu(out_108, inplace=True)
        out_108 = None
        out_110 = torch.conv2d(
            out_109,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        out_112 = torch.nn.functional.relu(out_111, inplace=True)
        out_111 = None
        x_51 = torch.conv2d(
            out_112,
            l_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_112 = l_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_gap_48 = x_53.mean((2, 3), keepdim=True)
        x_gap_49 = torch.conv2d(
            x_gap_48,
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_48 = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_50 = torch.nn.functional.batch_norm(
            x_gap_49,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_49 = l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_51 = torch.nn.functional.relu(x_gap_50, inplace=True)
        x_gap_50 = None
        x_attn_24 = torch.conv2d(
            x_gap_51,
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_51 = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_54 = torch.sigmoid(x_attn_24)
        x_attn_24 = None
        x_attn_25 = x_54.view(1, -1, 1, 1)
        x_54 = None
        out_113 = x_53 * x_attn_25
        x_53 = x_attn_25 = None
        out_114 = out_113.contiguous()
        out_113 = None
        out_115 = torch.conv2d(
            out_114,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_114 = (
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_
        ) = None
        out_116 = torch.nn.functional.batch_norm(
            out_115,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_115 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        out_116 += out_109
        out_117 = out_116
        out_116 = out_109 = None
        out_118 = torch.nn.functional.relu(out_117, inplace=True)
        out_117 = None
        out_119 = torch.conv2d(
            out_118,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        out_120 = torch.nn.functional.batch_norm(
            out_119,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_119 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        out_121 = torch.nn.functional.relu(out_120, inplace=True)
        out_120 = None
        out_122 = torch._C._nn.avg_pool2d(out_121, 3, 2, 1, False, True, None)
        out_121 = None
        x_55 = torch.conv2d(
            out_122,
            l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_122 = l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        x_gap_52 = x_57.mean((2, 3), keepdim=True)
        x_gap_53 = torch.conv2d(
            x_gap_52,
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_52 = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_54 = torch.nn.functional.batch_norm(
            x_gap_53,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_53 = l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_55 = torch.nn.functional.relu(x_gap_54, inplace=True)
        x_gap_54 = None
        x_attn_26 = torch.conv2d(
            x_gap_55,
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_55 = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_58 = torch.sigmoid(x_attn_26)
        x_attn_26 = None
        x_attn_27 = x_58.view(1, -1, 1, 1)
        x_58 = None
        out_123 = x_57 * x_attn_27
        x_57 = x_attn_27 = None
        out_124 = out_123.contiguous()
        out_123 = None
        out_125 = torch.conv2d(
            out_124,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_124 = (
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_126 = torch.nn.functional.batch_norm(
            out_125,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_125 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        input_16 = torch._C._nn.avg_pool2d(out_118, 2, 2, 0, True, False, None)
        out_118 = None
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
        out_126 += input_18
        out_127 = out_126
        out_126 = input_18 = None
        out_128 = torch.nn.functional.relu(out_127, inplace=True)
        out_127 = None
        out_129 = torch.conv2d(
            out_128,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        out_130 = torch.nn.functional.batch_norm(
            out_129,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_129 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        out_131 = torch.nn.functional.relu(out_130, inplace=True)
        out_130 = None
        x_59 = torch.conv2d(
            out_131,
            l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_131 = l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_gap_56 = x_61.mean((2, 3), keepdim=True)
        x_gap_57 = torch.conv2d(
            x_gap_56,
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_56 = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_58 = torch.nn.functional.batch_norm(
            x_gap_57,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_57 = l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_59 = torch.nn.functional.relu(x_gap_58, inplace=True)
        x_gap_58 = None
        x_attn_28 = torch.conv2d(
            x_gap_59,
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_59 = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_62 = torch.sigmoid(x_attn_28)
        x_attn_28 = None
        x_attn_29 = x_62.view(1, -1, 1, 1)
        x_62 = None
        out_132 = x_61 * x_attn_29
        x_61 = x_attn_29 = None
        out_133 = out_132.contiguous()
        out_132 = None
        out_134 = torch.conv2d(
            out_133,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_133 = (
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_135 = torch.nn.functional.batch_norm(
            out_134,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_134 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        out_135 += out_128
        out_136 = out_135
        out_135 = out_128 = None
        out_137 = torch.nn.functional.relu(out_136, inplace=True)
        out_136 = None
        out_138 = torch.conv2d(
            out_137,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        out_139 = torch.nn.functional.batch_norm(
            out_138,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_138 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        out_140 = torch.nn.functional.relu(out_139, inplace=True)
        out_139 = None
        x_63 = torch.conv2d(
            out_140,
            l_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        out_140 = l_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_
        ) = None
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_gap_60 = x_65.mean((2, 3), keepdim=True)
        x_gap_61 = torch.conv2d(
            x_gap_60,
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_60 = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_
        ) = None
        x_gap_62 = torch.nn.functional.batch_norm(
            x_gap_61,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_gap_61 = l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_
        ) = None
        x_gap_63 = torch.nn.functional.relu(x_gap_62, inplace=True)
        x_gap_62 = None
        x_attn_30 = torch.conv2d(
            x_gap_63,
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        x_gap_63 = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_
        ) = None
        x_66 = torch.sigmoid(x_attn_30)
        x_attn_30 = None
        x_attn_31 = x_66.view(1, -1, 1, 1)
        x_66 = None
        out_141 = x_65 * x_attn_31
        x_65 = x_attn_31 = None
        out_142 = out_141.contiguous()
        out_141 = None
        out_143 = torch.conv2d(
            out_142,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_142 = (
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_144 = torch.nn.functional.batch_norm(
            out_143,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_143 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        out_144 += out_137
        out_145 = out_144
        out_144 = out_137 = None
        out_146 = torch.nn.functional.relu(out_145, inplace=True)
        out_145 = None
        x_67 = torch.nn.functional.adaptive_avg_pool2d(out_146, 1)
        out_146 = None
        x_68 = x_67.flatten(1, -1)
        x_67 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_68 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_69,)
