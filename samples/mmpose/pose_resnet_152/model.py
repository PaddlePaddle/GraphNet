import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_conv1_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_var_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_var_
        )
        l_self_modules_backbone_modules_bn1_parameters_weight_ = (
            L_self_modules_backbone_modules_bn1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_parameters_bias_ = (
            L_self_modules_backbone_modules_bn1_parameters_bias_
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_
        l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_
        )
        l_self_modules_head_modules_final_layer_parameters_weight_ = (
            L_self_modules_head_modules_final_layer_parameters_weight_
        )
        l_self_modules_head_modules_final_layer_parameters_bias_ = (
            L_self_modules_head_modules_final_layer_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_backbone_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_bn1_parameters_weight_
        ) = l_self_modules_backbone_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        out = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_1 = torch.nn.functional.batch_norm(
            out,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (None)
        out_2 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        out_3 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (None)
        out_4 = torch.nn.functional.batch_norm(
            out_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (None)
        out_5 = torch.nn.functional.relu(out_4, inplace=True)
        out_4 = None
        out_6 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (None)
        out_7 = torch.nn.functional.batch_norm(
            out_6,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_6 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_1 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_7 += input_2
        out_8 = out_7
        out_7 = input_2 = None
        out_9 = torch.nn.functional.relu(out_8, inplace=True)
        out_8 = None
        out_10 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_11 = torch.nn.functional.batch_norm(
            out_10,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_10 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_12 = torch.nn.functional.relu(out_11, inplace=True)
        out_11 = None
        out_13 = torch.conv2d(
            out_12,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_12 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (None)
        out_14 = torch.nn.functional.batch_norm(
            out_13,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_13 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (None)
        out_15 = torch.nn.functional.relu(out_14, inplace=True)
        out_14 = None
        out_16 = torch.conv2d(
            out_15,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_15 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (None)
        out_17 = torch.nn.functional.batch_norm(
            out_16,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_16 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (None)
        out_17 += out_9
        out_18 = out_17
        out_17 = out_9 = None
        out_19 = torch.nn.functional.relu(out_18, inplace=True)
        out_18 = None
        out_20 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_21 = torch.nn.functional.batch_norm(
            out_20,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_20 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (None)
        out_22 = torch.nn.functional.relu(out_21, inplace=True)
        out_21 = None
        out_23 = torch.conv2d(
            out_22,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_22 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = (None)
        out_24 = torch.nn.functional.batch_norm(
            out_23,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_23 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = (None)
        out_25 = torch.nn.functional.relu(out_24, inplace=True)
        out_24 = None
        out_26 = torch.conv2d(
            out_25,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_25 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (None)
        out_27 = torch.nn.functional.batch_norm(
            out_26,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_26 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (None)
        out_27 += out_19
        out_28 = out_27
        out_27 = out_19 = None
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        out_30 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_31 = torch.nn.functional.batch_norm(
            out_30,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_30 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_32 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        out_33 = torch.conv2d(
            out_32,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_32 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (None)
        out_34 = torch.nn.functional.batch_norm(
            out_33,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_33 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (None)
        out_35 = torch.nn.functional.relu(out_34, inplace=True)
        out_34 = None
        out_36 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (None)
        out_37 = torch.nn.functional.batch_norm(
            out_36,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_36 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_3 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_29 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_37 += input_4
        out_38 = out_37
        out_37 = input_4 = None
        out_39 = torch.nn.functional.relu(out_38, inplace=True)
        out_38 = None
        out_40 = torch.conv2d(
            out_39,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_41 = torch.nn.functional.batch_norm(
            out_40,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_40 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_42 = torch.nn.functional.relu(out_41, inplace=True)
        out_41 = None
        out_43 = torch.conv2d(
            out_42,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_42 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (None)
        out_44 = torch.nn.functional.batch_norm(
            out_43,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_43 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (None)
        out_45 = torch.nn.functional.relu(out_44, inplace=True)
        out_44 = None
        out_46 = torch.conv2d(
            out_45,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_45 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (None)
        out_47 = torch.nn.functional.batch_norm(
            out_46,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_46 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (None)
        out_47 += out_39
        out_48 = out_47
        out_47 = out_39 = None
        out_49 = torch.nn.functional.relu(out_48, inplace=True)
        out_48 = None
        out_50 = torch.conv2d(
            out_49,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_51 = torch.nn.functional.batch_norm(
            out_50,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_50 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (None)
        out_52 = torch.nn.functional.relu(out_51, inplace=True)
        out_51 = None
        out_53 = torch.conv2d(
            out_52,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_52 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = (None)
        out_54 = torch.nn.functional.batch_norm(
            out_53,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_53 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = (None)
        out_55 = torch.nn.functional.relu(out_54, inplace=True)
        out_54 = None
        out_56 = torch.conv2d(
            out_55,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_55 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (None)
        out_57 = torch.nn.functional.batch_norm(
            out_56,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_56 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (None)
        out_57 += out_49
        out_58 = out_57
        out_57 = out_49 = None
        out_59 = torch.nn.functional.relu(out_58, inplace=True)
        out_58 = None
        out_60 = torch.conv2d(
            out_59,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (None)
        out_62 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        out_63 = torch.conv2d(
            out_62,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_62 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = (None)
        out_64 = torch.nn.functional.batch_norm(
            out_63,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_63 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = (None)
        out_65 = torch.nn.functional.relu(out_64, inplace=True)
        out_64 = None
        out_66 = torch.conv2d(
            out_65,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_65 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (None)
        out_67 = torch.nn.functional.batch_norm(
            out_66,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_66 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (None)
        out_67 += out_59
        out_68 = out_67
        out_67 = out_59 = None
        out_69 = torch.nn.functional.relu(out_68, inplace=True)
        out_68 = None
        out_70 = torch.conv2d(
            out_69,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_ = (
            None
        )
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_ = (None)
        out_72 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        out_73 = torch.conv2d(
            out_72,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_72 = l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_ = (None)
        out_74 = torch.nn.functional.batch_norm(
            out_73,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_73 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_ = (None)
        out_75 = torch.nn.functional.relu(out_74, inplace=True)
        out_74 = None
        out_76 = torch.conv2d(
            out_75,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_75 = l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_ = (None)
        out_77 = torch.nn.functional.batch_norm(
            out_76,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_76 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_ = (None)
        out_77 += out_69
        out_78 = out_77
        out_77 = out_69 = None
        out_79 = torch.nn.functional.relu(out_78, inplace=True)
        out_78 = None
        out_80 = torch.conv2d(
            out_79,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_ = (
            None
        )
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_ = (None)
        out_82 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        out_83 = torch.conv2d(
            out_82,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_82 = l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_ = (None)
        out_84 = torch.nn.functional.batch_norm(
            out_83,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_83 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_ = (None)
        out_85 = torch.nn.functional.relu(out_84, inplace=True)
        out_84 = None
        out_86 = torch.conv2d(
            out_85,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_85 = l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_ = (None)
        out_87 = torch.nn.functional.batch_norm(
            out_86,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_86 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_ = (None)
        out_87 += out_79
        out_88 = out_87
        out_87 = out_79 = None
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_90 = torch.conv2d(
            out_89,
            l_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_ = (
            None
        )
        out_91 = torch.nn.functional.batch_norm(
            out_90,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_90 = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_ = (None)
        out_92 = torch.nn.functional.relu(out_91, inplace=True)
        out_91 = None
        out_93 = torch.conv2d(
            out_92,
            l_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_92 = l_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_ = (None)
        out_94 = torch.nn.functional.batch_norm(
            out_93,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_93 = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_ = (None)
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        out_96 = torch.conv2d(
            out_95,
            l_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_95 = l_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_ = (None)
        out_97 = torch.nn.functional.batch_norm(
            out_96,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_96 = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_ = (None)
        out_97 += out_89
        out_98 = out_97
        out_97 = out_89 = None
        out_99 = torch.nn.functional.relu(out_98, inplace=True)
        out_98 = None
        out_100 = torch.conv2d(
            out_99,
            l_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_ = (
            None
        )
        out_101 = torch.nn.functional.batch_norm(
            out_100,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_100 = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_ = (None)
        out_102 = torch.nn.functional.relu(out_101, inplace=True)
        out_101 = None
        out_103 = torch.conv2d(
            out_102,
            l_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_102 = l_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_ = (None)
        out_104 = torch.nn.functional.batch_norm(
            out_103,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_103 = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_ = (None)
        out_105 = torch.nn.functional.relu(out_104, inplace=True)
        out_104 = None
        out_106 = torch.conv2d(
            out_105,
            l_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = l_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_ = (None)
        out_107 = torch.nn.functional.batch_norm(
            out_106,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_106 = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_ = (None)
        out_107 += out_99
        out_108 = out_107
        out_107 = out_99 = None
        out_109 = torch.nn.functional.relu(out_108, inplace=True)
        out_108 = None
        out_110 = torch.conv2d(
            out_109,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_112 = torch.nn.functional.relu(out_111, inplace=True)
        out_111 = None
        out_113 = torch.conv2d(
            out_112,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_112 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (None)
        out_114 = torch.nn.functional.batch_norm(
            out_113,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_113 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        out_115 = torch.nn.functional.relu(out_114, inplace=True)
        out_114 = None
        out_116 = torch.conv2d(
            out_115,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_115 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_117 = torch.nn.functional.batch_norm(
            out_116,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_116 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_5 = torch.conv2d(
            out_109,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_109 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_117 += input_6
        out_118 = out_117
        out_117 = input_6 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        out_120 = torch.conv2d(
            out_119,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_121 = torch.nn.functional.batch_norm(
            out_120,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_120 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_122 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        out_123 = torch.conv2d(
            out_122,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_122 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (None)
        out_124 = torch.nn.functional.batch_norm(
            out_123,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_123 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_125 = torch.nn.functional.relu(out_124, inplace=True)
        out_124 = None
        out_126 = torch.conv2d(
            out_125,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_125 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_127 = torch.nn.functional.batch_norm(
            out_126,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_126 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        out_127 += out_119
        out_128 = out_127
        out_127 = out_119 = None
        out_129 = torch.nn.functional.relu(out_128, inplace=True)
        out_128 = None
        out_130 = torch.conv2d(
            out_129,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_131 = torch.nn.functional.batch_norm(
            out_130,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_130 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_132 = torch.nn.functional.relu(out_131, inplace=True)
        out_131 = None
        out_133 = torch.conv2d(
            out_132,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_132 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (None)
        out_134 = torch.nn.functional.batch_norm(
            out_133,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_133 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (None)
        out_135 = torch.nn.functional.relu(out_134, inplace=True)
        out_134 = None
        out_136 = torch.conv2d(
            out_135,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_135 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_137 = torch.nn.functional.batch_norm(
            out_136,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_136 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        out_137 += out_129
        out_138 = out_137
        out_137 = out_129 = None
        out_139 = torch.nn.functional.relu(out_138, inplace=True)
        out_138 = None
        out_140 = torch.conv2d(
            out_139,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_141 = torch.nn.functional.batch_norm(
            out_140,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_140 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_142 = torch.nn.functional.relu(out_141, inplace=True)
        out_141 = None
        out_143 = torch.conv2d(
            out_142,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_142 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (None)
        out_144 = torch.nn.functional.batch_norm(
            out_143,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_143 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (None)
        out_145 = torch.nn.functional.relu(out_144, inplace=True)
        out_144 = None
        out_146 = torch.conv2d(
            out_145,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_145 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_147 = torch.nn.functional.batch_norm(
            out_146,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_146 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        out_147 += out_139
        out_148 = out_147
        out_147 = out_139 = None
        out_149 = torch.nn.functional.relu(out_148, inplace=True)
        out_148 = None
        out_150 = torch.conv2d(
            out_149,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_ = (
            None
        )
        out_151 = torch.nn.functional.batch_norm(
            out_150,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_150 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_152 = torch.nn.functional.relu(out_151, inplace=True)
        out_151 = None
        out_153 = torch.conv2d(
            out_152,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_152 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (None)
        out_154 = torch.nn.functional.batch_norm(
            out_153,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_153 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (None)
        out_155 = torch.nn.functional.relu(out_154, inplace=True)
        out_154 = None
        out_156 = torch.conv2d(
            out_155,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_155 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_157 = torch.nn.functional.batch_norm(
            out_156,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_156 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        out_157 += out_149
        out_158 = out_157
        out_157 = out_149 = None
        out_159 = torch.nn.functional.relu(out_158, inplace=True)
        out_158 = None
        out_160 = torch.conv2d(
            out_159,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_ = (
            None
        )
        out_161 = torch.nn.functional.batch_norm(
            out_160,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_160 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_162 = torch.nn.functional.relu(out_161, inplace=True)
        out_161 = None
        out_163 = torch.conv2d(
            out_162,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_162 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (None)
        out_164 = torch.nn.functional.batch_norm(
            out_163,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_163 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (None)
        out_165 = torch.nn.functional.relu(out_164, inplace=True)
        out_164 = None
        out_166 = torch.conv2d(
            out_165,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_165 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_167 = torch.nn.functional.batch_norm(
            out_166,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_166 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        out_167 += out_159
        out_168 = out_167
        out_167 = out_159 = None
        out_169 = torch.nn.functional.relu(out_168, inplace=True)
        out_168 = None
        out_170 = torch.conv2d(
            out_169,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_ = (
            None
        )
        out_171 = torch.nn.functional.batch_norm(
            out_170,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_170 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (None)
        out_172 = torch.nn.functional.relu(out_171, inplace=True)
        out_171 = None
        out_173 = torch.conv2d(
            out_172,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_172 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = (None)
        out_174 = torch.nn.functional.batch_norm(
            out_173,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_173 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = (None)
        out_175 = torch.nn.functional.relu(out_174, inplace=True)
        out_174 = None
        out_176 = torch.conv2d(
            out_175,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_175 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (None)
        out_177 = torch.nn.functional.batch_norm(
            out_176,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_176 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (None)
        out_177 += out_169
        out_178 = out_177
        out_177 = out_169 = None
        out_179 = torch.nn.functional.relu(out_178, inplace=True)
        out_178 = None
        out_180 = torch.conv2d(
            out_179,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_ = (
            None
        )
        out_181 = torch.nn.functional.batch_norm(
            out_180,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_180 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_ = (None)
        out_182 = torch.nn.functional.relu(out_181, inplace=True)
        out_181 = None
        out_183 = torch.conv2d(
            out_182,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_182 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_ = (None)
        out_184 = torch.nn.functional.batch_norm(
            out_183,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_183 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_ = (None)
        out_185 = torch.nn.functional.relu(out_184, inplace=True)
        out_184 = None
        out_186 = torch.conv2d(
            out_185,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_185 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_ = (None)
        out_187 = torch.nn.functional.batch_norm(
            out_186,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_186 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_ = (None)
        out_187 += out_179
        out_188 = out_187
        out_187 = out_179 = None
        out_189 = torch.nn.functional.relu(out_188, inplace=True)
        out_188 = None
        out_190 = torch.conv2d(
            out_189,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_ = (
            None
        )
        out_191 = torch.nn.functional.batch_norm(
            out_190,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_190 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_ = (None)
        out_192 = torch.nn.functional.relu(out_191, inplace=True)
        out_191 = None
        out_193 = torch.conv2d(
            out_192,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_192 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_ = (None)
        out_194 = torch.nn.functional.batch_norm(
            out_193,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_193 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_ = (None)
        out_195 = torch.nn.functional.relu(out_194, inplace=True)
        out_194 = None
        out_196 = torch.conv2d(
            out_195,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_195 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_ = (None)
        out_197 = torch.nn.functional.batch_norm(
            out_196,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_196 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_ = (None)
        out_197 += out_189
        out_198 = out_197
        out_197 = out_189 = None
        out_199 = torch.nn.functional.relu(out_198, inplace=True)
        out_198 = None
        out_200 = torch.conv2d(
            out_199,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_ = (
            None
        )
        out_201 = torch.nn.functional.batch_norm(
            out_200,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_200 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_ = (None)
        out_202 = torch.nn.functional.relu(out_201, inplace=True)
        out_201 = None
        out_203 = torch.conv2d(
            out_202,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_202 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_ = (None)
        out_204 = torch.nn.functional.batch_norm(
            out_203,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_203 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_ = (None)
        out_205 = torch.nn.functional.relu(out_204, inplace=True)
        out_204 = None
        out_206 = torch.conv2d(
            out_205,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_205 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_ = (None)
        out_207 = torch.nn.functional.batch_norm(
            out_206,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_206 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_ = (None)
        out_207 += out_199
        out_208 = out_207
        out_207 = out_199 = None
        out_209 = torch.nn.functional.relu(out_208, inplace=True)
        out_208 = None
        out_210 = torch.conv2d(
            out_209,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_ = (
            None
        )
        out_211 = torch.nn.functional.batch_norm(
            out_210,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_210 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_ = (None)
        out_212 = torch.nn.functional.relu(out_211, inplace=True)
        out_211 = None
        out_213 = torch.conv2d(
            out_212,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_212 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_ = (None)
        out_214 = torch.nn.functional.batch_norm(
            out_213,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_213 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_ = (None)
        out_215 = torch.nn.functional.relu(out_214, inplace=True)
        out_214 = None
        out_216 = torch.conv2d(
            out_215,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_215 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_ = (None)
        out_217 = torch.nn.functional.batch_norm(
            out_216,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_216 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_ = (None)
        out_217 += out_209
        out_218 = out_217
        out_217 = out_209 = None
        out_219 = torch.nn.functional.relu(out_218, inplace=True)
        out_218 = None
        out_220 = torch.conv2d(
            out_219,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_ = (
            None
        )
        out_221 = torch.nn.functional.batch_norm(
            out_220,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_220 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_ = (None)
        out_222 = torch.nn.functional.relu(out_221, inplace=True)
        out_221 = None
        out_223 = torch.conv2d(
            out_222,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_222 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_ = (None)
        out_224 = torch.nn.functional.batch_norm(
            out_223,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_223 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_ = (None)
        out_225 = torch.nn.functional.relu(out_224, inplace=True)
        out_224 = None
        out_226 = torch.conv2d(
            out_225,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_225 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_ = (None)
        out_227 = torch.nn.functional.batch_norm(
            out_226,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_226 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_ = (None)
        out_227 += out_219
        out_228 = out_227
        out_227 = out_219 = None
        out_229 = torch.nn.functional.relu(out_228, inplace=True)
        out_228 = None
        out_230 = torch.conv2d(
            out_229,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_ = (
            None
        )
        out_231 = torch.nn.functional.batch_norm(
            out_230,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_230 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_ = (None)
        out_232 = torch.nn.functional.relu(out_231, inplace=True)
        out_231 = None
        out_233 = torch.conv2d(
            out_232,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_232 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_ = (None)
        out_234 = torch.nn.functional.batch_norm(
            out_233,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_233 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_ = (None)
        out_235 = torch.nn.functional.relu(out_234, inplace=True)
        out_234 = None
        out_236 = torch.conv2d(
            out_235,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_235 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_ = (None)
        out_237 = torch.nn.functional.batch_norm(
            out_236,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_236 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_ = (None)
        out_237 += out_229
        out_238 = out_237
        out_237 = out_229 = None
        out_239 = torch.nn.functional.relu(out_238, inplace=True)
        out_238 = None
        out_240 = torch.conv2d(
            out_239,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_ = (
            None
        )
        out_241 = torch.nn.functional.batch_norm(
            out_240,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_240 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_ = (None)
        out_242 = torch.nn.functional.relu(out_241, inplace=True)
        out_241 = None
        out_243 = torch.conv2d(
            out_242,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_242 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_ = (None)
        out_244 = torch.nn.functional.batch_norm(
            out_243,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_243 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_ = (None)
        out_245 = torch.nn.functional.relu(out_244, inplace=True)
        out_244 = None
        out_246 = torch.conv2d(
            out_245,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_245 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_ = (None)
        out_247 = torch.nn.functional.batch_norm(
            out_246,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_246 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_ = (None)
        out_247 += out_239
        out_248 = out_247
        out_247 = out_239 = None
        out_249 = torch.nn.functional.relu(out_248, inplace=True)
        out_248 = None
        out_250 = torch.conv2d(
            out_249,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_ = (
            None
        )
        out_251 = torch.nn.functional.batch_norm(
            out_250,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_250 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_ = (None)
        out_252 = torch.nn.functional.relu(out_251, inplace=True)
        out_251 = None
        out_253 = torch.conv2d(
            out_252,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_252 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_ = (None)
        out_254 = torch.nn.functional.batch_norm(
            out_253,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_253 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_ = (None)
        out_255 = torch.nn.functional.relu(out_254, inplace=True)
        out_254 = None
        out_256 = torch.conv2d(
            out_255,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_255 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_ = (None)
        out_257 = torch.nn.functional.batch_norm(
            out_256,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_256 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_ = (None)
        out_257 += out_249
        out_258 = out_257
        out_257 = out_249 = None
        out_259 = torch.nn.functional.relu(out_258, inplace=True)
        out_258 = None
        out_260 = torch.conv2d(
            out_259,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_ = (
            None
        )
        out_261 = torch.nn.functional.batch_norm(
            out_260,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_260 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_ = (None)
        out_262 = torch.nn.functional.relu(out_261, inplace=True)
        out_261 = None
        out_263 = torch.conv2d(
            out_262,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_262 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_ = (None)
        out_264 = torch.nn.functional.batch_norm(
            out_263,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_263 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_ = (None)
        out_265 = torch.nn.functional.relu(out_264, inplace=True)
        out_264 = None
        out_266 = torch.conv2d(
            out_265,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_265 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_ = (None)
        out_267 = torch.nn.functional.batch_norm(
            out_266,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_266 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_ = (None)
        out_267 += out_259
        out_268 = out_267
        out_267 = out_259 = None
        out_269 = torch.nn.functional.relu(out_268, inplace=True)
        out_268 = None
        out_270 = torch.conv2d(
            out_269,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_ = (
            None
        )
        out_271 = torch.nn.functional.batch_norm(
            out_270,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_270 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_ = (None)
        out_272 = torch.nn.functional.relu(out_271, inplace=True)
        out_271 = None
        out_273 = torch.conv2d(
            out_272,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_272 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_ = (None)
        out_274 = torch.nn.functional.batch_norm(
            out_273,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_273 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_ = (None)
        out_275 = torch.nn.functional.relu(out_274, inplace=True)
        out_274 = None
        out_276 = torch.conv2d(
            out_275,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_275 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_ = (None)
        out_277 = torch.nn.functional.batch_norm(
            out_276,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_276 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_ = (None)
        out_277 += out_269
        out_278 = out_277
        out_277 = out_269 = None
        out_279 = torch.nn.functional.relu(out_278, inplace=True)
        out_278 = None
        out_280 = torch.conv2d(
            out_279,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_ = (
            None
        )
        out_281 = torch.nn.functional.batch_norm(
            out_280,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_280 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_ = (None)
        out_282 = torch.nn.functional.relu(out_281, inplace=True)
        out_281 = None
        out_283 = torch.conv2d(
            out_282,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_282 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_ = (None)
        out_284 = torch.nn.functional.batch_norm(
            out_283,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_283 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_ = (None)
        out_285 = torch.nn.functional.relu(out_284, inplace=True)
        out_284 = None
        out_286 = torch.conv2d(
            out_285,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_285 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_ = (None)
        out_287 = torch.nn.functional.batch_norm(
            out_286,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_286 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_ = (None)
        out_287 += out_279
        out_288 = out_287
        out_287 = out_279 = None
        out_289 = torch.nn.functional.relu(out_288, inplace=True)
        out_288 = None
        out_290 = torch.conv2d(
            out_289,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_ = (
            None
        )
        out_291 = torch.nn.functional.batch_norm(
            out_290,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_290 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_ = (None)
        out_292 = torch.nn.functional.relu(out_291, inplace=True)
        out_291 = None
        out_293 = torch.conv2d(
            out_292,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_292 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_ = (None)
        out_294 = torch.nn.functional.batch_norm(
            out_293,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_293 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_ = (None)
        out_295 = torch.nn.functional.relu(out_294, inplace=True)
        out_294 = None
        out_296 = torch.conv2d(
            out_295,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_295 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_ = (None)
        out_297 = torch.nn.functional.batch_norm(
            out_296,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_296 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_ = (None)
        out_297 += out_289
        out_298 = out_297
        out_297 = out_289 = None
        out_299 = torch.nn.functional.relu(out_298, inplace=True)
        out_298 = None
        out_300 = torch.conv2d(
            out_299,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_ = (
            None
        )
        out_301 = torch.nn.functional.batch_norm(
            out_300,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_300 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_ = (None)
        out_302 = torch.nn.functional.relu(out_301, inplace=True)
        out_301 = None
        out_303 = torch.conv2d(
            out_302,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_302 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_ = (None)
        out_304 = torch.nn.functional.batch_norm(
            out_303,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_303 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_ = (None)
        out_305 = torch.nn.functional.relu(out_304, inplace=True)
        out_304 = None
        out_306 = torch.conv2d(
            out_305,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_305 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_ = (None)
        out_307 = torch.nn.functional.batch_norm(
            out_306,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_306 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_ = (None)
        out_307 += out_299
        out_308 = out_307
        out_307 = out_299 = None
        out_309 = torch.nn.functional.relu(out_308, inplace=True)
        out_308 = None
        out_310 = torch.conv2d(
            out_309,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_ = (
            None
        )
        out_311 = torch.nn.functional.batch_norm(
            out_310,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_310 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_ = (None)
        out_312 = torch.nn.functional.relu(out_311, inplace=True)
        out_311 = None
        out_313 = torch.conv2d(
            out_312,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_312 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_ = (None)
        out_314 = torch.nn.functional.batch_norm(
            out_313,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_313 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_ = (None)
        out_315 = torch.nn.functional.relu(out_314, inplace=True)
        out_314 = None
        out_316 = torch.conv2d(
            out_315,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_315 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_ = (None)
        out_317 = torch.nn.functional.batch_norm(
            out_316,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_316 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_ = (None)
        out_317 += out_309
        out_318 = out_317
        out_317 = out_309 = None
        out_319 = torch.nn.functional.relu(out_318, inplace=True)
        out_318 = None
        out_320 = torch.conv2d(
            out_319,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_ = (
            None
        )
        out_321 = torch.nn.functional.batch_norm(
            out_320,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_320 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_ = (None)
        out_322 = torch.nn.functional.relu(out_321, inplace=True)
        out_321 = None
        out_323 = torch.conv2d(
            out_322,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_322 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_ = (None)
        out_324 = torch.nn.functional.batch_norm(
            out_323,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_323 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_ = (None)
        out_325 = torch.nn.functional.relu(out_324, inplace=True)
        out_324 = None
        out_326 = torch.conv2d(
            out_325,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_325 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_ = (None)
        out_327 = torch.nn.functional.batch_norm(
            out_326,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_326 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_ = (None)
        out_327 += out_319
        out_328 = out_327
        out_327 = out_319 = None
        out_329 = torch.nn.functional.relu(out_328, inplace=True)
        out_328 = None
        out_330 = torch.conv2d(
            out_329,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_ = (
            None
        )
        out_331 = torch.nn.functional.batch_norm(
            out_330,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_330 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_ = (None)
        out_332 = torch.nn.functional.relu(out_331, inplace=True)
        out_331 = None
        out_333 = torch.conv2d(
            out_332,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_332 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_ = (None)
        out_334 = torch.nn.functional.batch_norm(
            out_333,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_333 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_ = (None)
        out_335 = torch.nn.functional.relu(out_334, inplace=True)
        out_334 = None
        out_336 = torch.conv2d(
            out_335,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_335 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = (None)
        out_337 = torch.nn.functional.batch_norm(
            out_336,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_336 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = (None)
        out_337 += out_329
        out_338 = out_337
        out_337 = out_329 = None
        out_339 = torch.nn.functional.relu(out_338, inplace=True)
        out_338 = None
        out_340 = torch.conv2d(
            out_339,
            l_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_ = (
            None
        )
        out_341 = torch.nn.functional.batch_norm(
            out_340,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_340 = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_ = (None)
        out_342 = torch.nn.functional.relu(out_341, inplace=True)
        out_341 = None
        out_343 = torch.conv2d(
            out_342,
            l_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_342 = l_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_ = (None)
        out_344 = torch.nn.functional.batch_norm(
            out_343,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_343 = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_ = (None)
        out_345 = torch.nn.functional.relu(out_344, inplace=True)
        out_344 = None
        out_346 = torch.conv2d(
            out_345,
            l_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_345 = l_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_ = (None)
        out_347 = torch.nn.functional.batch_norm(
            out_346,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_346 = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_ = (None)
        out_347 += out_339
        out_348 = out_347
        out_347 = out_339 = None
        out_349 = torch.nn.functional.relu(out_348, inplace=True)
        out_348 = None
        out_350 = torch.conv2d(
            out_349,
            l_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_ = (
            None
        )
        out_351 = torch.nn.functional.batch_norm(
            out_350,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_350 = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_ = (None)
        out_352 = torch.nn.functional.relu(out_351, inplace=True)
        out_351 = None
        out_353 = torch.conv2d(
            out_352,
            l_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_352 = l_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_ = (None)
        out_354 = torch.nn.functional.batch_norm(
            out_353,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_353 = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_ = (None)
        out_355 = torch.nn.functional.relu(out_354, inplace=True)
        out_354 = None
        out_356 = torch.conv2d(
            out_355,
            l_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_355 = l_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_ = (None)
        out_357 = torch.nn.functional.batch_norm(
            out_356,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_356 = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_ = (None)
        out_357 += out_349
        out_358 = out_357
        out_357 = out_349 = None
        out_359 = torch.nn.functional.relu(out_358, inplace=True)
        out_358 = None
        out_360 = torch.conv2d(
            out_359,
            l_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_ = (
            None
        )
        out_361 = torch.nn.functional.batch_norm(
            out_360,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_360 = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_ = (None)
        out_362 = torch.nn.functional.relu(out_361, inplace=True)
        out_361 = None
        out_363 = torch.conv2d(
            out_362,
            l_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_362 = l_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_ = (None)
        out_364 = torch.nn.functional.batch_norm(
            out_363,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_363 = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_ = (None)
        out_365 = torch.nn.functional.relu(out_364, inplace=True)
        out_364 = None
        out_366 = torch.conv2d(
            out_365,
            l_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_365 = l_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_ = (None)
        out_367 = torch.nn.functional.batch_norm(
            out_366,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_366 = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_ = (None)
        out_367 += out_359
        out_368 = out_367
        out_367 = out_359 = None
        out_369 = torch.nn.functional.relu(out_368, inplace=True)
        out_368 = None
        out_370 = torch.conv2d(
            out_369,
            l_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_ = (
            None
        )
        out_371 = torch.nn.functional.batch_norm(
            out_370,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_370 = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_ = (None)
        out_372 = torch.nn.functional.relu(out_371, inplace=True)
        out_371 = None
        out_373 = torch.conv2d(
            out_372,
            l_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_372 = l_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_ = (None)
        out_374 = torch.nn.functional.batch_norm(
            out_373,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_373 = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_ = (None)
        out_375 = torch.nn.functional.relu(out_374, inplace=True)
        out_374 = None
        out_376 = torch.conv2d(
            out_375,
            l_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_375 = l_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_ = (None)
        out_377 = torch.nn.functional.batch_norm(
            out_376,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_376 = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_ = (None)
        out_377 += out_369
        out_378 = out_377
        out_377 = out_369 = None
        out_379 = torch.nn.functional.relu(out_378, inplace=True)
        out_378 = None
        out_380 = torch.conv2d(
            out_379,
            l_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_ = (
            None
        )
        out_381 = torch.nn.functional.batch_norm(
            out_380,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_380 = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_ = (None)
        out_382 = torch.nn.functional.relu(out_381, inplace=True)
        out_381 = None
        out_383 = torch.conv2d(
            out_382,
            l_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_382 = l_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_ = (None)
        out_384 = torch.nn.functional.batch_norm(
            out_383,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_383 = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_ = (None)
        out_385 = torch.nn.functional.relu(out_384, inplace=True)
        out_384 = None
        out_386 = torch.conv2d(
            out_385,
            l_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_385 = l_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_ = (None)
        out_387 = torch.nn.functional.batch_norm(
            out_386,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_386 = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_ = (None)
        out_387 += out_379
        out_388 = out_387
        out_387 = out_379 = None
        out_389 = torch.nn.functional.relu(out_388, inplace=True)
        out_388 = None
        out_390 = torch.conv2d(
            out_389,
            l_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_ = (
            None
        )
        out_391 = torch.nn.functional.batch_norm(
            out_390,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_390 = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_ = (None)
        out_392 = torch.nn.functional.relu(out_391, inplace=True)
        out_391 = None
        out_393 = torch.conv2d(
            out_392,
            l_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_392 = l_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_ = (None)
        out_394 = torch.nn.functional.batch_norm(
            out_393,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_393 = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_ = (None)
        out_395 = torch.nn.functional.relu(out_394, inplace=True)
        out_394 = None
        out_396 = torch.conv2d(
            out_395,
            l_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_395 = l_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_ = (None)
        out_397 = torch.nn.functional.batch_norm(
            out_396,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_396 = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_ = (None)
        out_397 += out_389
        out_398 = out_397
        out_397 = out_389 = None
        out_399 = torch.nn.functional.relu(out_398, inplace=True)
        out_398 = None
        out_400 = torch.conv2d(
            out_399,
            l_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_ = (
            None
        )
        out_401 = torch.nn.functional.batch_norm(
            out_400,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_400 = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_ = (None)
        out_402 = torch.nn.functional.relu(out_401, inplace=True)
        out_401 = None
        out_403 = torch.conv2d(
            out_402,
            l_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_402 = l_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_ = (None)
        out_404 = torch.nn.functional.batch_norm(
            out_403,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_403 = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_ = (None)
        out_405 = torch.nn.functional.relu(out_404, inplace=True)
        out_404 = None
        out_406 = torch.conv2d(
            out_405,
            l_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_405 = l_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_ = (None)
        out_407 = torch.nn.functional.batch_norm(
            out_406,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_406 = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_ = (None)
        out_407 += out_399
        out_408 = out_407
        out_407 = out_399 = None
        out_409 = torch.nn.functional.relu(out_408, inplace=True)
        out_408 = None
        out_410 = torch.conv2d(
            out_409,
            l_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_ = (
            None
        )
        out_411 = torch.nn.functional.batch_norm(
            out_410,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_410 = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_ = (None)
        out_412 = torch.nn.functional.relu(out_411, inplace=True)
        out_411 = None
        out_413 = torch.conv2d(
            out_412,
            l_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_412 = l_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_ = (None)
        out_414 = torch.nn.functional.batch_norm(
            out_413,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_413 = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_ = (None)
        out_415 = torch.nn.functional.relu(out_414, inplace=True)
        out_414 = None
        out_416 = torch.conv2d(
            out_415,
            l_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_415 = l_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_ = (None)
        out_417 = torch.nn.functional.batch_norm(
            out_416,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_416 = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_ = (None)
        out_417 += out_409
        out_418 = out_417
        out_417 = out_409 = None
        out_419 = torch.nn.functional.relu(out_418, inplace=True)
        out_418 = None
        out_420 = torch.conv2d(
            out_419,
            l_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_ = (
            None
        )
        out_421 = torch.nn.functional.batch_norm(
            out_420,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_420 = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_ = (None)
        out_422 = torch.nn.functional.relu(out_421, inplace=True)
        out_421 = None
        out_423 = torch.conv2d(
            out_422,
            l_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_422 = l_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_ = (None)
        out_424 = torch.nn.functional.batch_norm(
            out_423,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_423 = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_ = (None)
        out_425 = torch.nn.functional.relu(out_424, inplace=True)
        out_424 = None
        out_426 = torch.conv2d(
            out_425,
            l_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_425 = l_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_ = (None)
        out_427 = torch.nn.functional.batch_norm(
            out_426,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_426 = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_ = (None)
        out_427 += out_419
        out_428 = out_427
        out_427 = out_419 = None
        out_429 = torch.nn.functional.relu(out_428, inplace=True)
        out_428 = None
        out_430 = torch.conv2d(
            out_429,
            l_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_ = (
            None
        )
        out_431 = torch.nn.functional.batch_norm(
            out_430,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_430 = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_ = (None)
        out_432 = torch.nn.functional.relu(out_431, inplace=True)
        out_431 = None
        out_433 = torch.conv2d(
            out_432,
            l_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_432 = l_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_ = (None)
        out_434 = torch.nn.functional.batch_norm(
            out_433,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_433 = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_ = (None)
        out_435 = torch.nn.functional.relu(out_434, inplace=True)
        out_434 = None
        out_436 = torch.conv2d(
            out_435,
            l_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_435 = l_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_ = (None)
        out_437 = torch.nn.functional.batch_norm(
            out_436,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_436 = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_ = (None)
        out_437 += out_429
        out_438 = out_437
        out_437 = out_429 = None
        out_439 = torch.nn.functional.relu(out_438, inplace=True)
        out_438 = None
        out_440 = torch.conv2d(
            out_439,
            l_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_ = (
            None
        )
        out_441 = torch.nn.functional.batch_norm(
            out_440,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_440 = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_ = (None)
        out_442 = torch.nn.functional.relu(out_441, inplace=True)
        out_441 = None
        out_443 = torch.conv2d(
            out_442,
            l_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_442 = l_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_ = (None)
        out_444 = torch.nn.functional.batch_norm(
            out_443,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_443 = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_ = (None)
        out_445 = torch.nn.functional.relu(out_444, inplace=True)
        out_444 = None
        out_446 = torch.conv2d(
            out_445,
            l_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_445 = l_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_ = (None)
        out_447 = torch.nn.functional.batch_norm(
            out_446,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_446 = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_ = (None)
        out_447 += out_439
        out_448 = out_447
        out_447 = out_439 = None
        out_449 = torch.nn.functional.relu(out_448, inplace=True)
        out_448 = None
        out_450 = torch.conv2d(
            out_449,
            l_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_ = (
            None
        )
        out_451 = torch.nn.functional.batch_norm(
            out_450,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_450 = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_ = (None)
        out_452 = torch.nn.functional.relu(out_451, inplace=True)
        out_451 = None
        out_453 = torch.conv2d(
            out_452,
            l_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_452 = l_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_ = (None)
        out_454 = torch.nn.functional.batch_norm(
            out_453,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_453 = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_ = (None)
        out_455 = torch.nn.functional.relu(out_454, inplace=True)
        out_454 = None
        out_456 = torch.conv2d(
            out_455,
            l_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_455 = l_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_ = (None)
        out_457 = torch.nn.functional.batch_norm(
            out_456,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_456 = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_ = (None)
        out_457 += out_449
        out_458 = out_457
        out_457 = out_449 = None
        out_459 = torch.nn.functional.relu(out_458, inplace=True)
        out_458 = None
        out_460 = torch.conv2d(
            out_459,
            l_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_ = (
            None
        )
        out_461 = torch.nn.functional.batch_norm(
            out_460,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_460 = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_ = (None)
        out_462 = torch.nn.functional.relu(out_461, inplace=True)
        out_461 = None
        out_463 = torch.conv2d(
            out_462,
            l_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_462 = l_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_ = (None)
        out_464 = torch.nn.functional.batch_norm(
            out_463,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_463 = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_ = (None)
        out_465 = torch.nn.functional.relu(out_464, inplace=True)
        out_464 = None
        out_466 = torch.conv2d(
            out_465,
            l_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_465 = l_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_ = (None)
        out_467 = torch.nn.functional.batch_norm(
            out_466,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_466 = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_ = (None)
        out_467 += out_459
        out_468 = out_467
        out_467 = out_459 = None
        out_469 = torch.nn.functional.relu(out_468, inplace=True)
        out_468 = None
        out_470 = torch.conv2d(
            out_469,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_471 = torch.nn.functional.batch_norm(
            out_470,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_470 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_472 = torch.nn.functional.relu(out_471, inplace=True)
        out_471 = None
        out_473 = torch.conv2d(
            out_472,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_472 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (None)
        out_474 = torch.nn.functional.batch_norm(
            out_473,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_473 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        out_475 = torch.nn.functional.relu(out_474, inplace=True)
        out_474 = None
        out_476 = torch.conv2d(
            out_475,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_475 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_477 = torch.nn.functional.batch_norm(
            out_476,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_476 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_7 = torch.conv2d(
            out_469,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_469 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_477 += input_8
        out_478 = out_477
        out_477 = input_8 = None
        out_479 = torch.nn.functional.relu(out_478, inplace=True)
        out_478 = None
        out_480 = torch.conv2d(
            out_479,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_481 = torch.nn.functional.batch_norm(
            out_480,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_480 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_482 = torch.nn.functional.relu(out_481, inplace=True)
        out_481 = None
        out_483 = torch.conv2d(
            out_482,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_482 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (None)
        out_484 = torch.nn.functional.batch_norm(
            out_483,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_483 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_485 = torch.nn.functional.relu(out_484, inplace=True)
        out_484 = None
        out_486 = torch.conv2d(
            out_485,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_485 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_487 = torch.nn.functional.batch_norm(
            out_486,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_486 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_487 += out_479
        out_488 = out_487
        out_487 = out_479 = None
        out_489 = torch.nn.functional.relu(out_488, inplace=True)
        out_488 = None
        out_490 = torch.conv2d(
            out_489,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_491 = torch.nn.functional.batch_norm(
            out_490,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_490 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_492 = torch.nn.functional.relu(out_491, inplace=True)
        out_491 = None
        out_493 = torch.conv2d(
            out_492,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_492 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (None)
        out_494 = torch.nn.functional.batch_norm(
            out_493,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_493 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (None)
        out_495 = torch.nn.functional.relu(out_494, inplace=True)
        out_494 = None
        out_496 = torch.conv2d(
            out_495,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_495 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_497 = torch.nn.functional.batch_norm(
            out_496,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_496 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_497 += out_489
        out_498 = out_497
        out_497 = out_489 = None
        out_499 = torch.nn.functional.relu(out_498, inplace=True)
        out_498 = None
        input_9 = torch.conv_transpose2d(
            out_499,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        out_499 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv_transpose2d(
            input_11,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_11 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_14 = torch.nn.functional.relu(input_13, inplace=True)
        input_13 = None
        input_15 = torch.conv_transpose2d(
            input_14,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_14 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        x_4 = torch.conv2d(
            input_17,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_4,)
