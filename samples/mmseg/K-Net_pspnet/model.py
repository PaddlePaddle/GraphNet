import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_backbone_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_backbone_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_0_parameters_weight_
        )
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_stem_modules_1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_stem_modules_1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_stem_modules_1_buffers_running_var_ = (
            L_self_modules_backbone_modules_stem_modules_1_buffers_running_var_
        )
        l_self_modules_backbone_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_1_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_1_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_3_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_3_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_4_buffers_running_mean_ = (
            L_self_modules_backbone_modules_stem_modules_4_buffers_running_mean_
        )
        l_self_modules_backbone_modules_stem_modules_4_buffers_running_var_ = (
            L_self_modules_backbone_modules_stem_modules_4_buffers_running_var_
        )
        l_self_modules_backbone_modules_stem_modules_4_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_4_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_4_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_4_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_6_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_6_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_7_buffers_running_mean_ = (
            L_self_modules_backbone_modules_stem_modules_7_buffers_running_mean_
        )
        l_self_modules_backbone_modules_stem_modules_7_buffers_running_var_ = (
            L_self_modules_backbone_modules_stem_modules_7_buffers_running_var_
        )
        l_self_modules_backbone_modules_stem_modules_7_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_7_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_7_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_7_parameters_bias_
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
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_weight_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_weight_
        l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_bias_ = L_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_0_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_0_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_bias_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_weight_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_weight_
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_bias_ = L_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_bias_
        input_1 = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_stem_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = (
            l_self_modules_backbone_modules_stem_modules_0_parameters_weight_
        ) = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_backbone_modules_stem_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_backbone_modules_stem_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_stem_modules_1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_backbone_modules_stem_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_stem_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_backbone_modules_stem_modules_3_parameters_weight_
        ) = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_backbone_modules_stem_modules_4_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_backbone_modules_stem_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_stem_modules_4_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_stem_modules_4_parameters_weight_
        ) = l_self_modules_backbone_modules_stem_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_stem_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = (
            l_self_modules_backbone_modules_stem_modules_6_parameters_weight_
        ) = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_stem_modules_7_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_7_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_backbone_modules_stem_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_stem_modules_7_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_stem_modules_7_parameters_weight_
        ) = l_self_modules_backbone_modules_stem_modules_7_parameters_bias_ = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        x = torch.nn.functional.max_pool2d(
            input_9, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_9 = None
        out = torch.conv2d(
            x,
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
        input_10 = torch.conv2d(
            x,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_7 += input_11
        out_8 = out_7
        out_7 = input_11 = None
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
        input_12 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_29 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_37 += input_13
        out_38 = out_37
        out_37 = input_13 = None
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
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_72 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        out_73 = torch.conv2d(
            out_72,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_72 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (None)
        out_74 = torch.nn.functional.batch_norm(
            out_73,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_73 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        out_75 = torch.nn.functional.relu(out_74, inplace=True)
        out_74 = None
        out_76 = torch.conv2d(
            out_75,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_75 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_77 = torch.nn.functional.batch_norm(
            out_76,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_76 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_14 = torch.conv2d(
            out_69,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_69 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_77 += input_15
        out_78 = out_77
        out_77 = input_15 = None
        out_79 = torch.nn.functional.relu(out_78, inplace=True)
        out_78 = None
        out_80 = torch.conv2d(
            out_79,
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
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_82 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        out_83 = torch.conv2d(
            out_82,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        out_82 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (None)
        out_84 = torch.nn.functional.batch_norm(
            out_83,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_83 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_85 = torch.nn.functional.relu(out_84, inplace=True)
        out_84 = None
        out_86 = torch.conv2d(
            out_85,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_85 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_87 = torch.nn.functional.batch_norm(
            out_86,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_86 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        out_87 += out_79
        out_88 = out_87
        out_87 = out_79 = None
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_90 = torch.conv2d(
            out_89,
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
        out_91 = torch.nn.functional.batch_norm(
            out_90,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_90 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_92 = torch.nn.functional.relu(out_91, inplace=True)
        out_91 = None
        out_93 = torch.conv2d(
            out_92,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        out_92 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (None)
        out_94 = torch.nn.functional.batch_norm(
            out_93,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_93 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (None)
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        out_96 = torch.conv2d(
            out_95,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_95 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_97 = torch.nn.functional.batch_norm(
            out_96,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_96 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        out_97 += out_89
        out_98 = out_97
        out_97 = out_89 = None
        out_99 = torch.nn.functional.relu(out_98, inplace=True)
        out_98 = None
        out_100 = torch.conv2d(
            out_99,
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
        out_101 = torch.nn.functional.batch_norm(
            out_100,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_100 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_102 = torch.nn.functional.relu(out_101, inplace=True)
        out_101 = None
        out_103 = torch.conv2d(
            out_102,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        out_102 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (None)
        out_104 = torch.nn.functional.batch_norm(
            out_103,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_103 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (None)
        out_105 = torch.nn.functional.relu(out_104, inplace=True)
        out_104 = None
        out_106 = torch.conv2d(
            out_105,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_107 = torch.nn.functional.batch_norm(
            out_106,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_106 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        out_107 += out_99
        out_108 = out_107
        out_107 = out_99 = None
        out_109 = torch.nn.functional.relu(out_108, inplace=True)
        out_108 = None
        out_110 = torch.conv2d(
            out_109,
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
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_112 = torch.nn.functional.relu(out_111, inplace=True)
        out_111 = None
        out_113 = torch.conv2d(
            out_112,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        out_112 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (None)
        out_114 = torch.nn.functional.batch_norm(
            out_113,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_113 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (None)
        out_115 = torch.nn.functional.relu(out_114, inplace=True)
        out_114 = None
        out_116 = torch.conv2d(
            out_115,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_115 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_117 = torch.nn.functional.batch_norm(
            out_116,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_116 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        out_117 += out_109
        out_118 = out_117
        out_117 = out_109 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        out_120 = torch.conv2d(
            out_119,
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
        out_121 = torch.nn.functional.batch_norm(
            out_120,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_120 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_122 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        out_123 = torch.conv2d(
            out_122,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        out_122 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (None)
        out_124 = torch.nn.functional.batch_norm(
            out_123,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_123 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (None)
        out_125 = torch.nn.functional.relu(out_124, inplace=True)
        out_124 = None
        out_126 = torch.conv2d(
            out_125,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_125 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_127 = torch.nn.functional.batch_norm(
            out_126,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_126 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        out_127 += out_119
        out_128 = out_127
        out_127 = out_119 = None
        out_129 = torch.nn.functional.relu(out_128, inplace=True)
        out_128 = None
        out_130 = torch.conv2d(
            out_129,
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
        out_131 = torch.nn.functional.batch_norm(
            out_130,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_130 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_132 = torch.nn.functional.relu(out_131, inplace=True)
        out_131 = None
        out_133 = torch.conv2d(
            out_132,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        out_132 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (None)
        out_134 = torch.nn.functional.batch_norm(
            out_133,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_133 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        out_135 = torch.nn.functional.relu(out_134, inplace=True)
        out_134 = None
        out_136 = torch.conv2d(
            out_135,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_135 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_137 = torch.nn.functional.batch_norm(
            out_136,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_136 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_16 = torch.conv2d(
            out_129,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_129 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_137 += input_17
        out_138 = out_137
        out_137 = input_17 = None
        out_139 = torch.nn.functional.relu(out_138, inplace=True)
        out_138 = None
        out_140 = torch.conv2d(
            out_139,
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
        out_141 = torch.nn.functional.batch_norm(
            out_140,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_140 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_142 = torch.nn.functional.relu(out_141, inplace=True)
        out_141 = None
        out_143 = torch.conv2d(
            out_142,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        out_142 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (None)
        out_144 = torch.nn.functional.batch_norm(
            out_143,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_143 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_145 = torch.nn.functional.relu(out_144, inplace=True)
        out_144 = None
        out_146 = torch.conv2d(
            out_145,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_145 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_147 = torch.nn.functional.batch_norm(
            out_146,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_146 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_147 += out_139
        out_148 = out_147
        out_147 = out_139 = None
        out_149 = torch.nn.functional.relu(out_148, inplace=True)
        out_148 = None
        out_150 = torch.conv2d(
            out_149,
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
        out_151 = torch.nn.functional.batch_norm(
            out_150,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_150 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_152 = torch.nn.functional.relu(out_151, inplace=True)
        out_151 = None
        out_153 = torch.conv2d(
            out_152,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        out_152 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (None)
        out_154 = torch.nn.functional.batch_norm(
            out_153,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_153 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (None)
        out_155 = torch.nn.functional.relu(out_154, inplace=True)
        out_154 = None
        out_156 = torch.conv2d(
            out_155,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_155 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_157 = torch.nn.functional.batch_norm(
            out_156,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_156 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_157 += out_149
        out_158 = out_157
        out_157 = out_149 = None
        out_159 = torch.nn.functional.relu(out_158, inplace=True)
        out_158 = None
        input_18 = torch.nn.functional.adaptive_avg_pool2d(out_159, 1)
        x_1 = torch.conv2d(
            input_18,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_18 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_2 = torch.nn.functional.batch_norm(
            x_1,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_1 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_3 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_3, (64, 64), None, "bilinear", False
        )
        x_3 = None
        input_19 = torch.nn.functional.adaptive_avg_pool2d(out_159, 2)
        x_4 = torch.conv2d(
            input_19,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_6, (64, 64), None, "bilinear", False
        )
        x_6 = None
        input_20 = torch.nn.functional.adaptive_avg_pool2d(out_159, 3)
        x_7 = torch.conv2d(
            input_20,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_9, (64, 64), None, "bilinear", False
        )
        x_9 = None
        input_21 = torch.nn.functional.adaptive_avg_pool2d(out_159, 6)
        x_10 = torch.conv2d(
            input_21,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_12, (64, 64), None, "bilinear", False
        )
        x_12 = None
        psp_outs = torch.cat(
            [
                out_159,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        out_159 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_13 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_kernel_generate_head_modules_bottleneck_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        feat = torch.nn.functional.dropout2d(x_15, 0.1, False, False)
        output = torch.conv2d(
            feat,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat = l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_bias_ = (None)
        seg_kernels = (
            l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_weight_.clone()
        )
        l_self_modules_decode_head_modules_kernel_generate_head_modules_conv_seg_parameters_weight_ = (
            None
        )
        getitem = seg_kernels[None]
        seg_kernels = None
        seg_kernels_1 = getitem.expand(1, 150, 512, 1, 1)
        getitem = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_feat_transform_modules_conv_parameters_bias_ = (None)
        sigmoid_masks = output.softmax(dim=1)
        output = None
        x_feat = torch.functional.einsum("bnhw,bchw->bnc", sigmoid_masks, x_16)
        sigmoid_masks = None
        reshape = seg_kernels_1.reshape(1, 150, 512, -1)
        seg_kernels_1 = None
        proposal_feat = reshape.permute(0, 1, 3, 2)
        reshape = None
        update_feature = x_feat.reshape(-1, 256)
        x_feat = None
        parameters = torch._C._nn.linear(
            update_feature,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_,
        )
        update_feature = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_ = (None)
        getitem_1 = parameters[(slice(None, None, None), slice(None, 256, None))]
        param_in = getitem_1.view(-1, 256)
        getitem_1 = None
        getitem_2 = parameters[(slice(None, None, None), slice(-256, None, None))]
        parameters = None
        param_out = getitem_2.view(-1, 256)
        getitem_2 = None
        reshape_2 = proposal_feat.reshape(300, -1, 256)
        proposal_feat = None
        input_feats = torch._C._nn.linear(
            reshape_2,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_bias_,
        )
        reshape_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_layer_parameters_bias_ = (None)
        input_in = input_feats[(Ellipsis, slice(None, 256, None))]
        input_out = input_feats[(Ellipsis, slice(-256, None, None))]
        input_feats = None
        unsqueeze = param_in.unsqueeze(-2)
        param_in = None
        gate_feats = input_in * unsqueeze
        input_in = unsqueeze = None
        linear_2 = torch._C._nn.linear(
            gate_feats,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_bias_,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_gate_parameters_bias_ = (None)
        input_gate = torch.nn.functional.layer_norm(
            linear_2,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_,
            1e-05,
        )
        linear_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_ = (None)
        linear_3 = torch._C._nn.linear(
            gate_feats,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_bias_,
        )
        gate_feats = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_update_gate_parameters_bias_ = (None)
        update_gate = torch.nn.functional.layer_norm(
            linear_3,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_bias_,
            1e-05,
        )
        linear_3 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_in_parameters_bias_ = (None)
        input_gate_1 = input_gate.sigmoid()
        input_gate = None
        update_gate_1 = update_gate.sigmoid()
        update_gate = None
        param_out_1 = torch.nn.functional.layer_norm(
            param_out,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_bias_,
            1e-05,
        )
        param_out = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_norm_out_parameters_bias_ = (None)
        input_out_1 = torch.nn.functional.layer_norm(
            input_out,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_,
            1e-05,
        )
        input_out = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_ = (None)
        unsqueeze_1 = param_out_1.unsqueeze(-2)
        param_out_1 = None
        mul_1 = update_gate_1 * unsqueeze_1
        update_gate_1 = unsqueeze_1 = None
        mul_2 = input_gate_1 * input_out_1
        input_gate_1 = input_out_1 = None
        features = mul_1 + mul_2
        mul_1 = mul_2 = None
        features_1 = torch._C._nn.linear(
            features,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_bias_,
        )
        features = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_layer_parameters_bias_ = (None)
        features_2 = torch.nn.functional.layer_norm(
            features_1,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_bias_,
            1e-05,
        )
        features_1 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_kernel_update_conv_modules_fc_norm_parameters_bias_ = (None)
        features_3 = torch.nn.functional.relu(features_2, inplace=True)
        features_2 = None
        reshape_3 = features_3.reshape(1, 150, -1)
        features_3 = None
        obj_feat = reshape_3.permute(1, 0, 2)
        reshape_3 = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            obj_feat,
            obj_feat,
            obj_feat,
            512,
            8,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_parameters_in_proj_bias_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        dropout = torch.nn.functional.dropout(attn_output, 0.0, False, False)
        attn_output = None
        dropout_1 = torch.nn.functional.dropout(dropout, 0.0, False, False)
        dropout = None
        output_1 = obj_feat + dropout_1
        obj_feat = dropout_1 = None
        obj_feat_1 = torch.nn.functional.layer_norm(
            output_1,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_bias_,
            1e-05,
        )
        output_1 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_attention_norm_parameters_bias_ = (None)
        obj_feat_2 = obj_feat_1.permute(1, 0, 2)
        obj_feat_1 = None
        obj_feat_3 = obj_feat_2.reshape(1, 150, -1, 512)
        obj_feat_2 = None
        input_22 = torch._C._nn.linear(
            obj_feat_3,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.nn.functional.dropout(input_23, 0.0, False, False)
        input_23 = None
        input_25 = torch._C._nn.linear(
            input_24,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_24 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_26 = torch.nn.functional.dropout(input_25, 0.0, False, False)
        input_25 = None
        output_2 = obj_feat_3 + input_26
        obj_feat_3 = input_26 = None
        obj_feat_4 = torch.nn.functional.layer_norm(
            output_2,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_bias_,
            1e-05,
        )
        output_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_ffn_norm_parameters_bias_ = (None)
        mask_feat = torch._C._nn.linear(
            obj_feat_4,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_0_parameters_weight_,
            None,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_0_parameters_weight_ = (
            None
        )
        mask_feat_1 = torch.nn.functional.layer_norm(
            mask_feat,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_bias_,
            1e-05,
        )
        mask_feat = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_mask_fcs_modules_1_parameters_bias_ = (None)
        mask_feat_2 = torch.nn.functional.relu(mask_feat_1, inplace=True)
        mask_feat_1 = None
        linear_8 = torch._C._nn.linear(
            mask_feat_2,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_bias_,
        )
        mask_feat_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_0_modules_fc_mask_parameters_bias_ = (None)
        mask_feat_3 = linear_8.permute(0, 1, 3, 2)
        linear_8 = None
        mask_feat_4 = mask_feat_3.reshape(1, 150, 512, 1, 1)
        mask_feat_3 = None
        getitem_7 = x_16[slice(0, 1, None)]
        x_16 = None
        getitem_8 = mask_feat_4[0]
        mask_feat_4 = None
        conv2d_62 = torch.conv2d(getitem_7, getitem_8, padding=0)
        getitem_7 = getitem_8 = None
        new_mask_preds = torch.cat([conv2d_62], dim=0)
        conv2d_62 = None
        new_mask_preds_1 = new_mask_preds.reshape(1, 150, 64, 64)
        new_mask_preds = None
        permute_4 = obj_feat_4.permute(0, 1, 3, 2)
        obj_feat_4 = None
        seg_kernels_2 = permute_4.reshape(1, 150, 512, 1, 1)
        permute_4 = None
        x_17 = torch.conv2d(
            x_15,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_feat_transform_modules_conv_parameters_bias_ = (None)
        sigmoid_masks_1 = new_mask_preds_1.softmax(dim=1)
        new_mask_preds_1 = None
        x_feat_1 = torch.functional.einsum("bnhw,bchw->bnc", sigmoid_masks_1, x_17)
        sigmoid_masks_1 = None
        reshape_8 = seg_kernels_2.reshape(1, 150, 512, -1)
        seg_kernels_2 = None
        proposal_feat_1 = reshape_8.permute(0, 1, 3, 2)
        reshape_8 = None
        update_feature_1 = x_feat_1.reshape(-1, 256)
        x_feat_1 = None
        parameters_1 = torch._C._nn.linear(
            update_feature_1,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_,
        )
        update_feature_1 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_ = (None)
        getitem_9 = parameters_1[(slice(None, None, None), slice(None, 256, None))]
        param_in_1 = getitem_9.view(-1, 256)
        getitem_9 = None
        getitem_10 = parameters_1[(slice(None, None, None), slice(-256, None, None))]
        parameters_1 = None
        param_out_2 = getitem_10.view(-1, 256)
        getitem_10 = None
        reshape_10 = proposal_feat_1.reshape(300, -1, 256)
        proposal_feat_1 = None
        input_feats_1 = torch._C._nn.linear(
            reshape_10,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_bias_,
        )
        reshape_10 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_layer_parameters_bias_ = (None)
        input_in_1 = input_feats_1[(Ellipsis, slice(None, 256, None))]
        input_out_2 = input_feats_1[(Ellipsis, slice(-256, None, None))]
        input_feats_1 = None
        unsqueeze_2 = param_in_1.unsqueeze(-2)
        param_in_1 = None
        gate_feats_1 = input_in_1 * unsqueeze_2
        input_in_1 = unsqueeze_2 = None
        linear_11 = torch._C._nn.linear(
            gate_feats_1,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_bias_,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_gate_parameters_bias_ = (None)
        input_gate_2 = torch.nn.functional.layer_norm(
            linear_11,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_,
            1e-05,
        )
        linear_11 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_ = (None)
        linear_12 = torch._C._nn.linear(
            gate_feats_1,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_bias_,
        )
        gate_feats_1 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_update_gate_parameters_bias_ = (None)
        update_gate_2 = torch.nn.functional.layer_norm(
            linear_12,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_bias_,
            1e-05,
        )
        linear_12 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_in_parameters_bias_ = (None)
        input_gate_3 = input_gate_2.sigmoid()
        input_gate_2 = None
        update_gate_3 = update_gate_2.sigmoid()
        update_gate_2 = None
        param_out_3 = torch.nn.functional.layer_norm(
            param_out_2,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_bias_,
            1e-05,
        )
        param_out_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_norm_out_parameters_bias_ = (None)
        input_out_3 = torch.nn.functional.layer_norm(
            input_out_2,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_,
            1e-05,
        )
        input_out_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_ = (None)
        unsqueeze_3 = param_out_3.unsqueeze(-2)
        param_out_3 = None
        mul_4 = update_gate_3 * unsqueeze_3
        update_gate_3 = unsqueeze_3 = None
        mul_5 = input_gate_3 * input_out_3
        input_gate_3 = input_out_3 = None
        features_4 = mul_4 + mul_5
        mul_4 = mul_5 = None
        features_5 = torch._C._nn.linear(
            features_4,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_bias_,
        )
        features_4 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_layer_parameters_bias_ = (None)
        features_6 = torch.nn.functional.layer_norm(
            features_5,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_bias_,
            1e-05,
        )
        features_5 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_kernel_update_conv_modules_fc_norm_parameters_bias_ = (None)
        features_7 = torch.nn.functional.relu(features_6, inplace=True)
        features_6 = None
        reshape_11 = features_7.reshape(1, 150, -1)
        features_7 = None
        obj_feat_5 = reshape_11.permute(1, 0, 2)
        reshape_11 = None
        multi_head_attention_forward_1 = torch.nn.functional.multi_head_attention_forward(
            obj_feat_5,
            obj_feat_5,
            obj_feat_5,
            512,
            8,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_parameters_in_proj_bias_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_1 = multi_head_attention_forward_1[0]
        multi_head_attention_forward_1 = None
        dropout_4 = torch.nn.functional.dropout(attn_output_1, 0.0, False, False)
        attn_output_1 = None
        dropout_5 = torch.nn.functional.dropout(dropout_4, 0.0, False, False)
        dropout_4 = None
        output_3 = obj_feat_5 + dropout_5
        obj_feat_5 = dropout_5 = None
        obj_feat_6 = torch.nn.functional.layer_norm(
            output_3,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_bias_,
            1e-05,
        )
        output_3 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_attention_norm_parameters_bias_ = (None)
        obj_feat_7 = obj_feat_6.permute(1, 0, 2)
        obj_feat_6 = None
        obj_feat_8 = obj_feat_7.reshape(1, 150, -1, 512)
        obj_feat_7 = None
        input_27 = torch._C._nn.linear(
            obj_feat_8,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_28 = torch.nn.functional.relu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.nn.functional.dropout(input_28, 0.0, False, False)
        input_28 = None
        input_30 = torch._C._nn.linear(
            input_29,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_29 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_31 = torch.nn.functional.dropout(input_30, 0.0, False, False)
        input_30 = None
        output_4 = obj_feat_8 + input_31
        obj_feat_8 = input_31 = None
        obj_feat_9 = torch.nn.functional.layer_norm(
            output_4,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_bias_,
            1e-05,
        )
        output_4 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_ffn_norm_parameters_bias_ = (None)
        mask_feat_5 = torch._C._nn.linear(
            obj_feat_9,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_0_parameters_weight_,
            None,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_0_parameters_weight_ = (
            None
        )
        mask_feat_6 = torch.nn.functional.layer_norm(
            mask_feat_5,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_bias_,
            1e-05,
        )
        mask_feat_5 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_mask_fcs_modules_1_parameters_bias_ = (None)
        mask_feat_7 = torch.nn.functional.relu(mask_feat_6, inplace=True)
        mask_feat_6 = None
        linear_17 = torch._C._nn.linear(
            mask_feat_7,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_bias_,
        )
        mask_feat_7 = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_1_modules_fc_mask_parameters_bias_ = (None)
        mask_feat_8 = linear_17.permute(0, 1, 3, 2)
        linear_17 = None
        mask_feat_9 = mask_feat_8.reshape(1, 150, 512, 1, 1)
        mask_feat_8 = None
        getitem_15 = x_17[slice(0, 1, None)]
        x_17 = None
        getitem_16 = mask_feat_9[0]
        mask_feat_9 = None
        conv2d_64 = torch.conv2d(getitem_15, getitem_16, padding=0)
        getitem_15 = getitem_16 = None
        new_mask_preds_2 = torch.cat([conv2d_64], dim=0)
        conv2d_64 = None
        new_mask_preds_3 = new_mask_preds_2.reshape(1, 150, 64, 64)
        new_mask_preds_2 = None
        permute_9 = obj_feat_9.permute(0, 1, 3, 2)
        obj_feat_9 = None
        seg_kernels_3 = permute_9.reshape(1, 150, 512, 1, 1)
        permute_9 = None
        x_18 = torch.conv2d(
            x_15,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_feat_transform_modules_conv_parameters_bias_ = (None)
        sigmoid_masks_2 = new_mask_preds_3.softmax(dim=1)
        new_mask_preds_3 = None
        x_feat_2 = torch.functional.einsum("bnhw,bchw->bnc", sigmoid_masks_2, x_18)
        sigmoid_masks_2 = None
        reshape_16 = seg_kernels_3.reshape(1, 150, 512, -1)
        seg_kernels_3 = None
        proposal_feat_2 = reshape_16.permute(0, 1, 3, 2)
        reshape_16 = None
        update_feature_2 = x_feat_2.reshape(-1, 256)
        x_feat_2 = None
        parameters_2 = torch._C._nn.linear(
            update_feature_2,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_,
        )
        update_feature_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_dynamic_layer_parameters_bias_ = (None)
        getitem_17 = parameters_2[(slice(None, None, None), slice(None, 256, None))]
        param_in_2 = getitem_17.view(-1, 256)
        getitem_17 = None
        getitem_18 = parameters_2[(slice(None, None, None), slice(-256, None, None))]
        parameters_2 = None
        param_out_4 = getitem_18.view(-1, 256)
        getitem_18 = None
        reshape_18 = proposal_feat_2.reshape(300, -1, 256)
        proposal_feat_2 = None
        input_feats_2 = torch._C._nn.linear(
            reshape_18,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_bias_,
        )
        reshape_18 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_layer_parameters_bias_ = (None)
        input_in_2 = input_feats_2[(Ellipsis, slice(None, 256, None))]
        input_out_4 = input_feats_2[(Ellipsis, slice(-256, None, None))]
        input_feats_2 = None
        unsqueeze_4 = param_in_2.unsqueeze(-2)
        param_in_2 = None
        gate_feats_2 = input_in_2 * unsqueeze_4
        input_in_2 = unsqueeze_4 = None
        linear_20 = torch._C._nn.linear(
            gate_feats_2,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_bias_,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_gate_parameters_bias_ = (None)
        input_gate_4 = torch.nn.functional.layer_norm(
            linear_20,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_,
            1e-05,
        )
        linear_20 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_in_parameters_bias_ = (None)
        linear_21 = torch._C._nn.linear(
            gate_feats_2,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_bias_,
        )
        gate_feats_2 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_update_gate_parameters_bias_ = (None)
        update_gate_4 = torch.nn.functional.layer_norm(
            linear_21,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_bias_,
            1e-05,
        )
        linear_21 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_in_parameters_bias_ = (None)
        input_gate_5 = input_gate_4.sigmoid()
        input_gate_4 = None
        update_gate_5 = update_gate_4.sigmoid()
        update_gate_4 = None
        param_out_5 = torch.nn.functional.layer_norm(
            param_out_4,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_bias_,
            1e-05,
        )
        param_out_4 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_norm_out_parameters_bias_ = (None)
        input_out_5 = torch.nn.functional.layer_norm(
            input_out_4,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_,
            1e-05,
        )
        input_out_4 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_input_norm_out_parameters_bias_ = (None)
        unsqueeze_5 = param_out_5.unsqueeze(-2)
        param_out_5 = None
        mul_7 = update_gate_5 * unsqueeze_5
        update_gate_5 = unsqueeze_5 = None
        mul_8 = input_gate_5 * input_out_5
        input_gate_5 = input_out_5 = None
        features_8 = mul_7 + mul_8
        mul_7 = mul_8 = None
        features_9 = torch._C._nn.linear(
            features_8,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_bias_,
        )
        features_8 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_layer_parameters_bias_ = (None)
        features_10 = torch.nn.functional.layer_norm(
            features_9,
            (256,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_bias_,
            1e-05,
        )
        features_9 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_kernel_update_conv_modules_fc_norm_parameters_bias_ = (None)
        features_11 = torch.nn.functional.relu(features_10, inplace=True)
        features_10 = None
        reshape_19 = features_11.reshape(1, 150, -1)
        features_11 = None
        obj_feat_10 = reshape_19.permute(1, 0, 2)
        reshape_19 = None
        multi_head_attention_forward_2 = torch.nn.functional.multi_head_attention_forward(
            obj_feat_10,
            obj_feat_10,
            obj_feat_10,
            512,
            8,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            0.0,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_parameters_in_proj_bias_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_modules_attn_modules_out_proj_parameters_bias_ = (None)
        attn_output_2 = multi_head_attention_forward_2[0]
        multi_head_attention_forward_2 = None
        dropout_8 = torch.nn.functional.dropout(attn_output_2, 0.0, False, False)
        attn_output_2 = None
        dropout_9 = torch.nn.functional.dropout(dropout_8, 0.0, False, False)
        dropout_8 = None
        output_5 = obj_feat_10 + dropout_9
        obj_feat_10 = dropout_9 = None
        obj_feat_11 = torch.nn.functional.layer_norm(
            output_5,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_bias_,
            1e-05,
        )
        output_5 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_attention_norm_parameters_bias_ = (None)
        obj_feat_12 = obj_feat_11.permute(1, 0, 2)
        obj_feat_11 = None
        obj_feat_13 = obj_feat_12.reshape(1, 150, -1, 512)
        obj_feat_12 = None
        input_32 = torch._C._nn.linear(
            obj_feat_13,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_ = (None)
        input_33 = torch.nn.functional.relu(input_32, inplace=True)
        input_32 = None
        input_34 = torch.nn.functional.dropout(input_33, 0.0, False, False)
        input_33 = None
        input_35 = torch._C._nn.linear(
            input_34,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_,
        )
        input_34 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_ = (None)
        input_36 = torch.nn.functional.dropout(input_35, 0.0, False, False)
        input_35 = None
        output_6 = obj_feat_13 + input_36
        obj_feat_13 = input_36 = None
        obj_feat_14 = torch.nn.functional.layer_norm(
            output_6,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_bias_,
            1e-05,
        )
        output_6 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_ffn_norm_parameters_bias_ = (None)
        mask_feat_10 = torch._C._nn.linear(
            obj_feat_14,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_0_parameters_weight_,
            None,
        )
        l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_0_parameters_weight_ = (
            None
        )
        mask_feat_11 = torch.nn.functional.layer_norm(
            mask_feat_10,
            (512,),
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_bias_,
            1e-05,
        )
        mask_feat_10 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_mask_fcs_modules_1_parameters_bias_ = (None)
        mask_feat_12 = torch.nn.functional.relu(mask_feat_11, inplace=True)
        mask_feat_11 = None
        linear_26 = torch._C._nn.linear(
            mask_feat_12,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_weight_,
            l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_bias_,
        )
        mask_feat_12 = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_weight_ = l_self_modules_decode_head_modules_kernel_update_head_modules_2_modules_fc_mask_parameters_bias_ = (None)
        mask_feat_13 = linear_26.permute(0, 1, 3, 2)
        linear_26 = None
        mask_feat_14 = mask_feat_13.reshape(1, 150, 512, 1, 1)
        mask_feat_13 = None
        getitem_23 = x_18[slice(0, 1, None)]
        x_18 = None
        getitem_24 = mask_feat_14[0]
        mask_feat_14 = None
        conv2d_66 = torch.conv2d(getitem_23, getitem_24, padding=0)
        getitem_23 = getitem_24 = None
        new_mask_preds_4 = torch.cat([conv2d_66], dim=0)
        conv2d_66 = None
        new_mask_preds_5 = new_mask_preds_4.reshape(1, 150, 64, 64)
        new_mask_preds_4 = None
        permute_14 = obj_feat_14.permute(0, 1, 3, 2)
        obj_feat_14 = None
        seg_kernels_4 = permute_14.reshape(1, 150, 512, 1, 1)
        permute_14 = seg_kernels_4 = None
        return (new_mask_preds_5,)
