import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_
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
            l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = (None)
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        out = torch.conv2d(
            x_9,
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
        x_10 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        splits = x_12.view(1, 2, -1, 64, 48)
        x_12 = None
        gap = splits.sum(dim=1)
        gap_1 = torch.nn.functional.adaptive_avg_pool2d(gap, 1)
        gap = None
        gap_2 = torch.conv2d(
            gap_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_3 = torch.nn.functional.batch_norm(
            gap_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_4 = torch.nn.functional.relu(gap_3, inplace=True)
        gap_3 = None
        atten = torch.conv2d(
            gap_4,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_4 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_1 = atten.view(1, 1, 2, -1)
        atten = None
        x_13 = view_1.transpose(1, 2)
        view_1 = None
        x_14 = torch.nn.functional.softmax(x_13, dim=1)
        x_13 = None
        x_15 = x_14.reshape(1, -1)
        x_14 = None
        atten_1 = x_15.view(1, -1, 1, 1)
        x_15 = None
        attens = atten_1.view(1, 2, -1, 1, 1)
        atten_1 = None
        mul = attens * splits
        attens = splits = None
        out_3 = torch.sum(mul, dim=1)
        mul = None
        out_4 = out_3.contiguous()
        out_3 = None
        out_5 = torch.conv2d(
            out_4,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_4 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (None)
        out_6 = torch.nn.functional.batch_norm(
            out_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_1 = torch.conv2d(
            x_9,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        out_6 += input_2
        out_7 = out_6
        out_6 = input_2 = None
        out_8 = torch.nn.functional.relu(out_7, inplace=True)
        out_7 = None
        out_9 = torch.conv2d(
            out_8,
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
        out_10 = torch.nn.functional.batch_norm(
            out_9,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_9 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        x_16 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_11 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        splits_1 = x_18.view(1, 2, -1, 64, 48)
        x_18 = None
        gap_5 = splits_1.sum(dim=1)
        gap_6 = torch.nn.functional.adaptive_avg_pool2d(gap_5, 1)
        gap_5 = None
        gap_7 = torch.conv2d(
            gap_6,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_6 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_8 = torch.nn.functional.batch_norm(
            gap_7,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_7 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_9 = torch.nn.functional.relu(gap_8, inplace=True)
        gap_8 = None
        atten_2 = torch.conv2d(
            gap_9,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_9 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_5 = atten_2.view(1, 1, 2, -1)
        atten_2 = None
        x_19 = view_5.transpose(1, 2)
        view_5 = None
        x_20 = torch.nn.functional.softmax(x_19, dim=1)
        x_19 = None
        x_21 = x_20.reshape(1, -1)
        x_20 = None
        atten_3 = x_21.view(1, -1, 1, 1)
        x_21 = None
        attens_1 = atten_3.view(1, 2, -1, 1, 1)
        atten_3 = None
        mul_1 = attens_1 * splits_1
        attens_1 = splits_1 = None
        out_12 = torch.sum(mul_1, dim=1)
        mul_1 = None
        out_13 = out_12.contiguous()
        out_12 = None
        out_14 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (None)
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (None)
        out_15 += out_8
        out_16 = out_15
        out_15 = out_8 = None
        out_17 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        out_18 = torch.conv2d(
            out_17,
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
        out_19 = torch.nn.functional.batch_norm(
            out_18,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_18 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (None)
        out_20 = torch.nn.functional.relu(out_19, inplace=True)
        out_19 = None
        x_22 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_20 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        splits_2 = x_24.view(1, 2, -1, 64, 48)
        x_24 = None
        gap_10 = splits_2.sum(dim=1)
        gap_11 = torch.nn.functional.adaptive_avg_pool2d(gap_10, 1)
        gap_10 = None
        gap_12 = torch.conv2d(
            gap_11,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_11 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_13 = torch.nn.functional.batch_norm(
            gap_12,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_12 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_14 = torch.nn.functional.relu(gap_13, inplace=True)
        gap_13 = None
        atten_4 = torch.conv2d(
            gap_14,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_14 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_9 = atten_4.view(1, 1, 2, -1)
        atten_4 = None
        x_25 = view_9.transpose(1, 2)
        view_9 = None
        x_26 = torch.nn.functional.softmax(x_25, dim=1)
        x_25 = None
        x_27 = x_26.reshape(1, -1)
        x_26 = None
        atten_5 = x_27.view(1, -1, 1, 1)
        x_27 = None
        attens_2 = atten_5.view(1, 2, -1, 1, 1)
        atten_5 = None
        mul_2 = attens_2 * splits_2
        attens_2 = splits_2 = None
        out_21 = torch.sum(mul_2, dim=1)
        mul_2 = None
        out_22 = out_21.contiguous()
        out_21 = None
        out_23 = torch.conv2d(
            out_22,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_22 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (None)
        out_24 = torch.nn.functional.batch_norm(
            out_23,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_23 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (None)
        out_24 += out_17
        out_25 = out_24
        out_24 = out_17 = None
        out_26 = torch.nn.functional.relu(out_25, inplace=True)
        out_25 = None
        out_27 = torch.conv2d(
            out_26,
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
        out_28 = torch.nn.functional.batch_norm(
            out_27,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_27 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        x_28 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_29 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        splits_3 = x_30.view(1, 2, -1, 64, 48)
        x_30 = None
        gap_15 = splits_3.sum(dim=1)
        gap_16 = torch.nn.functional.adaptive_avg_pool2d(gap_15, 1)
        gap_15 = None
        gap_17 = torch.conv2d(
            gap_16,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_16 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_18 = torch.nn.functional.batch_norm(
            gap_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_17 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_19 = torch.nn.functional.relu(gap_18, inplace=True)
        gap_18 = None
        atten_6 = torch.conv2d(
            gap_19,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_19 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_13 = atten_6.view(1, 1, 2, -1)
        atten_6 = None
        x_31 = view_13.transpose(1, 2)
        view_13 = None
        x_32 = torch.nn.functional.softmax(x_31, dim=1)
        x_31 = None
        x_33 = x_32.reshape(1, -1)
        x_32 = None
        atten_7 = x_33.view(1, -1, 1, 1)
        x_33 = None
        attens_3 = atten_7.view(1, 2, -1, 1, 1)
        atten_7 = None
        mul_3 = attens_3 * splits_3
        attens_3 = splits_3 = None
        out_30 = torch.sum(mul_3, dim=1)
        mul_3 = None
        out_31 = out_30.contiguous()
        out_30 = None
        out_32 = torch._C._nn.avg_pool2d(out_31, 3, 2, 1, False, True, None)
        out_31 = None
        out_33 = torch.conv2d(
            out_32,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_32 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (None)
        out_34 = torch.nn.functional.batch_norm(
            out_33,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_33 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_3 = torch._C._nn.avg_pool2d(out_26, 2, 2, 0, True, False, None)
        out_26 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_34 += input_5
        out_35 = out_34
        out_34 = input_5 = None
        out_36 = torch.nn.functional.relu(out_35, inplace=True)
        out_35 = None
        out_37 = torch.conv2d(
            out_36,
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
        out_38 = torch.nn.functional.batch_norm(
            out_37,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_37 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_39 = torch.nn.functional.relu(out_38, inplace=True)
        out_38 = None
        x_34 = torch.conv2d(
            out_39,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_39 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        splits_4 = x_36.view(1, 2, -1, 32, 24)
        x_36 = None
        gap_20 = splits_4.sum(dim=1)
        gap_21 = torch.nn.functional.adaptive_avg_pool2d(gap_20, 1)
        gap_20 = None
        gap_22 = torch.conv2d(
            gap_21,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_21 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_23 = torch.nn.functional.batch_norm(
            gap_22,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_22 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_24 = torch.nn.functional.relu(gap_23, inplace=True)
        gap_23 = None
        atten_8 = torch.conv2d(
            gap_24,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_24 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_17 = atten_8.view(1, 1, 2, -1)
        atten_8 = None
        x_37 = view_17.transpose(1, 2)
        view_17 = None
        x_38 = torch.nn.functional.softmax(x_37, dim=1)
        x_37 = None
        x_39 = x_38.reshape(1, -1)
        x_38 = None
        atten_9 = x_39.view(1, -1, 1, 1)
        x_39 = None
        attens_4 = atten_9.view(1, 2, -1, 1, 1)
        atten_9 = None
        mul_4 = attens_4 * splits_4
        attens_4 = splits_4 = None
        out_40 = torch.sum(mul_4, dim=1)
        mul_4 = None
        out_41 = out_40.contiguous()
        out_40 = None
        out_42 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (None)
        out_43 = torch.nn.functional.batch_norm(
            out_42,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_42 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (None)
        out_43 += out_36
        out_44 = out_43
        out_43 = out_36 = None
        out_45 = torch.nn.functional.relu(out_44, inplace=True)
        out_44 = None
        out_46 = torch.conv2d(
            out_45,
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
        out_47 = torch.nn.functional.batch_norm(
            out_46,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_46 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (None)
        out_48 = torch.nn.functional.relu(out_47, inplace=True)
        out_47 = None
        x_40 = torch.conv2d(
            out_48,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_48 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        splits_5 = x_42.view(1, 2, -1, 32, 24)
        x_42 = None
        gap_25 = splits_5.sum(dim=1)
        gap_26 = torch.nn.functional.adaptive_avg_pool2d(gap_25, 1)
        gap_25 = None
        gap_27 = torch.conv2d(
            gap_26,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_26 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_28 = torch.nn.functional.batch_norm(
            gap_27,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_27 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_29 = torch.nn.functional.relu(gap_28, inplace=True)
        gap_28 = None
        atten_10 = torch.conv2d(
            gap_29,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_29 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_21 = atten_10.view(1, 1, 2, -1)
        atten_10 = None
        x_43 = view_21.transpose(1, 2)
        view_21 = None
        x_44 = torch.nn.functional.softmax(x_43, dim=1)
        x_43 = None
        x_45 = x_44.reshape(1, -1)
        x_44 = None
        atten_11 = x_45.view(1, -1, 1, 1)
        x_45 = None
        attens_5 = atten_11.view(1, 2, -1, 1, 1)
        atten_11 = None
        mul_5 = attens_5 * splits_5
        attens_5 = splits_5 = None
        out_49 = torch.sum(mul_5, dim=1)
        mul_5 = None
        out_50 = out_49.contiguous()
        out_49 = None
        out_51 = torch.conv2d(
            out_50,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_50 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (None)
        out_52 = torch.nn.functional.batch_norm(
            out_51,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_51 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (None)
        out_52 += out_45
        out_53 = out_52
        out_52 = out_45 = None
        out_54 = torch.nn.functional.relu(out_53, inplace=True)
        out_53 = None
        out_55 = torch.conv2d(
            out_54,
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
        out_56 = torch.nn.functional.batch_norm(
            out_55,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_55 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (None)
        out_57 = torch.nn.functional.relu(out_56, inplace=True)
        out_56 = None
        x_46 = torch.conv2d(
            out_57,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_57 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        splits_6 = x_48.view(1, 2, -1, 32, 24)
        x_48 = None
        gap_30 = splits_6.sum(dim=1)
        gap_31 = torch.nn.functional.adaptive_avg_pool2d(gap_30, 1)
        gap_30 = None
        gap_32 = torch.conv2d(
            gap_31,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_31 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_33 = torch.nn.functional.batch_norm(
            gap_32,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_32 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_34 = torch.nn.functional.relu(gap_33, inplace=True)
        gap_33 = None
        atten_12 = torch.conv2d(
            gap_34,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_34 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_25 = atten_12.view(1, 1, 2, -1)
        atten_12 = None
        x_49 = view_25.transpose(1, 2)
        view_25 = None
        x_50 = torch.nn.functional.softmax(x_49, dim=1)
        x_49 = None
        x_51 = x_50.reshape(1, -1)
        x_50 = None
        atten_13 = x_51.view(1, -1, 1, 1)
        x_51 = None
        attens_6 = atten_13.view(1, 2, -1, 1, 1)
        atten_13 = None
        mul_6 = attens_6 * splits_6
        attens_6 = splits_6 = None
        out_58 = torch.sum(mul_6, dim=1)
        mul_6 = None
        out_59 = out_58.contiguous()
        out_58 = None
        out_60 = torch.conv2d(
            out_59,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_59 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (None)
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (None)
        out_61 += out_54
        out_62 = out_61
        out_61 = out_54 = None
        out_63 = torch.nn.functional.relu(out_62, inplace=True)
        out_62 = None
        out_64 = torch.conv2d(
            out_63,
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
        out_65 = torch.nn.functional.batch_norm(
            out_64,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_64 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_66 = torch.nn.functional.relu(out_65, inplace=True)
        out_65 = None
        x_52 = torch.conv2d(
            out_66,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_66 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        splits_7 = x_54.view(1, 2, -1, 32, 24)
        x_54 = None
        gap_35 = splits_7.sum(dim=1)
        gap_36 = torch.nn.functional.adaptive_avg_pool2d(gap_35, 1)
        gap_35 = None
        gap_37 = torch.conv2d(
            gap_36,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_36 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_38 = torch.nn.functional.batch_norm(
            gap_37,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_37 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_39 = torch.nn.functional.relu(gap_38, inplace=True)
        gap_38 = None
        atten_14 = torch.conv2d(
            gap_39,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_39 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_29 = atten_14.view(1, 1, 2, -1)
        atten_14 = None
        x_55 = view_29.transpose(1, 2)
        view_29 = None
        x_56 = torch.nn.functional.softmax(x_55, dim=1)
        x_55 = None
        x_57 = x_56.reshape(1, -1)
        x_56 = None
        atten_15 = x_57.view(1, -1, 1, 1)
        x_57 = None
        attens_7 = atten_15.view(1, 2, -1, 1, 1)
        atten_15 = None
        mul_7 = attens_7 * splits_7
        attens_7 = splits_7 = None
        out_67 = torch.sum(mul_7, dim=1)
        mul_7 = None
        out_68 = out_67.contiguous()
        out_67 = None
        out_69 = torch._C._nn.avg_pool2d(out_68, 3, 2, 1, False, True, None)
        out_68 = None
        out_70 = torch.conv2d(
            out_69,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_69 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_71 = torch.nn.functional.batch_norm(
            out_70,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_70 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_6 = torch._C._nn.avg_pool2d(out_63, 2, 2, 0, True, False, None)
        out_63 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_71 += input_8
        out_72 = out_71
        out_71 = input_8 = None
        out_73 = torch.nn.functional.relu(out_72, inplace=True)
        out_72 = None
        out_74 = torch.conv2d(
            out_73,
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
        out_75 = torch.nn.functional.batch_norm(
            out_74,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_74 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_76 = torch.nn.functional.relu(out_75, inplace=True)
        out_75 = None
        x_58 = torch.conv2d(
            out_76,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_76 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        splits_8 = x_60.view(1, 2, -1, 16, 12)
        x_60 = None
        gap_40 = splits_8.sum(dim=1)
        gap_41 = torch.nn.functional.adaptive_avg_pool2d(gap_40, 1)
        gap_40 = None
        gap_42 = torch.conv2d(
            gap_41,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_41 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_43 = torch.nn.functional.batch_norm(
            gap_42,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_42 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_44 = torch.nn.functional.relu(gap_43, inplace=True)
        gap_43 = None
        atten_16 = torch.conv2d(
            gap_44,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_44 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_33 = atten_16.view(1, 1, 2, -1)
        atten_16 = None
        x_61 = view_33.transpose(1, 2)
        view_33 = None
        x_62 = torch.nn.functional.softmax(x_61, dim=1)
        x_61 = None
        x_63 = x_62.reshape(1, -1)
        x_62 = None
        atten_17 = x_63.view(1, -1, 1, 1)
        x_63 = None
        attens_8 = atten_17.view(1, 2, -1, 1, 1)
        atten_17 = None
        mul_8 = attens_8 * splits_8
        attens_8 = splits_8 = None
        out_77 = torch.sum(mul_8, dim=1)
        mul_8 = None
        out_78 = out_77.contiguous()
        out_77 = None
        out_79 = torch.conv2d(
            out_78,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_78 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_80 = torch.nn.functional.batch_norm(
            out_79,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_79 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        out_80 += out_73
        out_81 = out_80
        out_80 = out_73 = None
        out_82 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        out_83 = torch.conv2d(
            out_82,
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
        out_84 = torch.nn.functional.batch_norm(
            out_83,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_83 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_85 = torch.nn.functional.relu(out_84, inplace=True)
        out_84 = None
        x_64 = torch.conv2d(
            out_85,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_85 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        splits_9 = x_66.view(1, 2, -1, 16, 12)
        x_66 = None
        gap_45 = splits_9.sum(dim=1)
        gap_46 = torch.nn.functional.adaptive_avg_pool2d(gap_45, 1)
        gap_45 = None
        gap_47 = torch.conv2d(
            gap_46,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_46 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_48 = torch.nn.functional.batch_norm(
            gap_47,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_47 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_49 = torch.nn.functional.relu(gap_48, inplace=True)
        gap_48 = None
        atten_18 = torch.conv2d(
            gap_49,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_49 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_37 = atten_18.view(1, 1, 2, -1)
        atten_18 = None
        x_67 = view_37.transpose(1, 2)
        view_37 = None
        x_68 = torch.nn.functional.softmax(x_67, dim=1)
        x_67 = None
        x_69 = x_68.reshape(1, -1)
        x_68 = None
        atten_19 = x_69.view(1, -1, 1, 1)
        x_69 = None
        attens_9 = atten_19.view(1, 2, -1, 1, 1)
        atten_19 = None
        mul_9 = attens_9 * splits_9
        attens_9 = splits_9 = None
        out_86 = torch.sum(mul_9, dim=1)
        mul_9 = None
        out_87 = out_86.contiguous()
        out_86 = None
        out_88 = torch.conv2d(
            out_87,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_87 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_89 = torch.nn.functional.batch_norm(
            out_88,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_88 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        out_89 += out_82
        out_90 = out_89
        out_89 = out_82 = None
        out_91 = torch.nn.functional.relu(out_90, inplace=True)
        out_90 = None
        out_92 = torch.conv2d(
            out_91,
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
        out_93 = torch.nn.functional.batch_norm(
            out_92,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_92 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_94 = torch.nn.functional.relu(out_93, inplace=True)
        out_93 = None
        x_70 = torch.conv2d(
            out_94,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_94 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        splits_10 = x_72.view(1, 2, -1, 16, 12)
        x_72 = None
        gap_50 = splits_10.sum(dim=1)
        gap_51 = torch.nn.functional.adaptive_avg_pool2d(gap_50, 1)
        gap_50 = None
        gap_52 = torch.conv2d(
            gap_51,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_51 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_53 = torch.nn.functional.batch_norm(
            gap_52,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_52 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_54 = torch.nn.functional.relu(gap_53, inplace=True)
        gap_53 = None
        atten_20 = torch.conv2d(
            gap_54,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_54 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_41 = atten_20.view(1, 1, 2, -1)
        atten_20 = None
        x_73 = view_41.transpose(1, 2)
        view_41 = None
        x_74 = torch.nn.functional.softmax(x_73, dim=1)
        x_73 = None
        x_75 = x_74.reshape(1, -1)
        x_74 = None
        atten_21 = x_75.view(1, -1, 1, 1)
        x_75 = None
        attens_10 = atten_21.view(1, 2, -1, 1, 1)
        atten_21 = None
        mul_10 = attens_10 * splits_10
        attens_10 = splits_10 = None
        out_95 = torch.sum(mul_10, dim=1)
        mul_10 = None
        out_96 = out_95.contiguous()
        out_95 = None
        out_97 = torch.conv2d(
            out_96,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_96 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_98 = torch.nn.functional.batch_norm(
            out_97,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_97 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        out_98 += out_91
        out_99 = out_98
        out_98 = out_91 = None
        out_100 = torch.nn.functional.relu(out_99, inplace=True)
        out_99 = None
        out_101 = torch.conv2d(
            out_100,
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
        out_102 = torch.nn.functional.batch_norm(
            out_101,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_101 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_103 = torch.nn.functional.relu(out_102, inplace=True)
        out_102 = None
        x_76 = torch.conv2d(
            out_103,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_103 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        splits_11 = x_78.view(1, 2, -1, 16, 12)
        x_78 = None
        gap_55 = splits_11.sum(dim=1)
        gap_56 = torch.nn.functional.adaptive_avg_pool2d(gap_55, 1)
        gap_55 = None
        gap_57 = torch.conv2d(
            gap_56,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_56 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_58 = torch.nn.functional.batch_norm(
            gap_57,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_57 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_59 = torch.nn.functional.relu(gap_58, inplace=True)
        gap_58 = None
        atten_22 = torch.conv2d(
            gap_59,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_59 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_45 = atten_22.view(1, 1, 2, -1)
        atten_22 = None
        x_79 = view_45.transpose(1, 2)
        view_45 = None
        x_80 = torch.nn.functional.softmax(x_79, dim=1)
        x_79 = None
        x_81 = x_80.reshape(1, -1)
        x_80 = None
        atten_23 = x_81.view(1, -1, 1, 1)
        x_81 = None
        attens_11 = atten_23.view(1, 2, -1, 1, 1)
        atten_23 = None
        mul_11 = attens_11 * splits_11
        attens_11 = splits_11 = None
        out_104 = torch.sum(mul_11, dim=1)
        mul_11 = None
        out_105 = out_104.contiguous()
        out_104 = None
        out_106 = torch.conv2d(
            out_105,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_105 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_107 = torch.nn.functional.batch_norm(
            out_106,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_106 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        out_107 += out_100
        out_108 = out_107
        out_107 = out_100 = None
        out_109 = torch.nn.functional.relu(out_108, inplace=True)
        out_108 = None
        out_110 = torch.conv2d(
            out_109,
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
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_112 = torch.nn.functional.relu(out_111, inplace=True)
        out_111 = None
        x_82 = torch.conv2d(
            out_112,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_112 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        splits_12 = x_84.view(1, 2, -1, 16, 12)
        x_84 = None
        gap_60 = splits_12.sum(dim=1)
        gap_61 = torch.nn.functional.adaptive_avg_pool2d(gap_60, 1)
        gap_60 = None
        gap_62 = torch.conv2d(
            gap_61,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_61 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_63 = torch.nn.functional.batch_norm(
            gap_62,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_62 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_64 = torch.nn.functional.relu(gap_63, inplace=True)
        gap_63 = None
        atten_24 = torch.conv2d(
            gap_64,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_64 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_49 = atten_24.view(1, 1, 2, -1)
        atten_24 = None
        x_85 = view_49.transpose(1, 2)
        view_49 = None
        x_86 = torch.nn.functional.softmax(x_85, dim=1)
        x_85 = None
        x_87 = x_86.reshape(1, -1)
        x_86 = None
        atten_25 = x_87.view(1, -1, 1, 1)
        x_87 = None
        attens_12 = atten_25.view(1, 2, -1, 1, 1)
        atten_25 = None
        mul_12 = attens_12 * splits_12
        attens_12 = splits_12 = None
        out_113 = torch.sum(mul_12, dim=1)
        mul_12 = None
        out_114 = out_113.contiguous()
        out_113 = None
        out_115 = torch.conv2d(
            out_114,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_114 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_116 = torch.nn.functional.batch_norm(
            out_115,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_115 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        out_116 += out_109
        out_117 = out_116
        out_116 = out_109 = None
        out_118 = torch.nn.functional.relu(out_117, inplace=True)
        out_117 = None
        out_119 = torch.conv2d(
            out_118,
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
        out_120 = torch.nn.functional.batch_norm(
            out_119,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_119 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (None)
        out_121 = torch.nn.functional.relu(out_120, inplace=True)
        out_120 = None
        x_88 = torch.conv2d(
            out_121,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_121 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        splits_13 = x_90.view(1, 2, -1, 16, 12)
        x_90 = None
        gap_65 = splits_13.sum(dim=1)
        gap_66 = torch.nn.functional.adaptive_avg_pool2d(gap_65, 1)
        gap_65 = None
        gap_67 = torch.conv2d(
            gap_66,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_66 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_68 = torch.nn.functional.batch_norm(
            gap_67,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_67 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_69 = torch.nn.functional.relu(gap_68, inplace=True)
        gap_68 = None
        atten_26 = torch.conv2d(
            gap_69,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_69 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_53 = atten_26.view(1, 1, 2, -1)
        atten_26 = None
        x_91 = view_53.transpose(1, 2)
        view_53 = None
        x_92 = torch.nn.functional.softmax(x_91, dim=1)
        x_91 = None
        x_93 = x_92.reshape(1, -1)
        x_92 = None
        atten_27 = x_93.view(1, -1, 1, 1)
        x_93 = None
        attens_13 = atten_27.view(1, 2, -1, 1, 1)
        atten_27 = None
        mul_13 = attens_13 * splits_13
        attens_13 = splits_13 = None
        out_122 = torch.sum(mul_13, dim=1)
        mul_13 = None
        out_123 = out_122.contiguous()
        out_122 = None
        out_124 = torch.conv2d(
            out_123,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_123 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (None)
        out_125 = torch.nn.functional.batch_norm(
            out_124,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_124 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (None)
        out_125 += out_118
        out_126 = out_125
        out_125 = out_118 = None
        out_127 = torch.nn.functional.relu(out_126, inplace=True)
        out_126 = None
        out_128 = torch.conv2d(
            out_127,
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
        out_129 = torch.nn.functional.batch_norm(
            out_128,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_128 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_ = (None)
        out_130 = torch.nn.functional.relu(out_129, inplace=True)
        out_129 = None
        x_94 = torch.conv2d(
            out_130,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_130 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        splits_14 = x_96.view(1, 2, -1, 16, 12)
        x_96 = None
        gap_70 = splits_14.sum(dim=1)
        gap_71 = torch.nn.functional.adaptive_avg_pool2d(gap_70, 1)
        gap_70 = None
        gap_72 = torch.conv2d(
            gap_71,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_71 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_73 = torch.nn.functional.batch_norm(
            gap_72,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_72 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_74 = torch.nn.functional.relu(gap_73, inplace=True)
        gap_73 = None
        atten_28 = torch.conv2d(
            gap_74,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_74 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_57 = atten_28.view(1, 1, 2, -1)
        atten_28 = None
        x_97 = view_57.transpose(1, 2)
        view_57 = None
        x_98 = torch.nn.functional.softmax(x_97, dim=1)
        x_97 = None
        x_99 = x_98.reshape(1, -1)
        x_98 = None
        atten_29 = x_99.view(1, -1, 1, 1)
        x_99 = None
        attens_14 = atten_29.view(1, 2, -1, 1, 1)
        atten_29 = None
        mul_14 = attens_14 * splits_14
        attens_14 = splits_14 = None
        out_131 = torch.sum(mul_14, dim=1)
        mul_14 = None
        out_132 = out_131.contiguous()
        out_131 = None
        out_133 = torch.conv2d(
            out_132,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_132 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_ = (None)
        out_134 = torch.nn.functional.batch_norm(
            out_133,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_133 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_ = (None)
        out_134 += out_127
        out_135 = out_134
        out_134 = out_127 = None
        out_136 = torch.nn.functional.relu(out_135, inplace=True)
        out_135 = None
        out_137 = torch.conv2d(
            out_136,
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
        out_138 = torch.nn.functional.batch_norm(
            out_137,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_137 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_ = (None)
        out_139 = torch.nn.functional.relu(out_138, inplace=True)
        out_138 = None
        x_100 = torch.conv2d(
            out_139,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_139 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        splits_15 = x_102.view(1, 2, -1, 16, 12)
        x_102 = None
        gap_75 = splits_15.sum(dim=1)
        gap_76 = torch.nn.functional.adaptive_avg_pool2d(gap_75, 1)
        gap_75 = None
        gap_77 = torch.conv2d(
            gap_76,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_76 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_78 = torch.nn.functional.batch_norm(
            gap_77,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_77 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_79 = torch.nn.functional.relu(gap_78, inplace=True)
        gap_78 = None
        atten_30 = torch.conv2d(
            gap_79,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_79 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_61 = atten_30.view(1, 1, 2, -1)
        atten_30 = None
        x_103 = view_61.transpose(1, 2)
        view_61 = None
        x_104 = torch.nn.functional.softmax(x_103, dim=1)
        x_103 = None
        x_105 = x_104.reshape(1, -1)
        x_104 = None
        atten_31 = x_105.view(1, -1, 1, 1)
        x_105 = None
        attens_15 = atten_31.view(1, 2, -1, 1, 1)
        atten_31 = None
        mul_15 = attens_15 * splits_15
        attens_15 = splits_15 = None
        out_140 = torch.sum(mul_15, dim=1)
        mul_15 = None
        out_141 = out_140.contiguous()
        out_140 = None
        out_142 = torch.conv2d(
            out_141,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_141 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_ = (None)
        out_143 = torch.nn.functional.batch_norm(
            out_142,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_142 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_ = (None)
        out_143 += out_136
        out_144 = out_143
        out_143 = out_136 = None
        out_145 = torch.nn.functional.relu(out_144, inplace=True)
        out_144 = None
        out_146 = torch.conv2d(
            out_145,
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
        out_147 = torch.nn.functional.batch_norm(
            out_146,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_146 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_ = (None)
        out_148 = torch.nn.functional.relu(out_147, inplace=True)
        out_147 = None
        x_106 = torch.conv2d(
            out_148,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_148 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        splits_16 = x_108.view(1, 2, -1, 16, 12)
        x_108 = None
        gap_80 = splits_16.sum(dim=1)
        gap_81 = torch.nn.functional.adaptive_avg_pool2d(gap_80, 1)
        gap_80 = None
        gap_82 = torch.conv2d(
            gap_81,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_81 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_83 = torch.nn.functional.batch_norm(
            gap_82,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_82 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_84 = torch.nn.functional.relu(gap_83, inplace=True)
        gap_83 = None
        atten_32 = torch.conv2d(
            gap_84,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_84 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_65 = atten_32.view(1, 1, 2, -1)
        atten_32 = None
        x_109 = view_65.transpose(1, 2)
        view_65 = None
        x_110 = torch.nn.functional.softmax(x_109, dim=1)
        x_109 = None
        x_111 = x_110.reshape(1, -1)
        x_110 = None
        atten_33 = x_111.view(1, -1, 1, 1)
        x_111 = None
        attens_16 = atten_33.view(1, 2, -1, 1, 1)
        atten_33 = None
        mul_16 = attens_16 * splits_16
        attens_16 = splits_16 = None
        out_149 = torch.sum(mul_16, dim=1)
        mul_16 = None
        out_150 = out_149.contiguous()
        out_149 = None
        out_151 = torch.conv2d(
            out_150,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_150 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_ = (None)
        out_152 = torch.nn.functional.batch_norm(
            out_151,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_151 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_ = (None)
        out_152 += out_145
        out_153 = out_152
        out_152 = out_145 = None
        out_154 = torch.nn.functional.relu(out_153, inplace=True)
        out_153 = None
        out_155 = torch.conv2d(
            out_154,
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
        out_156 = torch.nn.functional.batch_norm(
            out_155,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_155 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_ = (None)
        out_157 = torch.nn.functional.relu(out_156, inplace=True)
        out_156 = None
        x_112 = torch.conv2d(
            out_157,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_157 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        splits_17 = x_114.view(1, 2, -1, 16, 12)
        x_114 = None
        gap_85 = splits_17.sum(dim=1)
        gap_86 = torch.nn.functional.adaptive_avg_pool2d(gap_85, 1)
        gap_85 = None
        gap_87 = torch.conv2d(
            gap_86,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_86 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_88 = torch.nn.functional.batch_norm(
            gap_87,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_87 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_89 = torch.nn.functional.relu(gap_88, inplace=True)
        gap_88 = None
        atten_34 = torch.conv2d(
            gap_89,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_89 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_69 = atten_34.view(1, 1, 2, -1)
        atten_34 = None
        x_115 = view_69.transpose(1, 2)
        view_69 = None
        x_116 = torch.nn.functional.softmax(x_115, dim=1)
        x_115 = None
        x_117 = x_116.reshape(1, -1)
        x_116 = None
        atten_35 = x_117.view(1, -1, 1, 1)
        x_117 = None
        attens_17 = atten_35.view(1, 2, -1, 1, 1)
        atten_35 = None
        mul_17 = attens_17 * splits_17
        attens_17 = splits_17 = None
        out_158 = torch.sum(mul_17, dim=1)
        mul_17 = None
        out_159 = out_158.contiguous()
        out_158 = None
        out_160 = torch.conv2d(
            out_159,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_159 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_ = (None)
        out_161 = torch.nn.functional.batch_norm(
            out_160,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_160 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_ = (None)
        out_161 += out_154
        out_162 = out_161
        out_161 = out_154 = None
        out_163 = torch.nn.functional.relu(out_162, inplace=True)
        out_162 = None
        out_164 = torch.conv2d(
            out_163,
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
        out_165 = torch.nn.functional.batch_norm(
            out_164,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_164 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_ = (None)
        out_166 = torch.nn.functional.relu(out_165, inplace=True)
        out_165 = None
        x_118 = torch.conv2d(
            out_166,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_166 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_120 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        splits_18 = x_120.view(1, 2, -1, 16, 12)
        x_120 = None
        gap_90 = splits_18.sum(dim=1)
        gap_91 = torch.nn.functional.adaptive_avg_pool2d(gap_90, 1)
        gap_90 = None
        gap_92 = torch.conv2d(
            gap_91,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_91 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_93 = torch.nn.functional.batch_norm(
            gap_92,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_92 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_94 = torch.nn.functional.relu(gap_93, inplace=True)
        gap_93 = None
        atten_36 = torch.conv2d(
            gap_94,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_94 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_73 = atten_36.view(1, 1, 2, -1)
        atten_36 = None
        x_121 = view_73.transpose(1, 2)
        view_73 = None
        x_122 = torch.nn.functional.softmax(x_121, dim=1)
        x_121 = None
        x_123 = x_122.reshape(1, -1)
        x_122 = None
        atten_37 = x_123.view(1, -1, 1, 1)
        x_123 = None
        attens_18 = atten_37.view(1, 2, -1, 1, 1)
        atten_37 = None
        mul_18 = attens_18 * splits_18
        attens_18 = splits_18 = None
        out_167 = torch.sum(mul_18, dim=1)
        mul_18 = None
        out_168 = out_167.contiguous()
        out_167 = None
        out_169 = torch.conv2d(
            out_168,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_168 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_ = (None)
        out_170 = torch.nn.functional.batch_norm(
            out_169,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_169 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_ = (None)
        out_170 += out_163
        out_171 = out_170
        out_170 = out_163 = None
        out_172 = torch.nn.functional.relu(out_171, inplace=True)
        out_171 = None
        out_173 = torch.conv2d(
            out_172,
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
        out_174 = torch.nn.functional.batch_norm(
            out_173,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_173 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_ = (None)
        out_175 = torch.nn.functional.relu(out_174, inplace=True)
        out_174 = None
        x_124 = torch.conv2d(
            out_175,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_175 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        splits_19 = x_126.view(1, 2, -1, 16, 12)
        x_126 = None
        gap_95 = splits_19.sum(dim=1)
        gap_96 = torch.nn.functional.adaptive_avg_pool2d(gap_95, 1)
        gap_95 = None
        gap_97 = torch.conv2d(
            gap_96,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_96 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_98 = torch.nn.functional.batch_norm(
            gap_97,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_97 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_99 = torch.nn.functional.relu(gap_98, inplace=True)
        gap_98 = None
        atten_38 = torch.conv2d(
            gap_99,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_99 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_77 = atten_38.view(1, 1, 2, -1)
        atten_38 = None
        x_127 = view_77.transpose(1, 2)
        view_77 = None
        x_128 = torch.nn.functional.softmax(x_127, dim=1)
        x_127 = None
        x_129 = x_128.reshape(1, -1)
        x_128 = None
        atten_39 = x_129.view(1, -1, 1, 1)
        x_129 = None
        attens_19 = atten_39.view(1, 2, -1, 1, 1)
        atten_39 = None
        mul_19 = attens_19 * splits_19
        attens_19 = splits_19 = None
        out_176 = torch.sum(mul_19, dim=1)
        mul_19 = None
        out_177 = out_176.contiguous()
        out_176 = None
        out_178 = torch.conv2d(
            out_177,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_177 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_ = (None)
        out_179 = torch.nn.functional.batch_norm(
            out_178,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_178 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_ = (None)
        out_179 += out_172
        out_180 = out_179
        out_179 = out_172 = None
        out_181 = torch.nn.functional.relu(out_180, inplace=True)
        out_180 = None
        out_182 = torch.conv2d(
            out_181,
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
        out_183 = torch.nn.functional.batch_norm(
            out_182,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_182 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_ = (None)
        out_184 = torch.nn.functional.relu(out_183, inplace=True)
        out_183 = None
        x_130 = torch.conv2d(
            out_184,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_184 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        splits_20 = x_132.view(1, 2, -1, 16, 12)
        x_132 = None
        gap_100 = splits_20.sum(dim=1)
        gap_101 = torch.nn.functional.adaptive_avg_pool2d(gap_100, 1)
        gap_100 = None
        gap_102 = torch.conv2d(
            gap_101,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_101 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_103 = torch.nn.functional.batch_norm(
            gap_102,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_102 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_104 = torch.nn.functional.relu(gap_103, inplace=True)
        gap_103 = None
        atten_40 = torch.conv2d(
            gap_104,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_104 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_81 = atten_40.view(1, 1, 2, -1)
        atten_40 = None
        x_133 = view_81.transpose(1, 2)
        view_81 = None
        x_134 = torch.nn.functional.softmax(x_133, dim=1)
        x_133 = None
        x_135 = x_134.reshape(1, -1)
        x_134 = None
        atten_41 = x_135.view(1, -1, 1, 1)
        x_135 = None
        attens_20 = atten_41.view(1, 2, -1, 1, 1)
        atten_41 = None
        mul_20 = attens_20 * splits_20
        attens_20 = splits_20 = None
        out_185 = torch.sum(mul_20, dim=1)
        mul_20 = None
        out_186 = out_185.contiguous()
        out_185 = None
        out_187 = torch.conv2d(
            out_186,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_186 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_ = (None)
        out_188 = torch.nn.functional.batch_norm(
            out_187,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_187 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_ = (None)
        out_188 += out_181
        out_189 = out_188
        out_188 = out_181 = None
        out_190 = torch.nn.functional.relu(out_189, inplace=True)
        out_189 = None
        out_191 = torch.conv2d(
            out_190,
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
        out_192 = torch.nn.functional.batch_norm(
            out_191,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_191 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_ = (None)
        out_193 = torch.nn.functional.relu(out_192, inplace=True)
        out_192 = None
        x_136 = torch.conv2d(
            out_193,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_193 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        splits_21 = x_138.view(1, 2, -1, 16, 12)
        x_138 = None
        gap_105 = splits_21.sum(dim=1)
        gap_106 = torch.nn.functional.adaptive_avg_pool2d(gap_105, 1)
        gap_105 = None
        gap_107 = torch.conv2d(
            gap_106,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_106 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_108 = torch.nn.functional.batch_norm(
            gap_107,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_107 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_109 = torch.nn.functional.relu(gap_108, inplace=True)
        gap_108 = None
        atten_42 = torch.conv2d(
            gap_109,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_109 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_85 = atten_42.view(1, 1, 2, -1)
        atten_42 = None
        x_139 = view_85.transpose(1, 2)
        view_85 = None
        x_140 = torch.nn.functional.softmax(x_139, dim=1)
        x_139 = None
        x_141 = x_140.reshape(1, -1)
        x_140 = None
        atten_43 = x_141.view(1, -1, 1, 1)
        x_141 = None
        attens_21 = atten_43.view(1, 2, -1, 1, 1)
        atten_43 = None
        mul_21 = attens_21 * splits_21
        attens_21 = splits_21 = None
        out_194 = torch.sum(mul_21, dim=1)
        mul_21 = None
        out_195 = out_194.contiguous()
        out_194 = None
        out_196 = torch.conv2d(
            out_195,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_195 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_ = (None)
        out_197 = torch.nn.functional.batch_norm(
            out_196,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_196 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_ = (None)
        out_197 += out_190
        out_198 = out_197
        out_197 = out_190 = None
        out_199 = torch.nn.functional.relu(out_198, inplace=True)
        out_198 = None
        out_200 = torch.conv2d(
            out_199,
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
        out_201 = torch.nn.functional.batch_norm(
            out_200,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_200 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_ = (None)
        out_202 = torch.nn.functional.relu(out_201, inplace=True)
        out_201 = None
        x_142 = torch.conv2d(
            out_202,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_202 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        splits_22 = x_144.view(1, 2, -1, 16, 12)
        x_144 = None
        gap_110 = splits_22.sum(dim=1)
        gap_111 = torch.nn.functional.adaptive_avg_pool2d(gap_110, 1)
        gap_110 = None
        gap_112 = torch.conv2d(
            gap_111,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_111 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_113 = torch.nn.functional.batch_norm(
            gap_112,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_112 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_114 = torch.nn.functional.relu(gap_113, inplace=True)
        gap_113 = None
        atten_44 = torch.conv2d(
            gap_114,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_114 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_89 = atten_44.view(1, 1, 2, -1)
        atten_44 = None
        x_145 = view_89.transpose(1, 2)
        view_89 = None
        x_146 = torch.nn.functional.softmax(x_145, dim=1)
        x_145 = None
        x_147 = x_146.reshape(1, -1)
        x_146 = None
        atten_45 = x_147.view(1, -1, 1, 1)
        x_147 = None
        attens_22 = atten_45.view(1, 2, -1, 1, 1)
        atten_45 = None
        mul_22 = attens_22 * splits_22
        attens_22 = splits_22 = None
        out_203 = torch.sum(mul_22, dim=1)
        mul_22 = None
        out_204 = out_203.contiguous()
        out_203 = None
        out_205 = torch.conv2d(
            out_204,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_204 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_ = (None)
        out_206 = torch.nn.functional.batch_norm(
            out_205,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_205 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_ = (None)
        out_206 += out_199
        out_207 = out_206
        out_206 = out_199 = None
        out_208 = torch.nn.functional.relu(out_207, inplace=True)
        out_207 = None
        out_209 = torch.conv2d(
            out_208,
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
        out_210 = torch.nn.functional.batch_norm(
            out_209,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_209 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_ = (None)
        out_211 = torch.nn.functional.relu(out_210, inplace=True)
        out_210 = None
        x_148 = torch.conv2d(
            out_211,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_211 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        splits_23 = x_150.view(1, 2, -1, 16, 12)
        x_150 = None
        gap_115 = splits_23.sum(dim=1)
        gap_116 = torch.nn.functional.adaptive_avg_pool2d(gap_115, 1)
        gap_115 = None
        gap_117 = torch.conv2d(
            gap_116,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_116 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_118 = torch.nn.functional.batch_norm(
            gap_117,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_117 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_119 = torch.nn.functional.relu(gap_118, inplace=True)
        gap_118 = None
        atten_46 = torch.conv2d(
            gap_119,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_119 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_93 = atten_46.view(1, 1, 2, -1)
        atten_46 = None
        x_151 = view_93.transpose(1, 2)
        view_93 = None
        x_152 = torch.nn.functional.softmax(x_151, dim=1)
        x_151 = None
        x_153 = x_152.reshape(1, -1)
        x_152 = None
        atten_47 = x_153.view(1, -1, 1, 1)
        x_153 = None
        attens_23 = atten_47.view(1, 2, -1, 1, 1)
        atten_47 = None
        mul_23 = attens_23 * splits_23
        attens_23 = splits_23 = None
        out_212 = torch.sum(mul_23, dim=1)
        mul_23 = None
        out_213 = out_212.contiguous()
        out_212 = None
        out_214 = torch.conv2d(
            out_213,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_213 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_ = (None)
        out_215 = torch.nn.functional.batch_norm(
            out_214,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_214 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_ = (None)
        out_215 += out_208
        out_216 = out_215
        out_215 = out_208 = None
        out_217 = torch.nn.functional.relu(out_216, inplace=True)
        out_216 = None
        out_218 = torch.conv2d(
            out_217,
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
        out_219 = torch.nn.functional.batch_norm(
            out_218,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_218 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_ = (None)
        out_220 = torch.nn.functional.relu(out_219, inplace=True)
        out_219 = None
        x_154 = torch.conv2d(
            out_220,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_220 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        splits_24 = x_156.view(1, 2, -1, 16, 12)
        x_156 = None
        gap_120 = splits_24.sum(dim=1)
        gap_121 = torch.nn.functional.adaptive_avg_pool2d(gap_120, 1)
        gap_120 = None
        gap_122 = torch.conv2d(
            gap_121,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_121 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_123 = torch.nn.functional.batch_norm(
            gap_122,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_122 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_124 = torch.nn.functional.relu(gap_123, inplace=True)
        gap_123 = None
        atten_48 = torch.conv2d(
            gap_124,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_124 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_97 = atten_48.view(1, 1, 2, -1)
        atten_48 = None
        x_157 = view_97.transpose(1, 2)
        view_97 = None
        x_158 = torch.nn.functional.softmax(x_157, dim=1)
        x_157 = None
        x_159 = x_158.reshape(1, -1)
        x_158 = None
        atten_49 = x_159.view(1, -1, 1, 1)
        x_159 = None
        attens_24 = atten_49.view(1, 2, -1, 1, 1)
        atten_49 = None
        mul_24 = attens_24 * splits_24
        attens_24 = splits_24 = None
        out_221 = torch.sum(mul_24, dim=1)
        mul_24 = None
        out_222 = out_221.contiguous()
        out_221 = None
        out_223 = torch.conv2d(
            out_222,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_222 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_ = (None)
        out_224 = torch.nn.functional.batch_norm(
            out_223,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_223 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_ = (None)
        out_224 += out_217
        out_225 = out_224
        out_224 = out_217 = None
        out_226 = torch.nn.functional.relu(out_225, inplace=True)
        out_225 = None
        out_227 = torch.conv2d(
            out_226,
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
        out_228 = torch.nn.functional.batch_norm(
            out_227,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_227 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_ = (None)
        out_229 = torch.nn.functional.relu(out_228, inplace=True)
        out_228 = None
        x_160 = torch.conv2d(
            out_229,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_229 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_162 = torch.nn.functional.relu(x_161, inplace=True)
        x_161 = None
        splits_25 = x_162.view(1, 2, -1, 16, 12)
        x_162 = None
        gap_125 = splits_25.sum(dim=1)
        gap_126 = torch.nn.functional.adaptive_avg_pool2d(gap_125, 1)
        gap_125 = None
        gap_127 = torch.conv2d(
            gap_126,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_126 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_128 = torch.nn.functional.batch_norm(
            gap_127,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_127 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_129 = torch.nn.functional.relu(gap_128, inplace=True)
        gap_128 = None
        atten_50 = torch.conv2d(
            gap_129,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_129 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_101 = atten_50.view(1, 1, 2, -1)
        atten_50 = None
        x_163 = view_101.transpose(1, 2)
        view_101 = None
        x_164 = torch.nn.functional.softmax(x_163, dim=1)
        x_163 = None
        x_165 = x_164.reshape(1, -1)
        x_164 = None
        atten_51 = x_165.view(1, -1, 1, 1)
        x_165 = None
        attens_25 = atten_51.view(1, 2, -1, 1, 1)
        atten_51 = None
        mul_25 = attens_25 * splits_25
        attens_25 = splits_25 = None
        out_230 = torch.sum(mul_25, dim=1)
        mul_25 = None
        out_231 = out_230.contiguous()
        out_230 = None
        out_232 = torch.conv2d(
            out_231,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_231 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_ = (None)
        out_233 = torch.nn.functional.batch_norm(
            out_232,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_232 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_ = (None)
        out_233 += out_226
        out_234 = out_233
        out_233 = out_226 = None
        out_235 = torch.nn.functional.relu(out_234, inplace=True)
        out_234 = None
        out_236 = torch.conv2d(
            out_235,
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
        out_237 = torch.nn.functional.batch_norm(
            out_236,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_236 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_ = (None)
        out_238 = torch.nn.functional.relu(out_237, inplace=True)
        out_237 = None
        x_166 = torch.conv2d(
            out_238,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_238 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        splits_26 = x_168.view(1, 2, -1, 16, 12)
        x_168 = None
        gap_130 = splits_26.sum(dim=1)
        gap_131 = torch.nn.functional.adaptive_avg_pool2d(gap_130, 1)
        gap_130 = None
        gap_132 = torch.conv2d(
            gap_131,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_131 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_133 = torch.nn.functional.batch_norm(
            gap_132,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_132 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_134 = torch.nn.functional.relu(gap_133, inplace=True)
        gap_133 = None
        atten_52 = torch.conv2d(
            gap_134,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_134 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_105 = atten_52.view(1, 1, 2, -1)
        atten_52 = None
        x_169 = view_105.transpose(1, 2)
        view_105 = None
        x_170 = torch.nn.functional.softmax(x_169, dim=1)
        x_169 = None
        x_171 = x_170.reshape(1, -1)
        x_170 = None
        atten_53 = x_171.view(1, -1, 1, 1)
        x_171 = None
        attens_26 = atten_53.view(1, 2, -1, 1, 1)
        atten_53 = None
        mul_26 = attens_26 * splits_26
        attens_26 = splits_26 = None
        out_239 = torch.sum(mul_26, dim=1)
        mul_26 = None
        out_240 = out_239.contiguous()
        out_239 = None
        out_241 = torch.conv2d(
            out_240,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_240 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_ = (None)
        out_242 = torch.nn.functional.batch_norm(
            out_241,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_241 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_ = (None)
        out_242 += out_235
        out_243 = out_242
        out_242 = out_235 = None
        out_244 = torch.nn.functional.relu(out_243, inplace=True)
        out_243 = None
        out_245 = torch.conv2d(
            out_244,
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
        out_246 = torch.nn.functional.batch_norm(
            out_245,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_245 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_ = (None)
        out_247 = torch.nn.functional.relu(out_246, inplace=True)
        out_246 = None
        x_172 = torch.conv2d(
            out_247,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_247 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        splits_27 = x_174.view(1, 2, -1, 16, 12)
        x_174 = None
        gap_135 = splits_27.sum(dim=1)
        gap_136 = torch.nn.functional.adaptive_avg_pool2d(gap_135, 1)
        gap_135 = None
        gap_137 = torch.conv2d(
            gap_136,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_136 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_138 = torch.nn.functional.batch_norm(
            gap_137,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_137 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_139 = torch.nn.functional.relu(gap_138, inplace=True)
        gap_138 = None
        atten_54 = torch.conv2d(
            gap_139,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_139 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_109 = atten_54.view(1, 1, 2, -1)
        atten_54 = None
        x_175 = view_109.transpose(1, 2)
        view_109 = None
        x_176 = torch.nn.functional.softmax(x_175, dim=1)
        x_175 = None
        x_177 = x_176.reshape(1, -1)
        x_176 = None
        atten_55 = x_177.view(1, -1, 1, 1)
        x_177 = None
        attens_27 = atten_55.view(1, 2, -1, 1, 1)
        atten_55 = None
        mul_27 = attens_27 * splits_27
        attens_27 = splits_27 = None
        out_248 = torch.sum(mul_27, dim=1)
        mul_27 = None
        out_249 = out_248.contiguous()
        out_248 = None
        out_250 = torch.conv2d(
            out_249,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_249 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_ = (None)
        out_251 = torch.nn.functional.batch_norm(
            out_250,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_250 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_ = (None)
        out_251 += out_244
        out_252 = out_251
        out_251 = out_244 = None
        out_253 = torch.nn.functional.relu(out_252, inplace=True)
        out_252 = None
        out_254 = torch.conv2d(
            out_253,
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
        out_255 = torch.nn.functional.batch_norm(
            out_254,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_254 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_ = (None)
        out_256 = torch.nn.functional.relu(out_255, inplace=True)
        out_255 = None
        x_178 = torch.conv2d(
            out_256,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_256 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        splits_28 = x_180.view(1, 2, -1, 16, 12)
        x_180 = None
        gap_140 = splits_28.sum(dim=1)
        gap_141 = torch.nn.functional.adaptive_avg_pool2d(gap_140, 1)
        gap_140 = None
        gap_142 = torch.conv2d(
            gap_141,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_141 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_143 = torch.nn.functional.batch_norm(
            gap_142,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_142 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_144 = torch.nn.functional.relu(gap_143, inplace=True)
        gap_143 = None
        atten_56 = torch.conv2d(
            gap_144,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_144 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_113 = atten_56.view(1, 1, 2, -1)
        atten_56 = None
        x_181 = view_113.transpose(1, 2)
        view_113 = None
        x_182 = torch.nn.functional.softmax(x_181, dim=1)
        x_181 = None
        x_183 = x_182.reshape(1, -1)
        x_182 = None
        atten_57 = x_183.view(1, -1, 1, 1)
        x_183 = None
        attens_28 = atten_57.view(1, 2, -1, 1, 1)
        atten_57 = None
        mul_28 = attens_28 * splits_28
        attens_28 = splits_28 = None
        out_257 = torch.sum(mul_28, dim=1)
        mul_28 = None
        out_258 = out_257.contiguous()
        out_257 = None
        out_259 = torch.conv2d(
            out_258,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_258 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_ = (None)
        out_260 = torch.nn.functional.batch_norm(
            out_259,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_259 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_ = (None)
        out_260 += out_253
        out_261 = out_260
        out_260 = out_253 = None
        out_262 = torch.nn.functional.relu(out_261, inplace=True)
        out_261 = None
        out_263 = torch.conv2d(
            out_262,
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
        out_264 = torch.nn.functional.batch_norm(
            out_263,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_263 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_ = (None)
        out_265 = torch.nn.functional.relu(out_264, inplace=True)
        out_264 = None
        x_184 = torch.conv2d(
            out_265,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_265 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        splits_29 = x_186.view(1, 2, -1, 16, 12)
        x_186 = None
        gap_145 = splits_29.sum(dim=1)
        gap_146 = torch.nn.functional.adaptive_avg_pool2d(gap_145, 1)
        gap_145 = None
        gap_147 = torch.conv2d(
            gap_146,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_146 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_148 = torch.nn.functional.batch_norm(
            gap_147,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_147 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_149 = torch.nn.functional.relu(gap_148, inplace=True)
        gap_148 = None
        atten_58 = torch.conv2d(
            gap_149,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_149 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_117 = atten_58.view(1, 1, 2, -1)
        atten_58 = None
        x_187 = view_117.transpose(1, 2)
        view_117 = None
        x_188 = torch.nn.functional.softmax(x_187, dim=1)
        x_187 = None
        x_189 = x_188.reshape(1, -1)
        x_188 = None
        atten_59 = x_189.view(1, -1, 1, 1)
        x_189 = None
        attens_29 = atten_59.view(1, 2, -1, 1, 1)
        atten_59 = None
        mul_29 = attens_29 * splits_29
        attens_29 = splits_29 = None
        out_266 = torch.sum(mul_29, dim=1)
        mul_29 = None
        out_267 = out_266.contiguous()
        out_266 = None
        out_268 = torch.conv2d(
            out_267,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_267 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = (None)
        out_269 = torch.nn.functional.batch_norm(
            out_268,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_268 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = (None)
        out_269 += out_262
        out_270 = out_269
        out_269 = out_262 = None
        out_271 = torch.nn.functional.relu(out_270, inplace=True)
        out_270 = None
        out_272 = torch.conv2d(
            out_271,
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
        out_273 = torch.nn.functional.batch_norm(
            out_272,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_272 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_274 = torch.nn.functional.relu(out_273, inplace=True)
        out_273 = None
        x_190 = torch.conv2d(
            out_274,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_274 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        splits_30 = x_192.view(1, 2, -1, 16, 12)
        x_192 = None
        gap_150 = splits_30.sum(dim=1)
        gap_151 = torch.nn.functional.adaptive_avg_pool2d(gap_150, 1)
        gap_150 = None
        gap_152 = torch.conv2d(
            gap_151,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_151 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_153 = torch.nn.functional.batch_norm(
            gap_152,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_152 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_154 = torch.nn.functional.relu(gap_153, inplace=True)
        gap_153 = None
        atten_60 = torch.conv2d(
            gap_154,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_154 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_121 = atten_60.view(1, 1, 2, -1)
        atten_60 = None
        x_193 = view_121.transpose(1, 2)
        view_121 = None
        x_194 = torch.nn.functional.softmax(x_193, dim=1)
        x_193 = None
        x_195 = x_194.reshape(1, -1)
        x_194 = None
        atten_61 = x_195.view(1, -1, 1, 1)
        x_195 = None
        attens_30 = atten_61.view(1, 2, -1, 1, 1)
        atten_61 = None
        mul_30 = attens_30 * splits_30
        attens_30 = splits_30 = None
        out_275 = torch.sum(mul_30, dim=1)
        mul_30 = None
        out_276 = out_275.contiguous()
        out_275 = None
        out_277 = torch._C._nn.avg_pool2d(out_276, 3, 2, 1, False, True, None)
        out_276 = None
        out_278 = torch.conv2d(
            out_277,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_277 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_279 = torch.nn.functional.batch_norm(
            out_278,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_278 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_9 = torch._C._nn.avg_pool2d(out_271, 2, 2, 0, True, False, None)
        out_271 = None
        input_10 = torch.conv2d(
            input_9,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_279 += input_11
        out_280 = out_279
        out_279 = input_11 = None
        out_281 = torch.nn.functional.relu(out_280, inplace=True)
        out_280 = None
        out_282 = torch.conv2d(
            out_281,
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
        out_283 = torch.nn.functional.batch_norm(
            out_282,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_282 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_284 = torch.nn.functional.relu(out_283, inplace=True)
        out_283 = None
        x_196 = torch.conv2d(
            out_284,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_284 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        splits_31 = x_198.view(1, 2, -1, 8, 6)
        x_198 = None
        gap_155 = splits_31.sum(dim=1)
        gap_156 = torch.nn.functional.adaptive_avg_pool2d(gap_155, 1)
        gap_155 = None
        gap_157 = torch.conv2d(
            gap_156,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_156 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_158 = torch.nn.functional.batch_norm(
            gap_157,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_157 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_159 = torch.nn.functional.relu(gap_158, inplace=True)
        gap_158 = None
        atten_62 = torch.conv2d(
            gap_159,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_159 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_125 = atten_62.view(1, 1, 2, -1)
        atten_62 = None
        x_199 = view_125.transpose(1, 2)
        view_125 = None
        x_200 = torch.nn.functional.softmax(x_199, dim=1)
        x_199 = None
        x_201 = x_200.reshape(1, -1)
        x_200 = None
        atten_63 = x_201.view(1, -1, 1, 1)
        x_201 = None
        attens_31 = atten_63.view(1, 2, -1, 1, 1)
        atten_63 = None
        mul_31 = attens_31 * splits_31
        attens_31 = splits_31 = None
        out_285 = torch.sum(mul_31, dim=1)
        mul_31 = None
        out_286 = out_285.contiguous()
        out_285 = None
        out_287 = torch.conv2d(
            out_286,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_286 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_288 = torch.nn.functional.batch_norm(
            out_287,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_287 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_288 += out_281
        out_289 = out_288
        out_288 = out_281 = None
        out_290 = torch.nn.functional.relu(out_289, inplace=True)
        out_289 = None
        out_291 = torch.conv2d(
            out_290,
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
        out_292 = torch.nn.functional.batch_norm(
            out_291,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_291 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_293 = torch.nn.functional.relu(out_292, inplace=True)
        out_292 = None
        x_202 = torch.conv2d(
            out_293,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_293 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        splits_32 = x_204.view(1, 2, -1, 8, 6)
        x_204 = None
        gap_160 = splits_32.sum(dim=1)
        gap_161 = torch.nn.functional.adaptive_avg_pool2d(gap_160, 1)
        gap_160 = None
        gap_162 = torch.conv2d(
            gap_161,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_161 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_163 = torch.nn.functional.batch_norm(
            gap_162,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_162 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_164 = torch.nn.functional.relu(gap_163, inplace=True)
        gap_163 = None
        atten_64 = torch.conv2d(
            gap_164,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_164 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (None)
        view_129 = atten_64.view(1, 1, 2, -1)
        atten_64 = None
        x_205 = view_129.transpose(1, 2)
        view_129 = None
        x_206 = torch.nn.functional.softmax(x_205, dim=1)
        x_205 = None
        x_207 = x_206.reshape(1, -1)
        x_206 = None
        atten_65 = x_207.view(1, -1, 1, 1)
        x_207 = None
        attens_32 = atten_65.view(1, 2, -1, 1, 1)
        atten_65 = None
        mul_32 = attens_32 * splits_32
        attens_32 = splits_32 = None
        out_294 = torch.sum(mul_32, dim=1)
        mul_32 = None
        out_295 = out_294.contiguous()
        out_294 = None
        out_296 = torch.conv2d(
            out_295,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_295 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_297 = torch.nn.functional.batch_norm(
            out_296,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_296 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_297 += out_290
        out_298 = out_297
        out_297 = out_290 = None
        out_299 = torch.nn.functional.relu(out_298, inplace=True)
        out_298 = None
        input_12 = torch.conv_transpose2d(
            out_299,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        out_299 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_14 = torch.nn.functional.relu(input_13, inplace=True)
        input_13 = None
        input_15 = torch.conv_transpose2d(
            input_14,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_14 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        input_18 = torch.conv_transpose2d(
            input_17,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_17 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_20 = torch.nn.functional.relu(input_19, inplace=True)
        input_19 = None
        x_208 = torch.conv2d(
            input_20,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_20 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_208,)
