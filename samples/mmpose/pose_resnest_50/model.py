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
        out_120 = torch.nn.functional.batch_norm(
            out_119,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_119 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_121 = torch.nn.functional.relu(out_120, inplace=True)
        out_120 = None
        x_88 = torch.conv2d(
            out_121,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_121 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        splits_13 = x_90.view(1, 2, -1, 16, 12)
        x_90 = None
        gap_65 = splits_13.sum(dim=1)
        gap_66 = torch.nn.functional.adaptive_avg_pool2d(gap_65, 1)
        gap_65 = None
        gap_67 = torch.conv2d(
            gap_66,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_66 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_68 = torch.nn.functional.batch_norm(
            gap_67,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_67 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_69 = torch.nn.functional.relu(gap_68, inplace=True)
        gap_68 = None
        atten_26 = torch.conv2d(
            gap_69,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_69 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_ = (None)
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
        out_124 = torch._C._nn.avg_pool2d(out_123, 3, 2, 1, False, True, None)
        out_123 = None
        out_125 = torch.conv2d(
            out_124,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_124 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_126 = torch.nn.functional.batch_norm(
            out_125,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_125 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_9 = torch._C._nn.avg_pool2d(out_118, 2, 2, 0, True, False, None)
        out_118 = None
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
        out_126 += input_11
        out_127 = out_126
        out_126 = input_11 = None
        out_128 = torch.nn.functional.relu(out_127, inplace=True)
        out_127 = None
        out_129 = torch.conv2d(
            out_128,
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
        out_130 = torch.nn.functional.batch_norm(
            out_129,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_129 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_131 = torch.nn.functional.relu(out_130, inplace=True)
        out_130 = None
        x_94 = torch.conv2d(
            out_131,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_131 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        splits_14 = x_96.view(1, 2, -1, 8, 6)
        x_96 = None
        gap_70 = splits_14.sum(dim=1)
        gap_71 = torch.nn.functional.adaptive_avg_pool2d(gap_70, 1)
        gap_70 = None
        gap_72 = torch.conv2d(
            gap_71,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_71 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_73 = torch.nn.functional.batch_norm(
            gap_72,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_72 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_74 = torch.nn.functional.relu(gap_73, inplace=True)
        gap_73 = None
        atten_28 = torch.conv2d(
            gap_74,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_74 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_ = (None)
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
        out_132 = torch.sum(mul_14, dim=1)
        mul_14 = None
        out_133 = out_132.contiguous()
        out_132 = None
        out_134 = torch.conv2d(
            out_133,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_133 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_135 = torch.nn.functional.batch_norm(
            out_134,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_134 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_135 += out_128
        out_136 = out_135
        out_135 = out_128 = None
        out_137 = torch.nn.functional.relu(out_136, inplace=True)
        out_136 = None
        out_138 = torch.conv2d(
            out_137,
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
        out_139 = torch.nn.functional.batch_norm(
            out_138,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_138 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_140 = torch.nn.functional.relu(out_139, inplace=True)
        out_139 = None
        x_100 = torch.conv2d(
            out_140,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        out_140 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        splits_15 = x_102.view(1, 2, -1, 8, 6)
        x_102 = None
        gap_75 = splits_15.sum(dim=1)
        gap_76 = torch.nn.functional.adaptive_avg_pool2d(gap_75, 1)
        gap_75 = None
        gap_77 = torch.conv2d(
            gap_76,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_76 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_ = (None)
        gap_78 = torch.nn.functional.batch_norm(
            gap_77,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        gap_77 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_ = (None)
        gap_79 = torch.nn.functional.relu(gap_78, inplace=True)
        gap_78 = None
        atten_30 = torch.conv2d(
            gap_79,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        gap_79 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_ = (None)
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
        out_141 = torch.sum(mul_15, dim=1)
        mul_15 = None
        out_142 = out_141.contiguous()
        out_141 = None
        out_143 = torch.conv2d(
            out_142,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_142 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_144 = torch.nn.functional.batch_norm(
            out_143,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_143 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_144 += out_137
        out_145 = out_144
        out_144 = out_137 = None
        out_146 = torch.nn.functional.relu(out_145, inplace=True)
        out_145 = None
        input_12 = torch.conv_transpose2d(
            out_146,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        out_146 = (
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
        x_106 = torch.conv2d(
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
        return (x_106,)
