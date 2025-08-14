import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_0_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_1_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_3_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_4_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_5_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_6_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_7_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_8_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_9_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_10_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_11_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_12_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_13_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_14_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_15_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_16_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_17_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_18_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_19_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_20_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_21_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_22_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_23_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_24_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_25_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_2_modules_26_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_bias_
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_0_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_0_modules_0_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_1_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_0_modules_1_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_0_modules_2_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_0_modules_2_parameters_gamma_
        )
        l_self_modules_backbone_modules_norm0_parameters_weight_ = (
            L_self_modules_backbone_modules_norm0_parameters_weight_
        )
        l_self_modules_backbone_modules_norm0_parameters_bias_ = (
            L_self_modules_backbone_modules_norm0_parameters_bias_
        )
        l_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_bias_
        l_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_0_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_1_modules_0_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_1_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_1_modules_1_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_1_modules_2_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_1_modules_2_parameters_gamma_
        )
        l_self_modules_backbone_modules_norm1_parameters_weight_ = (
            L_self_modules_backbone_modules_norm1_parameters_weight_
        )
        l_self_modules_backbone_modules_norm1_parameters_bias_ = (
            L_self_modules_backbone_modules_norm1_parameters_bias_
        )
        l_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_bias_
        l_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_0_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_0_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_1_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_1_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_2_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_2_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_3_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_3_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_4_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_4_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_5_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_5_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_6_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_6_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_7_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_7_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_8_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_8_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_9_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_2_modules_9_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_10_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_10_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_11_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_11_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_12_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_12_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_13_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_13_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_14_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_14_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_15_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_15_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_16_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_16_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_17_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_17_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_18_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_18_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_19_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_19_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_20_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_20_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_21_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_21_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_22_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_22_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_23_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_23_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_24_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_24_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_25_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_25_parameters_gamma_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_2_modules_26_parameters_gamma_ = L_self_modules_backbone_modules_stages_modules_2_modules_26_parameters_gamma_
        l_self_modules_backbone_modules_norm2_parameters_weight_ = (
            L_self_modules_backbone_modules_norm2_parameters_weight_
        )
        l_self_modules_backbone_modules_norm2_parameters_bias_ = (
            L_self_modules_backbone_modules_norm2_parameters_bias_
        )
        l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_
        l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_ = L_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_
        l_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_ = (
            L_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_
        )
        l_self_modules_backbone_modules_norm3_parameters_weight_ = (
            L_self_modules_backbone_modules_norm3_parameters_weight_
        )
        l_self_modules_backbone_modules_norm3_parameters_bias_ = (
            L_self_modules_backbone_modules_norm3_parameters_bias_
        )
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        )
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_0_modules_0_parameters_bias_ = (None)
        x = input_1.permute(0, 2, 3, 1)
        input_1 = None
        x_1 = torch.nn.functional.layer_norm(
            x,
            (256,),
            l_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_bias_,
            1e-06,
        )
        x = l_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_0_modules_1_parameters_bias_ = (None)
        permute_1 = x_1.permute(0, 3, 1, 2)
        x_1 = None
        x_2 = permute_1.contiguous()
        permute_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_4 = x_3.permute(0, 2, 3, 1)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (256,),
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_4 = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_norm_parameters_bias_ = (None)
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_bias_,
        )
        x_5 = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv1_parameters_bias_ = (None)
        x_7 = torch._C._nn.gelu(x_6, approximate="none")
        x_6 = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_bias_,
        )
        x_7 = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_pointwise_conv2_parameters_bias_ = (None)
        x_9 = x_8.permute(0, 3, 1, 2)
        x_8 = None
        view = l_self_modules_backbone_modules_stages_modules_0_modules_0_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_0_parameters_gamma_ = (
            None
        )
        x_10 = x_9.mul(view)
        x_9 = view = None
        x_11 = x_2 + x_10
        x_2 = x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_13 = x_12.permute(0, 2, 3, 1)
        x_12 = None
        x_14 = torch.nn.functional.layer_norm(
            x_13,
            (256,),
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_13 = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_norm_parameters_bias_ = (None)
        x_15 = torch._C._nn.linear(
            x_14,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_bias_,
        )
        x_14 = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv1_parameters_bias_ = (None)
        x_16 = torch._C._nn.gelu(x_15, approximate="none")
        x_15 = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_bias_,
        )
        x_16 = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_pointwise_conv2_parameters_bias_ = (None)
        x_18 = x_17.permute(0, 3, 1, 2)
        x_17 = None
        view_1 = l_self_modules_backbone_modules_stages_modules_0_modules_1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_1_parameters_gamma_ = (
            None
        )
        x_19 = x_18.mul(view_1)
        x_18 = view_1 = None
        x_20 = x_11 + x_19
        x_11 = x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_22 = x_21.permute(0, 2, 3, 1)
        x_21 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (256,),
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_22 = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_norm_parameters_bias_ = (None)
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_bias_,
        )
        x_23 = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv1_parameters_bias_ = (None)
        x_25 = torch._C._nn.gelu(x_24, approximate="none")
        x_24 = None
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_bias_,
        )
        x_25 = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_pointwise_conv2_parameters_bias_ = (None)
        x_27 = x_26.permute(0, 3, 1, 2)
        x_26 = None
        view_2 = l_self_modules_backbone_modules_stages_modules_0_modules_2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_2_parameters_gamma_ = (
            None
        )
        x_28 = x_27.mul(view_2)
        x_27 = view_2 = None
        x_29 = x_20 + x_28
        x_20 = x_28 = None
        x_30 = x_29.permute(0, 2, 3, 1)
        x_31 = torch.nn.functional.layer_norm(
            x_30,
            (256,),
            l_self_modules_backbone_modules_norm0_parameters_weight_,
            l_self_modules_backbone_modules_norm0_parameters_bias_,
            1e-06,
        )
        x_30 = (
            l_self_modules_backbone_modules_norm0_parameters_weight_
        ) = l_self_modules_backbone_modules_norm0_parameters_bias_ = None
        permute_9 = x_31.permute(0, 3, 1, 2)
        x_31 = None
        x_32 = permute_9.contiguous()
        permute_9 = None
        x_33 = x_29.permute(0, 2, 3, 1)
        x_29 = None
        x_34 = torch.nn.functional.layer_norm(
            x_33,
            (256,),
            l_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_bias_,
            1e-06,
        )
        x_33 = l_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_1_modules_0_parameters_bias_ = (None)
        permute_11 = x_34.permute(0, 3, 1, 2)
        x_34 = None
        x_35 = permute_11.contiguous()
        permute_11 = None
        input_2 = torch.conv2d(
            x_35,
            l_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_1_modules_1_parameters_bias_ = (None)
        x_36 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_37 = x_36.permute(0, 2, 3, 1)
        x_36 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (512,),
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_37 = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_norm_parameters_bias_ = (None)
        x_39 = torch._C._nn.linear(
            x_38,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_bias_,
        )
        x_38 = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv1_parameters_bias_ = (None)
        x_40 = torch._C._nn.gelu(x_39, approximate="none")
        x_39 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_bias_,
        )
        x_40 = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_pointwise_conv2_parameters_bias_ = (None)
        x_42 = x_41.permute(0, 3, 1, 2)
        x_41 = None
        view_3 = l_self_modules_backbone_modules_stages_modules_1_modules_0_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_0_parameters_gamma_ = (
            None
        )
        x_43 = x_42.mul(view_3)
        x_42 = view_3 = None
        x_44 = input_2 + x_43
        input_2 = x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_46 = x_45.permute(0, 2, 3, 1)
        x_45 = None
        x_47 = torch.nn.functional.layer_norm(
            x_46,
            (512,),
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_46 = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_norm_parameters_bias_ = (None)
        x_48 = torch._C._nn.linear(
            x_47,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_bias_,
        )
        x_47 = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv1_parameters_bias_ = (None)
        x_49 = torch._C._nn.gelu(x_48, approximate="none")
        x_48 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_bias_,
        )
        x_49 = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_pointwise_conv2_parameters_bias_ = (None)
        x_51 = x_50.permute(0, 3, 1, 2)
        x_50 = None
        view_4 = l_self_modules_backbone_modules_stages_modules_1_modules_1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_1_parameters_gamma_ = (
            None
        )
        x_52 = x_51.mul(view_4)
        x_51 = view_4 = None
        x_53 = x_44 + x_52
        x_44 = x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            512,
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_55 = x_54.permute(0, 2, 3, 1)
        x_54 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (512,),
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_55 = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_norm_parameters_bias_ = (None)
        x_57 = torch._C._nn.linear(
            x_56,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_bias_,
        )
        x_56 = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv1_parameters_bias_ = (None)
        x_58 = torch._C._nn.gelu(x_57, approximate="none")
        x_57 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_bias_,
        )
        x_58 = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_pointwise_conv2_parameters_bias_ = (None)
        x_60 = x_59.permute(0, 3, 1, 2)
        x_59 = None
        view_5 = l_self_modules_backbone_modules_stages_modules_1_modules_2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_2_parameters_gamma_ = (
            None
        )
        x_61 = x_60.mul(view_5)
        x_60 = view_5 = None
        x_62 = x_53 + x_61
        x_53 = x_61 = None
        x_63 = x_62.permute(0, 2, 3, 1)
        x_64 = torch.nn.functional.layer_norm(
            x_63,
            (512,),
            l_self_modules_backbone_modules_norm1_parameters_weight_,
            l_self_modules_backbone_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_63 = (
            l_self_modules_backbone_modules_norm1_parameters_weight_
        ) = l_self_modules_backbone_modules_norm1_parameters_bias_ = None
        permute_19 = x_64.permute(0, 3, 1, 2)
        x_64 = None
        x_65 = permute_19.contiguous()
        permute_19 = None
        x_66 = x_62.permute(0, 2, 3, 1)
        x_62 = None
        x_67 = torch.nn.functional.layer_norm(
            x_66,
            (512,),
            l_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_bias_,
            1e-06,
        )
        x_66 = l_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_2_modules_0_parameters_bias_ = (None)
        permute_21 = x_67.permute(0, 3, 1, 2)
        x_67 = None
        x_68 = permute_21.contiguous()
        permute_21 = None
        input_3 = torch.conv2d(
            x_68,
            l_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_2_modules_1_parameters_bias_ = (None)
        x_69 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_70 = x_69.permute(0, 2, 3, 1)
        x_69 = None
        x_71 = torch.nn.functional.layer_norm(
            x_70,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_70 = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_norm_parameters_bias_ = (None)
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_bias_,
        )
        x_71 = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv1_parameters_bias_ = (None)
        x_73 = torch._C._nn.gelu(x_72, approximate="none")
        x_72 = None
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_bias_,
        )
        x_73 = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_pointwise_conv2_parameters_bias_ = (None)
        x_75 = x_74.permute(0, 3, 1, 2)
        x_74 = None
        view_6 = l_self_modules_backbone_modules_stages_modules_2_modules_0_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_0_parameters_gamma_ = (
            None
        )
        x_76 = x_75.mul(view_6)
        x_75 = view_6 = None
        x_77 = input_3 + x_76
        input_3 = x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_79 = x_78.permute(0, 2, 3, 1)
        x_78 = None
        x_80 = torch.nn.functional.layer_norm(
            x_79,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_79 = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_norm_parameters_bias_ = (None)
        x_81 = torch._C._nn.linear(
            x_80,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_bias_,
        )
        x_80 = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv1_parameters_bias_ = (None)
        x_82 = torch._C._nn.gelu(x_81, approximate="none")
        x_81 = None
        x_83 = torch._C._nn.linear(
            x_82,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_bias_,
        )
        x_82 = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_pointwise_conv2_parameters_bias_ = (None)
        x_84 = x_83.permute(0, 3, 1, 2)
        x_83 = None
        view_7 = l_self_modules_backbone_modules_stages_modules_2_modules_1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_1_parameters_gamma_ = (
            None
        )
        x_85 = x_84.mul(view_7)
        x_84 = view_7 = None
        x_86 = x_77 + x_85
        x_77 = x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_88 = x_87.permute(0, 2, 3, 1)
        x_87 = None
        x_89 = torch.nn.functional.layer_norm(
            x_88,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_88 = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_norm_parameters_bias_ = (None)
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_bias_,
        )
        x_89 = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv1_parameters_bias_ = (None)
        x_91 = torch._C._nn.gelu(x_90, approximate="none")
        x_90 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_bias_,
        )
        x_91 = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_pointwise_conv2_parameters_bias_ = (None)
        x_93 = x_92.permute(0, 3, 1, 2)
        x_92 = None
        view_8 = l_self_modules_backbone_modules_stages_modules_2_modules_2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_2_parameters_gamma_ = (
            None
        )
        x_94 = x_93.mul(view_8)
        x_93 = view_8 = None
        x_95 = x_86 + x_94
        x_86 = x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_bias_ = (None)
        x_97 = x_96.permute(0, 2, 3, 1)
        x_96 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_bias_,
            1e-06,
        )
        x_97 = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_norm_parameters_bias_ = (None)
        x_99 = torch._C._nn.linear(
            x_98,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_bias_,
        )
        x_98 = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv1_parameters_bias_ = (None)
        x_100 = torch._C._nn.gelu(x_99, approximate="none")
        x_99 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_bias_,
        )
        x_100 = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_pointwise_conv2_parameters_bias_ = (None)
        x_102 = x_101.permute(0, 3, 1, 2)
        x_101 = None
        view_9 = l_self_modules_backbone_modules_stages_modules_2_modules_3_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_3_parameters_gamma_ = (
            None
        )
        x_103 = x_102.mul(view_9)
        x_102 = view_9 = None
        x_104 = x_95 + x_103
        x_95 = x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_bias_ = (None)
        x_106 = x_105.permute(0, 2, 3, 1)
        x_105 = None
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_bias_,
            1e-06,
        )
        x_106 = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_norm_parameters_bias_ = (None)
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_bias_,
        )
        x_107 = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv1_parameters_bias_ = (None)
        x_109 = torch._C._nn.gelu(x_108, approximate="none")
        x_108 = None
        x_110 = torch._C._nn.linear(
            x_109,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_bias_,
        )
        x_109 = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_pointwise_conv2_parameters_bias_ = (None)
        x_111 = x_110.permute(0, 3, 1, 2)
        x_110 = None
        view_10 = l_self_modules_backbone_modules_stages_modules_2_modules_4_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_4_parameters_gamma_ = (
            None
        )
        x_112 = x_111.mul(view_10)
        x_111 = view_10 = None
        x_113 = x_104 + x_112
        x_104 = x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_bias_ = (None)
        x_115 = x_114.permute(0, 2, 3, 1)
        x_114 = None
        x_116 = torch.nn.functional.layer_norm(
            x_115,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_bias_,
            1e-06,
        )
        x_115 = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_norm_parameters_bias_ = (None)
        x_117 = torch._C._nn.linear(
            x_116,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_bias_,
        )
        x_116 = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv1_parameters_bias_ = (None)
        x_118 = torch._C._nn.gelu(x_117, approximate="none")
        x_117 = None
        x_119 = torch._C._nn.linear(
            x_118,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_bias_,
        )
        x_118 = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_pointwise_conv2_parameters_bias_ = (None)
        x_120 = x_119.permute(0, 3, 1, 2)
        x_119 = None
        view_11 = l_self_modules_backbone_modules_stages_modules_2_modules_5_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_5_parameters_gamma_ = (
            None
        )
        x_121 = x_120.mul(view_11)
        x_120 = view_11 = None
        x_122 = x_113 + x_121
        x_113 = x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_bias_ = (None)
        x_124 = x_123.permute(0, 2, 3, 1)
        x_123 = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_bias_,
            1e-06,
        )
        x_124 = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_norm_parameters_bias_ = (None)
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_bias_,
        )
        x_125 = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv1_parameters_bias_ = (None)
        x_127 = torch._C._nn.gelu(x_126, approximate="none")
        x_126 = None
        x_128 = torch._C._nn.linear(
            x_127,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_bias_,
        )
        x_127 = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_pointwise_conv2_parameters_bias_ = (None)
        x_129 = x_128.permute(0, 3, 1, 2)
        x_128 = None
        view_12 = l_self_modules_backbone_modules_stages_modules_2_modules_6_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_6_parameters_gamma_ = (
            None
        )
        x_130 = x_129.mul(view_12)
        x_129 = view_12 = None
        x_131 = x_122 + x_130
        x_122 = x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_bias_ = (None)
        x_133 = x_132.permute(0, 2, 3, 1)
        x_132 = None
        x_134 = torch.nn.functional.layer_norm(
            x_133,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_bias_,
            1e-06,
        )
        x_133 = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_norm_parameters_bias_ = (None)
        x_135 = torch._C._nn.linear(
            x_134,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_bias_,
        )
        x_134 = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv1_parameters_bias_ = (None)
        x_136 = torch._C._nn.gelu(x_135, approximate="none")
        x_135 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_bias_,
        )
        x_136 = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_pointwise_conv2_parameters_bias_ = (None)
        x_138 = x_137.permute(0, 3, 1, 2)
        x_137 = None
        view_13 = l_self_modules_backbone_modules_stages_modules_2_modules_7_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_7_parameters_gamma_ = (
            None
        )
        x_139 = x_138.mul(view_13)
        x_138 = view_13 = None
        x_140 = x_131 + x_139
        x_131 = x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_bias_ = (None)
        x_142 = x_141.permute(0, 2, 3, 1)
        x_141 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_bias_,
            1e-06,
        )
        x_142 = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_norm_parameters_bias_ = (None)
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_bias_,
        )
        x_143 = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv1_parameters_bias_ = (None)
        x_145 = torch._C._nn.gelu(x_144, approximate="none")
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_bias_,
        )
        x_145 = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_pointwise_conv2_parameters_bias_ = (None)
        x_147 = x_146.permute(0, 3, 1, 2)
        x_146 = None
        view_14 = l_self_modules_backbone_modules_stages_modules_2_modules_8_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_8_parameters_gamma_ = (
            None
        )
        x_148 = x_147.mul(view_14)
        x_147 = view_14 = None
        x_149 = x_140 + x_148
        x_140 = x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_depthwise_conv_parameters_bias_ = (None)
        x_151 = x_150.permute(0, 2, 3, 1)
        x_150 = None
        x_152 = torch.nn.functional.layer_norm(
            x_151,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_bias_,
            1e-06,
        )
        x_151 = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_norm_parameters_bias_ = (None)
        x_153 = torch._C._nn.linear(
            x_152,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_bias_,
        )
        x_152 = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv1_parameters_bias_ = (None)
        x_154 = torch._C._nn.gelu(x_153, approximate="none")
        x_153 = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_bias_,
        )
        x_154 = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_9_modules_pointwise_conv2_parameters_bias_ = (None)
        x_156 = x_155.permute(0, 3, 1, 2)
        x_155 = None
        view_15 = l_self_modules_backbone_modules_stages_modules_2_modules_9_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_9_parameters_gamma_ = (
            None
        )
        x_157 = x_156.mul(view_15)
        x_156 = view_15 = None
        x_158 = x_149 + x_157
        x_149 = x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_depthwise_conv_parameters_bias_ = (None)
        x_160 = x_159.permute(0, 2, 3, 1)
        x_159 = None
        x_161 = torch.nn.functional.layer_norm(
            x_160,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_bias_,
            1e-06,
        )
        x_160 = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_norm_parameters_bias_ = (None)
        x_162 = torch._C._nn.linear(
            x_161,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_bias_,
        )
        x_161 = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv1_parameters_bias_ = (None)
        x_163 = torch._C._nn.gelu(x_162, approximate="none")
        x_162 = None
        x_164 = torch._C._nn.linear(
            x_163,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_bias_,
        )
        x_163 = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_10_modules_pointwise_conv2_parameters_bias_ = (None)
        x_165 = x_164.permute(0, 3, 1, 2)
        x_164 = None
        view_16 = l_self_modules_backbone_modules_stages_modules_2_modules_10_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_10_parameters_gamma_ = (
            None
        )
        x_166 = x_165.mul(view_16)
        x_165 = view_16 = None
        x_167 = x_158 + x_166
        x_158 = x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_depthwise_conv_parameters_bias_ = (None)
        x_169 = x_168.permute(0, 2, 3, 1)
        x_168 = None
        x_170 = torch.nn.functional.layer_norm(
            x_169,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_bias_,
            1e-06,
        )
        x_169 = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_norm_parameters_bias_ = (None)
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_bias_,
        )
        x_170 = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv1_parameters_bias_ = (None)
        x_172 = torch._C._nn.gelu(x_171, approximate="none")
        x_171 = None
        x_173 = torch._C._nn.linear(
            x_172,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_bias_,
        )
        x_172 = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_11_modules_pointwise_conv2_parameters_bias_ = (None)
        x_174 = x_173.permute(0, 3, 1, 2)
        x_173 = None
        view_17 = l_self_modules_backbone_modules_stages_modules_2_modules_11_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_11_parameters_gamma_ = (
            None
        )
        x_175 = x_174.mul(view_17)
        x_174 = view_17 = None
        x_176 = x_167 + x_175
        x_167 = x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_depthwise_conv_parameters_bias_ = (None)
        x_178 = x_177.permute(0, 2, 3, 1)
        x_177 = None
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_bias_,
            1e-06,
        )
        x_178 = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_norm_parameters_bias_ = (None)
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_bias_,
        )
        x_179 = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv1_parameters_bias_ = (None)
        x_181 = torch._C._nn.gelu(x_180, approximate="none")
        x_180 = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_bias_,
        )
        x_181 = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_12_modules_pointwise_conv2_parameters_bias_ = (None)
        x_183 = x_182.permute(0, 3, 1, 2)
        x_182 = None
        view_18 = l_self_modules_backbone_modules_stages_modules_2_modules_12_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_12_parameters_gamma_ = (
            None
        )
        x_184 = x_183.mul(view_18)
        x_183 = view_18 = None
        x_185 = x_176 + x_184
        x_176 = x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_depthwise_conv_parameters_bias_ = (None)
        x_187 = x_186.permute(0, 2, 3, 1)
        x_186 = None
        x_188 = torch.nn.functional.layer_norm(
            x_187,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_bias_,
            1e-06,
        )
        x_187 = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_norm_parameters_bias_ = (None)
        x_189 = torch._C._nn.linear(
            x_188,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_bias_,
        )
        x_188 = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv1_parameters_bias_ = (None)
        x_190 = torch._C._nn.gelu(x_189, approximate="none")
        x_189 = None
        x_191 = torch._C._nn.linear(
            x_190,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_bias_,
        )
        x_190 = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_13_modules_pointwise_conv2_parameters_bias_ = (None)
        x_192 = x_191.permute(0, 3, 1, 2)
        x_191 = None
        view_19 = l_self_modules_backbone_modules_stages_modules_2_modules_13_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_13_parameters_gamma_ = (
            None
        )
        x_193 = x_192.mul(view_19)
        x_192 = view_19 = None
        x_194 = x_185 + x_193
        x_185 = x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_depthwise_conv_parameters_bias_ = (None)
        x_196 = x_195.permute(0, 2, 3, 1)
        x_195 = None
        x_197 = torch.nn.functional.layer_norm(
            x_196,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_bias_,
            1e-06,
        )
        x_196 = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_norm_parameters_bias_ = (None)
        x_198 = torch._C._nn.linear(
            x_197,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_bias_,
        )
        x_197 = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv1_parameters_bias_ = (None)
        x_199 = torch._C._nn.gelu(x_198, approximate="none")
        x_198 = None
        x_200 = torch._C._nn.linear(
            x_199,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_bias_,
        )
        x_199 = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_14_modules_pointwise_conv2_parameters_bias_ = (None)
        x_201 = x_200.permute(0, 3, 1, 2)
        x_200 = None
        view_20 = l_self_modules_backbone_modules_stages_modules_2_modules_14_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_14_parameters_gamma_ = (
            None
        )
        x_202 = x_201.mul(view_20)
        x_201 = view_20 = None
        x_203 = x_194 + x_202
        x_194 = x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_depthwise_conv_parameters_bias_ = (None)
        x_205 = x_204.permute(0, 2, 3, 1)
        x_204 = None
        x_206 = torch.nn.functional.layer_norm(
            x_205,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_bias_,
            1e-06,
        )
        x_205 = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_norm_parameters_bias_ = (None)
        x_207 = torch._C._nn.linear(
            x_206,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_bias_,
        )
        x_206 = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv1_parameters_bias_ = (None)
        x_208 = torch._C._nn.gelu(x_207, approximate="none")
        x_207 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_bias_,
        )
        x_208 = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_15_modules_pointwise_conv2_parameters_bias_ = (None)
        x_210 = x_209.permute(0, 3, 1, 2)
        x_209 = None
        view_21 = l_self_modules_backbone_modules_stages_modules_2_modules_15_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_15_parameters_gamma_ = (
            None
        )
        x_211 = x_210.mul(view_21)
        x_210 = view_21 = None
        x_212 = x_203 + x_211
        x_203 = x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_depthwise_conv_parameters_bias_ = (None)
        x_214 = x_213.permute(0, 2, 3, 1)
        x_213 = None
        x_215 = torch.nn.functional.layer_norm(
            x_214,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_bias_,
            1e-06,
        )
        x_214 = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_norm_parameters_bias_ = (None)
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_bias_,
        )
        x_215 = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv1_parameters_bias_ = (None)
        x_217 = torch._C._nn.gelu(x_216, approximate="none")
        x_216 = None
        x_218 = torch._C._nn.linear(
            x_217,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_bias_,
        )
        x_217 = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_16_modules_pointwise_conv2_parameters_bias_ = (None)
        x_219 = x_218.permute(0, 3, 1, 2)
        x_218 = None
        view_22 = l_self_modules_backbone_modules_stages_modules_2_modules_16_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_16_parameters_gamma_ = (
            None
        )
        x_220 = x_219.mul(view_22)
        x_219 = view_22 = None
        x_221 = x_212 + x_220
        x_212 = x_220 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_depthwise_conv_parameters_bias_ = (None)
        x_223 = x_222.permute(0, 2, 3, 1)
        x_222 = None
        x_224 = torch.nn.functional.layer_norm(
            x_223,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_bias_,
            1e-06,
        )
        x_223 = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_norm_parameters_bias_ = (None)
        x_225 = torch._C._nn.linear(
            x_224,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_bias_,
        )
        x_224 = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv1_parameters_bias_ = (None)
        x_226 = torch._C._nn.gelu(x_225, approximate="none")
        x_225 = None
        x_227 = torch._C._nn.linear(
            x_226,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_bias_,
        )
        x_226 = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_17_modules_pointwise_conv2_parameters_bias_ = (None)
        x_228 = x_227.permute(0, 3, 1, 2)
        x_227 = None
        view_23 = l_self_modules_backbone_modules_stages_modules_2_modules_17_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_17_parameters_gamma_ = (
            None
        )
        x_229 = x_228.mul(view_23)
        x_228 = view_23 = None
        x_230 = x_221 + x_229
        x_221 = x_229 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_depthwise_conv_parameters_bias_ = (None)
        x_232 = x_231.permute(0, 2, 3, 1)
        x_231 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_bias_,
            1e-06,
        )
        x_232 = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_norm_parameters_bias_ = (None)
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_bias_,
        )
        x_233 = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv1_parameters_bias_ = (None)
        x_235 = torch._C._nn.gelu(x_234, approximate="none")
        x_234 = None
        x_236 = torch._C._nn.linear(
            x_235,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_bias_,
        )
        x_235 = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_18_modules_pointwise_conv2_parameters_bias_ = (None)
        x_237 = x_236.permute(0, 3, 1, 2)
        x_236 = None
        view_24 = l_self_modules_backbone_modules_stages_modules_2_modules_18_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_18_parameters_gamma_ = (
            None
        )
        x_238 = x_237.mul(view_24)
        x_237 = view_24 = None
        x_239 = x_230 + x_238
        x_230 = x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_depthwise_conv_parameters_bias_ = (None)
        x_241 = x_240.permute(0, 2, 3, 1)
        x_240 = None
        x_242 = torch.nn.functional.layer_norm(
            x_241,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_bias_,
            1e-06,
        )
        x_241 = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_norm_parameters_bias_ = (None)
        x_243 = torch._C._nn.linear(
            x_242,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_bias_,
        )
        x_242 = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv1_parameters_bias_ = (None)
        x_244 = torch._C._nn.gelu(x_243, approximate="none")
        x_243 = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_bias_,
        )
        x_244 = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_19_modules_pointwise_conv2_parameters_bias_ = (None)
        x_246 = x_245.permute(0, 3, 1, 2)
        x_245 = None
        view_25 = l_self_modules_backbone_modules_stages_modules_2_modules_19_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_19_parameters_gamma_ = (
            None
        )
        x_247 = x_246.mul(view_25)
        x_246 = view_25 = None
        x_248 = x_239 + x_247
        x_239 = x_247 = None
        x_249 = torch.conv2d(
            x_248,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_depthwise_conv_parameters_bias_ = (None)
        x_250 = x_249.permute(0, 2, 3, 1)
        x_249 = None
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_bias_,
            1e-06,
        )
        x_250 = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_norm_parameters_bias_ = (None)
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_bias_,
        )
        x_251 = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv1_parameters_bias_ = (None)
        x_253 = torch._C._nn.gelu(x_252, approximate="none")
        x_252 = None
        x_254 = torch._C._nn.linear(
            x_253,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_bias_,
        )
        x_253 = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_20_modules_pointwise_conv2_parameters_bias_ = (None)
        x_255 = x_254.permute(0, 3, 1, 2)
        x_254 = None
        view_26 = l_self_modules_backbone_modules_stages_modules_2_modules_20_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_20_parameters_gamma_ = (
            None
        )
        x_256 = x_255.mul(view_26)
        x_255 = view_26 = None
        x_257 = x_248 + x_256
        x_248 = x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_depthwise_conv_parameters_bias_ = (None)
        x_259 = x_258.permute(0, 2, 3, 1)
        x_258 = None
        x_260 = torch.nn.functional.layer_norm(
            x_259,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_bias_,
            1e-06,
        )
        x_259 = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_norm_parameters_bias_ = (None)
        x_261 = torch._C._nn.linear(
            x_260,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_bias_,
        )
        x_260 = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv1_parameters_bias_ = (None)
        x_262 = torch._C._nn.gelu(x_261, approximate="none")
        x_261 = None
        x_263 = torch._C._nn.linear(
            x_262,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_bias_,
        )
        x_262 = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_21_modules_pointwise_conv2_parameters_bias_ = (None)
        x_264 = x_263.permute(0, 3, 1, 2)
        x_263 = None
        view_27 = l_self_modules_backbone_modules_stages_modules_2_modules_21_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_21_parameters_gamma_ = (
            None
        )
        x_265 = x_264.mul(view_27)
        x_264 = view_27 = None
        x_266 = x_257 + x_265
        x_257 = x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_depthwise_conv_parameters_bias_ = (None)
        x_268 = x_267.permute(0, 2, 3, 1)
        x_267 = None
        x_269 = torch.nn.functional.layer_norm(
            x_268,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_bias_,
            1e-06,
        )
        x_268 = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_norm_parameters_bias_ = (None)
        x_270 = torch._C._nn.linear(
            x_269,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_bias_,
        )
        x_269 = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv1_parameters_bias_ = (None)
        x_271 = torch._C._nn.gelu(x_270, approximate="none")
        x_270 = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_bias_,
        )
        x_271 = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_22_modules_pointwise_conv2_parameters_bias_ = (None)
        x_273 = x_272.permute(0, 3, 1, 2)
        x_272 = None
        view_28 = l_self_modules_backbone_modules_stages_modules_2_modules_22_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_22_parameters_gamma_ = (
            None
        )
        x_274 = x_273.mul(view_28)
        x_273 = view_28 = None
        x_275 = x_266 + x_274
        x_266 = x_274 = None
        x_276 = torch.conv2d(
            x_275,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_depthwise_conv_parameters_bias_ = (None)
        x_277 = x_276.permute(0, 2, 3, 1)
        x_276 = None
        x_278 = torch.nn.functional.layer_norm(
            x_277,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_bias_,
            1e-06,
        )
        x_277 = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_norm_parameters_bias_ = (None)
        x_279 = torch._C._nn.linear(
            x_278,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_bias_,
        )
        x_278 = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv1_parameters_bias_ = (None)
        x_280 = torch._C._nn.gelu(x_279, approximate="none")
        x_279 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_bias_,
        )
        x_280 = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_23_modules_pointwise_conv2_parameters_bias_ = (None)
        x_282 = x_281.permute(0, 3, 1, 2)
        x_281 = None
        view_29 = l_self_modules_backbone_modules_stages_modules_2_modules_23_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_23_parameters_gamma_ = (
            None
        )
        x_283 = x_282.mul(view_29)
        x_282 = view_29 = None
        x_284 = x_275 + x_283
        x_275 = x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_depthwise_conv_parameters_bias_ = (None)
        x_286 = x_285.permute(0, 2, 3, 1)
        x_285 = None
        x_287 = torch.nn.functional.layer_norm(
            x_286,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_bias_,
            1e-06,
        )
        x_286 = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_norm_parameters_bias_ = (None)
        x_288 = torch._C._nn.linear(
            x_287,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_bias_,
        )
        x_287 = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv1_parameters_bias_ = (None)
        x_289 = torch._C._nn.gelu(x_288, approximate="none")
        x_288 = None
        x_290 = torch._C._nn.linear(
            x_289,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_bias_,
        )
        x_289 = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_24_modules_pointwise_conv2_parameters_bias_ = (None)
        x_291 = x_290.permute(0, 3, 1, 2)
        x_290 = None
        view_30 = l_self_modules_backbone_modules_stages_modules_2_modules_24_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_24_parameters_gamma_ = (
            None
        )
        x_292 = x_291.mul(view_30)
        x_291 = view_30 = None
        x_293 = x_284 + x_292
        x_284 = x_292 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_depthwise_conv_parameters_bias_ = (None)
        x_295 = x_294.permute(0, 2, 3, 1)
        x_294 = None
        x_296 = torch.nn.functional.layer_norm(
            x_295,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_bias_,
            1e-06,
        )
        x_295 = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_norm_parameters_bias_ = (None)
        x_297 = torch._C._nn.linear(
            x_296,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_bias_,
        )
        x_296 = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv1_parameters_bias_ = (None)
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        x_299 = torch._C._nn.linear(
            x_298,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_bias_,
        )
        x_298 = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_25_modules_pointwise_conv2_parameters_bias_ = (None)
        x_300 = x_299.permute(0, 3, 1, 2)
        x_299 = None
        view_31 = l_self_modules_backbone_modules_stages_modules_2_modules_25_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_25_parameters_gamma_ = (
            None
        )
        x_301 = x_300.mul(view_31)
        x_300 = view_31 = None
        x_302 = x_293 + x_301
        x_293 = x_301 = None
        x_303 = torch.conv2d(
            x_302,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1024,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_depthwise_conv_parameters_bias_ = (None)
        x_304 = x_303.permute(0, 2, 3, 1)
        x_303 = None
        x_305 = torch.nn.functional.layer_norm(
            x_304,
            (1024,),
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_bias_,
            1e-06,
        )
        x_304 = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_norm_parameters_bias_ = (None)
        x_306 = torch._C._nn.linear(
            x_305,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_bias_,
        )
        x_305 = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv1_parameters_bias_ = (None)
        x_307 = torch._C._nn.gelu(x_306, approximate="none")
        x_306 = None
        x_308 = torch._C._nn.linear(
            x_307,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_bias_,
        )
        x_307 = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_26_modules_pointwise_conv2_parameters_bias_ = (None)
        x_309 = x_308.permute(0, 3, 1, 2)
        x_308 = None
        view_32 = l_self_modules_backbone_modules_stages_modules_2_modules_26_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_26_parameters_gamma_ = (
            None
        )
        x_310 = x_309.mul(view_32)
        x_309 = view_32 = None
        x_311 = x_302 + x_310
        x_302 = x_310 = None
        x_312 = x_311.permute(0, 2, 3, 1)
        x_313 = torch.nn.functional.layer_norm(
            x_312,
            (1024,),
            l_self_modules_backbone_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_312 = (
            l_self_modules_backbone_modules_norm2_parameters_weight_
        ) = l_self_modules_backbone_modules_norm2_parameters_bias_ = None
        permute_77 = x_313.permute(0, 3, 1, 2)
        x_313 = None
        x_314 = permute_77.contiguous()
        permute_77 = None
        x_315 = x_311.permute(0, 2, 3, 1)
        x_311 = None
        x_316 = torch.nn.functional.layer_norm(
            x_315,
            (1024,),
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_,
            1e-06,
        )
        x_315 = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_ = (None)
        permute_79 = x_316.permute(0, 3, 1, 2)
        x_316 = None
        x_317 = permute_79.contiguous()
        permute_79 = None
        input_4 = torch.conv2d(
            x_317,
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_317 = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_ = (None)
        x_318 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            2048,
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_319 = x_318.permute(0, 2, 3, 1)
        x_318 = None
        x_320 = torch.nn.functional.layer_norm(
            x_319,
            (2048,),
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_319 = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_ = (None)
        x_321 = torch._C._nn.linear(
            x_320,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_,
        )
        x_320 = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_ = (None)
        x_322 = torch._C._nn.gelu(x_321, approximate="none")
        x_321 = None
        x_323 = torch._C._nn.linear(
            x_322,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_,
        )
        x_322 = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_ = (None)
        x_324 = x_323.permute(0, 3, 1, 2)
        x_323 = None
        view_33 = l_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_ = (
            None
        )
        x_325 = x_324.mul(view_33)
        x_324 = view_33 = None
        x_326 = input_4 + x_325
        input_4 = x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            2048,
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_328 = x_327.permute(0, 2, 3, 1)
        x_327 = None
        x_329 = torch.nn.functional.layer_norm(
            x_328,
            (2048,),
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_328 = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_ = (None)
        x_330 = torch._C._nn.linear(
            x_329,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_,
        )
        x_329 = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_ = (None)
        x_331 = torch._C._nn.gelu(x_330, approximate="none")
        x_330 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_,
        )
        x_331 = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_ = (None)
        x_333 = x_332.permute(0, 3, 1, 2)
        x_332 = None
        view_34 = l_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_ = (
            None
        )
        x_334 = x_333.mul(view_34)
        x_333 = view_34 = None
        x_335 = x_326 + x_334
        x_326 = x_334 = None
        x_336 = torch.conv2d(
            x_335,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            2048,
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_337 = x_336.permute(0, 2, 3, 1)
        x_336 = None
        x_338 = torch.nn.functional.layer_norm(
            x_337,
            (2048,),
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_337 = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_ = (None)
        x_339 = torch._C._nn.linear(
            x_338,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_,
        )
        x_338 = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_ = (None)
        x_340 = torch._C._nn.gelu(x_339, approximate="none")
        x_339 = None
        x_341 = torch._C._nn.linear(
            x_340,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_,
        )
        x_340 = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_ = (None)
        x_342 = x_341.permute(0, 3, 1, 2)
        x_341 = None
        view_35 = l_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_ = (
            None
        )
        x_343 = x_342.mul(view_35)
        x_342 = view_35 = None
        x_344 = x_335 + x_343
        x_335 = x_343 = None
        x_345 = x_344.permute(0, 2, 3, 1)
        x_344 = None
        x_346 = torch.nn.functional.layer_norm(
            x_345,
            (2048,),
            l_self_modules_backbone_modules_norm3_parameters_weight_,
            l_self_modules_backbone_modules_norm3_parameters_bias_,
            1e-06,
        )
        x_345 = (
            l_self_modules_backbone_modules_norm3_parameters_weight_
        ) = l_self_modules_backbone_modules_norm3_parameters_bias_ = None
        permute_87 = x_346.permute(0, 3, 1, 2)
        x_346 = None
        x_347 = permute_87.contiguous()
        permute_87 = None
        x_348 = torch.conv2d(
            x_32,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_348 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_350 = torch.nn.functional.relu(x_349, inplace=False)
        x_349 = None
        x_351 = torch.conv2d(
            x_65,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_352 = torch.nn.functional.batch_norm(
            x_351,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_351 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_353 = torch.nn.functional.relu(x_352, inplace=False)
        x_352 = None
        x_354 = torch.conv2d(
            x_314,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_355 = torch.nn.functional.batch_norm(
            x_354,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_354 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_356 = torch.nn.functional.relu(x_355, inplace=False)
        x_355 = None
        input_5 = torch.nn.functional.adaptive_avg_pool2d(x_347, 1)
        x_357 = torch.conv2d(
            input_5,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_358 = torch.nn.functional.batch_norm(
            x_357,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_357 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_359 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_359, (16, 16), None, "bilinear", False
        )
        x_359 = None
        input_6 = torch.nn.functional.adaptive_avg_pool2d(x_347, 2)
        x_360 = torch.conv2d(
            input_6,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_360 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_362 = torch.nn.functional.relu(x_361, inplace=True)
        x_361 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_362, (16, 16), None, "bilinear", False
        )
        x_362 = None
        input_7 = torch.nn.functional.adaptive_avg_pool2d(x_347, 3)
        x_363 = torch.conv2d(
            input_7,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_363 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_365 = torch.nn.functional.relu(x_364, inplace=True)
        x_364 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_365, (16, 16), None, "bilinear", False
        )
        x_365 = None
        input_8 = torch.nn.functional.adaptive_avg_pool2d(x_347, 6)
        x_366 = torch.conv2d(
            input_8,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_367 = torch.nn.functional.batch_norm(
            x_366,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_366 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_368 = torch.nn.functional.relu(x_367, inplace=True)
        x_367 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_368, (16, 16), None, "bilinear", False
        )
        x_368 = None
        psp_outs = torch.cat(
            [
                x_347,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        x_347 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_369 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_371 = torch.nn.functional.relu(x_370, inplace=True)
        x_370 = None
        interpolate_4 = torch.nn.functional.interpolate(
            x_371, (32, 32), None, "bilinear", False
        )
        add_36 = x_356 + interpolate_4
        x_356 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add_36, (64, 64), None, "bilinear", False
        )
        add_37 = x_353 + interpolate_5
        x_353 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_37, (128, 128), None, "bilinear", False
        )
        add_38 = x_350 + interpolate_6
        x_350 = interpolate_6 = None
        x_372 = torch.conv2d(
            add_38,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_38 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_373 = torch.nn.functional.batch_norm(
            x_372,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_372 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_374 = torch.nn.functional.relu(x_373, inplace=False)
        x_373 = None
        x_375 = torch.conv2d(
            add_37,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_37 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_376 = torch.nn.functional.batch_norm(
            x_375,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_375 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_377 = torch.nn.functional.relu(x_376, inplace=False)
        x_376 = None
        x_378 = torch.conv2d(
            add_36,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_36 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_379 = torch.nn.functional.batch_norm(
            x_378,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_378 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_380 = torch.nn.functional.relu(x_379, inplace=False)
        x_379 = None
        interpolate_7 = torch.nn.functional.interpolate(
            x_371, (128, 128), None, "bilinear", False
        )
        x_371 = None
        interpolate_8 = torch.nn.functional.interpolate(
            x_380, (128, 128), None, "bilinear", False
        )
        x_380 = None
        interpolate_9 = torch.nn.functional.interpolate(
            x_377, (128, 128), None, "bilinear", False
        )
        x_377 = None
        fpn_outs = torch.cat(
            [x_374, interpolate_9, interpolate_8, interpolate_7], dim=1
        )
        x_374 = interpolate_9 = interpolate_8 = interpolate_7 = None
        x_381 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        x_382 = torch.nn.functional.batch_norm(
            x_381,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_381 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = (None)
        x_383 = torch.nn.functional.relu(x_382, inplace=True)
        x_382 = None
        feat = torch.nn.functional.dropout2d(x_383, 0.1, False, False)
        x_383 = None
        output = torch.conv2d(
            feat,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output,)
