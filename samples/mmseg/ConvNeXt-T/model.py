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
            (96,),
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
            96,
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_4 = x_3.permute(0, 2, 3, 1)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (96,),
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
            96,
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_13 = x_12.permute(0, 2, 3, 1)
        x_12 = None
        x_14 = torch.nn.functional.layer_norm(
            x_13,
            (96,),
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
            96,
        )
        l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_0_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_22 = x_21.permute(0, 2, 3, 1)
        x_21 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (96,),
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
            (96,),
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
            (96,),
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
            192,
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_37 = x_36.permute(0, 2, 3, 1)
        x_36 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (192,),
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
            192,
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_46 = x_45.permute(0, 2, 3, 1)
        x_45 = None
        x_47 = torch.nn.functional.layer_norm(
            x_46,
            (192,),
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
            192,
        )
        l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_1_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_55 = x_54.permute(0, 2, 3, 1)
        x_54 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (192,),
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
            (192,),
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
            (192,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_70 = x_69.permute(0, 2, 3, 1)
        x_69 = None
        x_71 = torch.nn.functional.layer_norm(
            x_70,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_79 = x_78.permute(0, 2, 3, 1)
        x_78 = None
        x_80 = torch.nn.functional.layer_norm(
            x_79,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_88 = x_87.permute(0, 2, 3, 1)
        x_87 = None
        x_89 = torch.nn.functional.layer_norm(
            x_88,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_3_modules_depthwise_conv_parameters_bias_ = (None)
        x_97 = x_96.permute(0, 2, 3, 1)
        x_96 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_4_modules_depthwise_conv_parameters_bias_ = (None)
        x_106 = x_105.permute(0, 2, 3, 1)
        x_105 = None
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_5_modules_depthwise_conv_parameters_bias_ = (None)
        x_115 = x_114.permute(0, 2, 3, 1)
        x_114 = None
        x_116 = torch.nn.functional.layer_norm(
            x_115,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_6_modules_depthwise_conv_parameters_bias_ = (None)
        x_124 = x_123.permute(0, 2, 3, 1)
        x_123 = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_7_modules_depthwise_conv_parameters_bias_ = (None)
        x_133 = x_132.permute(0, 2, 3, 1)
        x_132 = None
        x_134 = torch.nn.functional.layer_norm(
            x_133,
            (384,),
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
            384,
        )
        l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_2_modules_8_modules_depthwise_conv_parameters_bias_ = (None)
        x_142 = x_141.permute(0, 2, 3, 1)
        x_141 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (384,),
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
        x_150 = x_149.permute(0, 2, 3, 1)
        x_151 = torch.nn.functional.layer_norm(
            x_150,
            (384,),
            l_self_modules_backbone_modules_norm2_parameters_weight_,
            l_self_modules_backbone_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_150 = (
            l_self_modules_backbone_modules_norm2_parameters_weight_
        ) = l_self_modules_backbone_modules_norm2_parameters_bias_ = None
        permute_41 = x_151.permute(0, 3, 1, 2)
        x_151 = None
        x_152 = permute_41.contiguous()
        permute_41 = None
        x_153 = x_149.permute(0, 2, 3, 1)
        x_149 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (384,),
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_,
            1e-06,
        )
        x_153 = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_0_parameters_bias_ = (None)
        permute_43 = x_154.permute(0, 3, 1, 2)
        x_154 = None
        x_155 = permute_43.contiguous()
        permute_43 = None
        input_4 = torch.conv2d(
            x_155,
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_downsample_layers_modules_3_modules_1_parameters_bias_ = (None)
        x_156 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_depthwise_conv_parameters_bias_ = (None)
        x_157 = x_156.permute(0, 2, 3, 1)
        x_156 = None
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (768,),
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        x_157 = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_norm_parameters_bias_ = (None)
        x_159 = torch._C._nn.linear(
            x_158,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_,
        )
        x_158 = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv1_parameters_bias_ = (None)
        x_160 = torch._C._nn.gelu(x_159, approximate="none")
        x_159 = None
        x_161 = torch._C._nn.linear(
            x_160,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_,
        )
        x_160 = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_0_modules_pointwise_conv2_parameters_bias_ = (None)
        x_162 = x_161.permute(0, 3, 1, 2)
        x_161 = None
        view_15 = l_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_0_parameters_gamma_ = (
            None
        )
        x_163 = x_162.mul(view_15)
        x_162 = view_15 = None
        x_164 = input_4 + x_163
        input_4 = x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_depthwise_conv_parameters_bias_ = (None)
        x_166 = x_165.permute(0, 2, 3, 1)
        x_165 = None
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (768,),
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        x_166 = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_norm_parameters_bias_ = (None)
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_,
        )
        x_167 = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv1_parameters_bias_ = (None)
        x_169 = torch._C._nn.gelu(x_168, approximate="none")
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_,
        )
        x_169 = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_1_modules_pointwise_conv2_parameters_bias_ = (None)
        x_171 = x_170.permute(0, 3, 1, 2)
        x_170 = None
        view_16 = l_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_1_parameters_gamma_ = (
            None
        )
        x_172 = x_171.mul(view_16)
        x_171 = view_16 = None
        x_173 = x_164 + x_172
        x_164 = x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_depthwise_conv_parameters_bias_ = (None)
        x_175 = x_174.permute(0, 2, 3, 1)
        x_174 = None
        x_176 = torch.nn.functional.layer_norm(
            x_175,
            (768,),
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        x_175 = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_norm_parameters_bias_ = (None)
        x_177 = torch._C._nn.linear(
            x_176,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_,
        )
        x_176 = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv1_parameters_bias_ = (None)
        x_178 = torch._C._nn.gelu(x_177, approximate="none")
        x_177 = None
        x_179 = torch._C._nn.linear(
            x_178,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_,
            l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_,
        )
        x_178 = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_weight_ = l_self_modules_backbone_modules_stages_modules_3_modules_2_modules_pointwise_conv2_parameters_bias_ = (None)
        x_180 = x_179.permute(0, 3, 1, 2)
        x_179 = None
        view_17 = l_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_.view(
            1, -1, 1, 1
        )
        l_self_modules_backbone_modules_stages_modules_3_modules_2_parameters_gamma_ = (
            None
        )
        x_181 = x_180.mul(view_17)
        x_180 = view_17 = None
        x_182 = x_173 + x_181
        x_173 = x_181 = None
        x_183 = x_182.permute(0, 2, 3, 1)
        x_182 = None
        x_184 = torch.nn.functional.layer_norm(
            x_183,
            (768,),
            l_self_modules_backbone_modules_norm3_parameters_weight_,
            l_self_modules_backbone_modules_norm3_parameters_bias_,
            1e-06,
        )
        x_183 = (
            l_self_modules_backbone_modules_norm3_parameters_weight_
        ) = l_self_modules_backbone_modules_norm3_parameters_bias_ = None
        permute_51 = x_184.permute(0, 3, 1, 2)
        x_184 = None
        x_185 = permute_51.contiguous()
        permute_51 = None
        x_186 = torch.conv2d(
            x_32,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_188 = torch.nn.functional.relu(x_187, inplace=False)
        x_187 = None
        x_189 = torch.conv2d(
            x_65,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_191 = torch.nn.functional.relu(x_190, inplace=False)
        x_190 = None
        x_192 = torch.conv2d(
            x_152,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=False)
        x_193 = None
        input_5 = torch.nn.functional.adaptive_avg_pool2d(x_185, 1)
        x_195 = torch.conv2d(
            input_5,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_195 = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_ = (None)
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            x_197, (16, 16), None, "bilinear", False
        )
        x_197 = None
        input_6 = torch.nn.functional.adaptive_avg_pool2d(x_185, 2)
        x_198 = torch.conv2d(
            input_6,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            x_200, (16, 16), None, "bilinear", False
        )
        x_200 = None
        input_7 = torch.nn.functional.adaptive_avg_pool2d(x_185, 3)
        x_201 = torch.conv2d(
            input_7,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            x_203, (16, 16), None, "bilinear", False
        )
        x_203 = None
        input_8 = torch.nn.functional.adaptive_avg_pool2d(x_185, 6)
        x_204 = torch.conv2d(
            input_8,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            x_206, (16, 16), None, "bilinear", False
        )
        x_206 = None
        psp_outs = torch.cat(
            [
                x_185,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        x_185 = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        x_207 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        interpolate_4 = torch.nn.functional.interpolate(
            x_209, (32, 32), None, "bilinear", False
        )
        add_18 = x_194 + interpolate_4
        x_194 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add_18, (64, 64), None, "bilinear", False
        )
        add_19 = x_191 + interpolate_5
        x_191 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_19, (128, 128), None, "bilinear", False
        )
        add_20 = x_188 + interpolate_6
        x_188 = interpolate_6 = None
        x_210 = torch.conv2d(
            add_20,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_20 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_212 = torch.nn.functional.relu(x_211, inplace=False)
        x_211 = None
        x_213 = torch.conv2d(
            add_19,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_19 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_213 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_ = (None)
        x_215 = torch.nn.functional.relu(x_214, inplace=False)
        x_214 = None
        x_216 = torch.conv2d(
            add_18,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_18 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_ = (None)
        x_218 = torch.nn.functional.relu(x_217, inplace=False)
        x_217 = None
        interpolate_7 = torch.nn.functional.interpolate(
            x_209, (128, 128), None, "bilinear", False
        )
        x_209 = None
        interpolate_8 = torch.nn.functional.interpolate(
            x_218, (128, 128), None, "bilinear", False
        )
        x_218 = None
        interpolate_9 = torch.nn.functional.interpolate(
            x_215, (128, 128), None, "bilinear", False
        )
        x_215 = None
        fpn_outs = torch.cat(
            [x_212, interpolate_9, interpolate_8, interpolate_7], dim=1
        )
        x_212 = interpolate_9 = interpolate_8 = interpolate_7 = None
        x_219 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        x_220 = torch.nn.functional.batch_norm(
            x_219,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_219 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_ = (None)
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        feat = torch.nn.functional.dropout2d(x_221, 0.1, False, False)
        x_221 = None
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
