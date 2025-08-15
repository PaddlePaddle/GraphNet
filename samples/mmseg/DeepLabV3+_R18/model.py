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
        L_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_c1_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        )
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_c1_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_c1_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
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
            (1, 1),
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
        out_4 += x
        out_5 = out_4
        out_4 = x = None
        out_6 = torch.nn.functional.relu(out_5, inplace=True)
        out_5 = None
        out_7 = torch.conv2d(
            out_6,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_8 = torch.nn.functional.batch_norm(
            out_7,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_7 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_9 = torch.nn.functional.relu(out_8, inplace=True)
        out_8 = None
        out_10 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_9 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (None)
        out_11 = torch.nn.functional.batch_norm(
            out_10,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_10 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (None)
        out_11 += out_6
        out_12 = out_11
        out_11 = out_6 = None
        out_13 = torch.nn.functional.relu(out_12, inplace=True)
        out_12 = None
        out_14 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_16 = torch.nn.functional.relu(out_15, inplace=True)
        out_15 = None
        out_17 = torch.conv2d(
            out_16,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_16 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (None)
        out_18 = torch.nn.functional.batch_norm(
            out_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_17 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (None)
        input_10 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (
            None
        )
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_18 += input_11
        out_19 = out_18
        out_18 = input_11 = None
        out_20 = torch.nn.functional.relu(out_19, inplace=True)
        out_19 = None
        out_21 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_22 = torch.nn.functional.batch_norm(
            out_21,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_21 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_23 = torch.nn.functional.relu(out_22, inplace=True)
        out_22 = None
        out_24 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_23 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (None)
        out_25 = torch.nn.functional.batch_norm(
            out_24,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_24 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (None)
        out_25 += out_20
        out_26 = out_25
        out_25 = out_20 = None
        out_27 = torch.nn.functional.relu(out_26, inplace=True)
        out_26 = None
        out_28 = torch.conv2d(
            out_27,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_29 = torch.nn.functional.batch_norm(
            out_28,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_28 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_30 = torch.nn.functional.relu(out_29, inplace=True)
        out_29 = None
        out_31 = torch.conv2d(
            out_30,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_30 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (None)
        out_32 = torch.nn.functional.batch_norm(
            out_31,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_31 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        input_12 = torch.conv2d(
            out_27,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_27 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_32 += input_13
        out_33 = out_32
        out_32 = input_13 = None
        out_34 = torch.nn.functional.relu(out_33, inplace=True)
        out_33 = None
        out_35 = torch.conv2d(
            out_34,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_36 = torch.nn.functional.batch_norm(
            out_35,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_35 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_37 = torch.nn.functional.relu(out_36, inplace=True)
        out_36 = None
        out_38 = torch.conv2d(
            out_37,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_37 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (None)
        out_39 = torch.nn.functional.batch_norm(
            out_38,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_38 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_39 += out_34
        out_40 = out_39
        out_39 = out_34 = None
        out_41 = torch.nn.functional.relu(out_40, inplace=True)
        out_40 = None
        out_42 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_43 = torch.nn.functional.batch_norm(
            out_42,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_42 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_44 = torch.nn.functional.relu(out_43, inplace=True)
        out_43 = None
        out_45 = torch.conv2d(
            out_44,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_44 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (None)
        out_46 = torch.nn.functional.batch_norm(
            out_45,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_45 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        input_14 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_46 += input_15
        out_47 = out_46
        out_46 = input_15 = None
        out_48 = torch.nn.functional.relu(out_47, inplace=True)
        out_47 = None
        out_49 = torch.conv2d(
            out_48,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_50 = torch.nn.functional.batch_norm(
            out_49,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_49 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_51 = torch.nn.functional.relu(out_50, inplace=True)
        out_50 = None
        out_52 = torch.conv2d(
            out_51,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_51 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (None)
        out_53 = torch.nn.functional.batch_norm(
            out_52,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_52 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_53 += out_48
        out_54 = out_53
        out_53 = out_48 = None
        out_55 = torch.nn.functional.relu(out_54, inplace=True)
        out_54 = None
        input_16 = torch.nn.functional.adaptive_avg_pool2d(out_55, 1)
        x_1 = torch.conv2d(
            input_16,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_ = (None)
        x_2 = torch.nn.functional.batch_norm(
            x_1,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_1 = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_ = (None)
        x_3 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        interpolate = torch.nn.functional.interpolate(
            x_3, (64, 64), None, "bilinear", False
        )
        x_3 = None
        x_4 = torch.conv2d(
            out_55,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            out_55,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (12, 12),
            (12, 12),
            512,
        )
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.conv2d(
            out_55,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (24, 24),
            (24, 24),
            512,
        )
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_pointwise_conv_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            out_55,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (36, 36),
            (36, 36),
            512,
        )
        out_55 = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_pointwise_conv_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        aspp_outs = torch.cat([interpolate, x_6, x_12, x_18, x_24], dim=1)
        interpolate = x_6 = x_12 = x_18 = x_24 = None
        x_25 = torch.conv2d(
            aspp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        aspp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            out_13,
            l_self_modules_decode_head_modules_c1_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_decode_head_modules_c1_bottleneck_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_weight_ = (
            l_self_modules_decode_head_modules_c1_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        output = torch.nn.functional.interpolate(
            x_27, (128, 128), None, "bilinear", False
        )
        x_27 = None
        output_1 = torch.cat([output, x_30], dim=1)
        output = x_30 = None
        x_31 = torch.conv2d(
            output_1,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            140,
        )
        output_1 = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_0_modules_pointwise_conv_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        x_36 = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_sep_bottleneck_modules_1_modules_pointwise_conv_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        feat = torch.nn.functional.dropout2d(x_42, 0.1, False, False)
        x_42 = None
        output_2 = torch.conv2d(
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
        return (output_2,)
