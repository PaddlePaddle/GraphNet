import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_layer0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_up_input_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_up_input_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_layer0_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_layer0_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_layer0_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_layer0_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_layer0_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_layer0_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_layer0_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_layer0_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_layer0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_layer0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer12_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_layer12_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_layer12_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_layer12_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_layer12_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_layer12_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_layer12_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_layer12_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_layer12_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_layer12_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_aspp_conv_modules_conv_parameters_weight_ = (
            L_self_modules_decode_head_modules_aspp_conv_modules_conv_parameters_weight_
        )
        l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_ = (
            L_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_
        )
        l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_ = (
            L_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_
        )
        l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_conv_up_input_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_up_input_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_up_input_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_up_input_parameters_bias_
        )
        l_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_ = (
            L_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_ = (
            L_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch._C._nn.pad(l_inputs_, [0, 1, 0, 1], "constant", None)
        l_inputs_ = None
        x_1 = torch.conv2d(
            x,
            l_self_modules_backbone_modules_layer0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x = (
            l_self_modules_backbone_modules_layer0_modules_conv_parameters_weight_
        ) = None
        x_2 = torch.nn.functional.batch_norm(
            x_1,
            l_self_modules_backbone_modules_layer0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_1 = (
            l_self_modules_backbone_modules_layer0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_layer0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_layer0_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_layer0_modules_bn_parameters_bias_ = None
        x_3 = torch.nn.functional.hardswish(x_2, True)
        x_2 = None
        x_4 = torch._C._nn.pad(x_3, [0, 1, 0, 1], "constant", None)
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            16,
        )
        x_4 = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_6 = torch.nn.functional.batch_norm(
            x_5,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_5 = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        out = torch.nn.functional.adaptive_avg_pool2d(x_7, 1)
        x_8 = torch.conv2d(
            out,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add = x_10 + 3.0
        x_10 = None
        x_11 = add / 6.0
        add = None
        x_12 = x_11.clamp_(0.0, 1.0)
        x_11 = None
        out_1 = x_7 * x_12
        x_7 = x_12 = None
        x_13 = torch.conv2d(
            out_1,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_1 = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_13 = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_15 = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch._C._nn.pad(x_17, [0, 1, 0, 1], "constant", None)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            72,
        )
        x_18 = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_19 = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_22 = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_24 = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        x_26 = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_27 = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_30 = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_2 = x_23 + x_31
        x_23 = x_31 = None
        x_32 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_2 = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_32 = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.hardswish(x_33, True)
        x_33 = None
        x_35 = torch._C._nn.pad(x_34, [4, 4, 4, 4], "constant", None)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (2, 2),
            96,
        )
        x_35 = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_36 = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.hardswish(x_37, True)
        x_37 = None
        out_3 = torch.nn.functional.adaptive_avg_pool2d(x_38, 1)
        x_39 = torch.conv2d(
            out_3,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_3 = l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_2 = x_41 + 3.0
        x_41 = None
        x_42 = add_2 / 6.0
        add_2 = None
        x_43 = x_42.clamp_(0.0, 1.0)
        x_42 = None
        out_4 = x_38 * x_43
        x_38 = x_43 = None
        x_44 = torch.conv2d(
            out_4,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_4 = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_44 = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_46 = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.hardswish(x_47, True)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            240,
        )
        x_48 = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_49 = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.hardswish(x_50, True)
        x_50 = None
        out_5 = torch.nn.functional.adaptive_avg_pool2d(x_51, 1)
        x_52 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_3 = x_54 + 3.0
        x_54 = None
        x_55 = add_3 / 6.0
        add_3 = None
        x_56 = x_55.clamp_(0.0, 1.0)
        x_55 = None
        out_6 = x_51 * x_56
        x_51 = x_56 = None
        x_57 = torch.conv2d(
            out_6,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_6 = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_57 = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_7 = x_45 + x_58
        x_45 = x_58 = None
        x_59 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_59 = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.hardswish(x_60, True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            240,
        )
        x_61 = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_62 = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.hardswish(x_63, True)
        x_63 = None
        out_8 = torch.nn.functional.adaptive_avg_pool2d(x_64, 1)
        x_65 = torch.conv2d(
            out_8,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_8 = l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_66 = l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_5 = x_67 + 3.0
        x_67 = None
        x_68 = add_5 / 6.0
        add_5 = None
        x_69 = x_68.clamp_(0.0, 1.0)
        x_68 = None
        out_9 = x_64 * x_69
        x_64 = x_69 = None
        x_70 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_9 = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_70 = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_10 = out_7 + x_71
        out_7 = x_71 = None
        x_72 = torch.conv2d(
            out_10,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_10 = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_72 = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.hardswish(x_73, True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            120,
        )
        x_74 = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_75 = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.hardswish(x_76, True)
        x_76 = None
        out_11 = torch.nn.functional.adaptive_avg_pool2d(x_77, 1)
        x_78 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_11 = l_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_7 = x_80 + 3.0
        x_80 = None
        x_81 = add_7 / 6.0
        add_7 = None
        x_82 = x_81.clamp_(0.0, 1.0)
        x_81 = None
        out_12 = x_77 * x_82
        x_77 = x_82 = None
        x_83 = torch.conv2d(
            out_12,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_12 = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_83 = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_85 = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.hardswish(x_86, True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            144,
        )
        x_87 = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_88 = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.hardswish(x_89, True)
        x_89 = None
        out_13 = torch.nn.functional.adaptive_avg_pool2d(x_90, 1)
        x_91 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_8 = x_93 + 3.0
        x_93 = None
        x_94 = add_8 / 6.0
        add_8 = None
        x_95 = x_94.clamp_(0.0, 1.0)
        x_94 = None
        out_14 = x_90 * x_95
        x_90 = x_95 = None
        x_96 = torch.conv2d(
            out_14,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_14 = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_96 = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_15 = x_84 + x_97
        x_84 = x_97 = None
        x_98 = torch.conv2d(
            out_15,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_15 = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_98 = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.hardswish(x_99, True)
        x_99 = None
        x_101 = torch._C._nn.pad(x_100, [8, 8, 8, 8], "constant", None)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (4, 4),
            288,
        )
        x_101 = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_102 = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.hardswish(x_103, True)
        x_103 = None
        out_16 = torch.nn.functional.adaptive_avg_pool2d(x_104, 1)
        x_105 = torch.conv2d(
            out_16,
            l_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_16 = l_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_10 = x_107 + 3.0
        x_107 = None
        x_108 = add_10 / 6.0
        add_10 = None
        x_109 = x_108.clamp_(0.0, 1.0)
        x_108 = None
        out_17 = x_104 * x_109
        x_104 = x_109 = None
        x_110 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_110 = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_112 = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.hardswish(x_113, True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (8, 8),
            (4, 4),
            576,
        )
        x_114 = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_115 = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.hardswish(x_116, True)
        x_116 = None
        out_18 = torch.nn.functional.adaptive_avg_pool2d(x_117, 1)
        x_118 = torch.conv2d(
            out_18,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_18 = l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_11 = x_120 + 3.0
        x_120 = None
        x_121 = add_11 / 6.0
        add_11 = None
        x_122 = x_121.clamp_(0.0, 1.0)
        x_121 = None
        out_19 = x_117 * x_122
        x_117 = x_122 = None
        x_123 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_19 = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_123 = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_20 = x_111 + x_124
        x_111 = x_124 = None
        x_125 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_125 = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.hardswish(x_126, True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (8, 8),
            (4, 4),
            576,
        )
        x_127 = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_128 = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.hardswish(x_129, True)
        x_129 = None
        out_21 = torch.nn.functional.adaptive_avg_pool2d(x_130, 1)
        x_131 = torch.conv2d(
            out_21,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_13 = x_133 + 3.0
        x_133 = None
        x_134 = add_13 / 6.0
        add_13 = None
        x_135 = x_134.clamp_(0.0, 1.0)
        x_134 = None
        out_22 = x_130 * x_135
        x_130 = x_135 = None
        x_136 = torch.conv2d(
            out_22,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_22 = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_136 = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_23 = out_20 + x_137
        out_20 = x_137 = None
        x_138 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer12_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (4, 4),
            1,
        )
        out_23 = (
            l_self_modules_backbone_modules_layer12_modules_conv_parameters_weight_
        ) = None
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_backbone_modules_layer12_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_138 = (
            l_self_modules_backbone_modules_layer12_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_layer12_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_layer12_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_layer12_modules_bn_parameters_bias_ = None
        x_140 = torch.nn.functional.hardswish(x_139, True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_decode_head_modules_aspp_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_aspp_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_141 = l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_ = (
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_
        ) = None
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        input_1 = torch._C._nn.avg_pool2d(x_140, 49, (16, 20), 0, False, True, None)
        x_140 = None
        x_144 = torch.conv2d(
            input_1,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_ = (None)
        x_145 = torch.sigmoid(x_144)
        x_144 = None
        interpolate = torch.nn.functional.interpolate(
            x_145, (64, 128), None, "bilinear", False
        )
        x_145 = None
        x_146 = x_143 * interpolate
        x_143 = interpolate = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_decode_head_modules_conv_up_input_parameters_weight_,
            l_self_modules_decode_head_modules_conv_up_input_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = (
            l_self_modules_decode_head_modules_conv_up_input_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_up_input_parameters_bias_ = None
        x_148 = torch.nn.functional.interpolate(
            x_147, (128, 256), None, "bilinear", False
        )
        x_147 = None
        conv2d_55 = torch.conv2d(
            x_14,
            l_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = (
            l_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_
        ) = None
        x_149 = torch.cat([x_148, conv2d_55], 1)
        x_148 = conv2d_55 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_150 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_ = (None)
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.nn.functional.interpolate(
            x_152, (256, 512), None, "bilinear", False
        )
        x_152 = None
        conv2d_57 = torch.conv2d(
            x_3,
            l_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_
        ) = None
        x_154 = torch.cat([x_153, conv2d_57], 1)
        x_153 = conv2d_57 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_155 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        feat = torch.nn.functional.dropout2d(x_157, 0.1, False, False)
        x_157 = None
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
