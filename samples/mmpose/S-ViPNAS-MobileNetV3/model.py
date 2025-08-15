import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_conv1_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_bias_
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
            l_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = (
            l_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_
        ) = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_backbone_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_conv1_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_conv1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.hardswish(x_1, True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        out = torch.nn.functional.adaptive_avg_pool2d(x_5, 1)
        x_6 = torch.conv2d(
            out,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add = x_8 + 1.0
        x_8 = None
        x_9 = add / 2.0
        add = None
        x_10 = x_9.clamp_(0.0, 1.0)
        x_9 = None
        out_1 = x_5 * x_10
        x_5 = x_10 = None
        x_11 = torch.conv2d(
            out_1,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_1 = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_2 = x_2 + x_12
        x_2 = x_12 = None
        x_13 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_2 = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            120,
        )
        x_15 = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        out_3 = torch.nn.functional.adaptive_avg_pool2d(x_18, 1)
        x_19 = torch.conv2d(
            out_3,
            l_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_3 = l_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_2 = x_21 + 1.0
        x_21 = None
        x_22 = add_2 / 2.0
        add_2 = None
        x_23 = x_22.clamp_(0.0, 1.0)
        x_22 = None
        out_4 = x_18 * x_23
        x_18 = x_23 = None
        x_24 = torch.conv2d(
            out_4,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_4 = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_26 = torch.conv2d(
            x_25,
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
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        x_28 = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        out_5 = torch.nn.functional.adaptive_avg_pool2d(x_31, 1)
        x_32 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_3 = x_34 + 1.0
        x_34 = None
        x_35 = add_3 / 2.0
        add_3 = None
        x_36 = x_35.clamp_(0.0, 1.0)
        x_35 = None
        out_6 = x_31 * x_36
        x_31 = x_36 = None
        x_37 = torch.conv2d(
            out_6,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_6 = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_7 = x_25 + x_38
        x_25 = x_38 = None
        x_39 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        x_41 = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        out_8 = torch.nn.functional.adaptive_avg_pool2d(x_44, 1)
        x_45 = torch.conv2d(
            out_8,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_8 = l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_5 = x_47 + 1.0
        x_47 = None
        x_48 = add_5 / 2.0
        add_5 = None
        x_49 = x_48.clamp_(0.0, 1.0)
        x_48 = None
        out_9 = x_44 * x_49
        x_44 = x_49 = None
        x_50 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_9 = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_10 = out_7 + x_51
        out_7 = x_51 = None
        x_52 = torch.conv2d(
            out_10,
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
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        x_54 = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        out_11 = torch.nn.functional.adaptive_avg_pool2d(x_57, 1)
        x_58 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_11 = l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_7 = x_60 + 1.0
        x_60 = None
        x_61 = add_7 / 2.0
        add_7 = None
        x_62 = x_61.clamp_(0.0, 1.0)
        x_61 = None
        out_12 = x_57 * x_62
        x_57 = x_62 = None
        x_63 = torch.conv2d(
            out_12,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_12 = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_13 = out_10 + x_64
        out_10 = x_64 = None
        x_65 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            20,
        )
        x_67 = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            20,
        )
        x_75 = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_14 = x_72 + x_80
        x_72 = x_80 = None
        x_81 = torch.conv2d(
            out_14,
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
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            20,
        )
        x_83 = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_15 = out_14 + x_88
        out_14 = x_88 = None
        x_89 = torch.conv2d(
            out_15,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            20,
        )
        x_91 = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_16 = out_15 + x_96
        out_15 = x_96 = None
        x_97 = torch.conv2d(
            out_16,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_16 = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.hardswish(x_98, True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            100,
        )
        x_99 = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.hardswish(x_101, True)
        x_101 = None
        out_17 = torch.nn.functional.adaptive_avg_pool2d(x_102, 1)
        x_103 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_12 = x_105 + 1.0
        x_105 = None
        x_106 = add_12 / 2.0
        add_12 = None
        x_107 = x_106.clamp_(0.0, 1.0)
        x_106 = None
        out_18 = x_102 * x_107
        x_102 = x_107 = None
        x_108 = torch.conv2d(
            out_18,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_18 = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_110 = torch.conv2d(
            x_109,
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
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.hardswish(x_111, True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            100,
        )
        x_112 = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.hardswish(x_114, True)
        x_114 = None
        out_19 = torch.nn.functional.adaptive_avg_pool2d(x_115, 1)
        x_116 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_19 = l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_13 = x_118 + 1.0
        x_118 = None
        x_119 = add_13 / 2.0
        add_13 = None
        x_120 = x_119.clamp_(0.0, 1.0)
        x_119 = None
        out_20 = x_115 * x_120
        x_115 = x_120 = None
        x_121 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_20 = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_21 = x_109 + x_122
        x_109 = x_122 = None
        x_123 = torch.conv2d(
            out_21,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.hardswish(x_124, True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            100,
        )
        x_125 = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.hardswish(x_127, True)
        x_127 = None
        out_22 = torch.nn.functional.adaptive_avg_pool2d(x_128, 1)
        x_129 = torch.conv2d(
            out_22,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_22 = l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_15 = x_131 + 1.0
        x_131 = None
        x_132 = add_15 / 2.0
        add_15 = None
        x_133 = x_132.clamp_(0.0, 1.0)
        x_132 = None
        out_23 = x_128 * x_133
        x_128 = x_133 = None
        x_134 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_23 = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_24 = out_21 + x_135
        out_21 = x_135 = None
        x_136 = torch.conv2d(
            out_24,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.hardswish(x_137, True)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            100,
        )
        x_138 = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.hardswish(x_140, True)
        x_140 = None
        out_25 = torch.nn.functional.adaptive_avg_pool2d(x_141, 1)
        x_142 = torch.conv2d(
            out_25,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_25 = l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_17 = x_144 + 1.0
        x_144 = None
        x_145 = add_17 / 2.0
        add_17 = None
        x_146 = x_145.clamp_(0.0, 1.0)
        x_145 = None
        out_26 = x_141 * x_146
        x_141 = x_146 = None
        x_147 = torch.conv2d(
            out_26,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_26 = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_27 = out_24 + x_148
        out_24 = x_148 = None
        x_149 = torch.conv2d(
            out_27,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_27 = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_151 = torch.nn.functional.hardswish(x_150, True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            280,
        )
        x_151 = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.hardswish(x_153, True)
        x_153 = None
        out_28 = torch.nn.functional.adaptive_avg_pool2d(x_154, 1)
        x_155 = torch.conv2d(
            out_28,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_28 = l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_19 = x_157 + 1.0
        x_157 = None
        x_158 = add_19 / 2.0
        add_19 = None
        x_159 = x_158.clamp_(0.0, 1.0)
        x_158 = None
        out_29 = x_154 * x_159
        x_154 = x_159 = None
        x_160 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_29 = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_164 = torch.nn.functional.hardswish(x_163, True)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            280,
        )
        x_164 = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_167 = torch.nn.functional.hardswish(x_166, True)
        x_166 = None
        out_30 = torch.nn.functional.adaptive_avg_pool2d(x_167, 1)
        x_168 = torch.conv2d(
            out_30,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_30 = l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_20 = x_170 + 1.0
        x_170 = None
        x_171 = add_20 / 2.0
        add_20 = None
        x_172 = x_171.clamp_(0.0, 1.0)
        x_171 = None
        out_31 = x_167 * x_172
        x_167 = x_172 = None
        x_173 = torch.conv2d(
            out_31,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_31 = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_32 = x_161 + x_174
        x_161 = x_174 = None
        x_175 = torch.conv2d(
            out_32,
            l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer16_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_177 = torch.nn.functional.hardswish(x_176, True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            280,
        )
        x_177 = l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer16_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_180 = torch.nn.functional.hardswish(x_179, True)
        x_179 = None
        out_33 = torch.nn.functional.adaptive_avg_pool2d(x_180, 1)
        x_181 = torch.conv2d(
            out_33,
            l_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_33 = l_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer16_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer16_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_22 = x_183 + 1.0
        x_183 = None
        x_184 = add_22 / 2.0
        add_22 = None
        x_185 = x_184.clamp_(0.0, 1.0)
        x_184 = None
        out_34 = x_180 * x_185
        x_180 = x_185 = None
        x_186 = torch.conv2d(
            out_34,
            l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_34 = l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer16_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_35 = out_32 + x_187
        out_32 = x_187 = None
        x_188 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer17_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.hardswish(x_189, True)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            280,
        )
        x_190 = l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer17_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.hardswish(x_192, True)
        x_192 = None
        out_36 = torch.nn.functional.adaptive_avg_pool2d(x_193, 1)
        x_194 = torch.conv2d(
            out_36,
            l_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_36 = l_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer17_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer17_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_24 = x_196 + 1.0
        x_196 = None
        x_197 = add_24 / 2.0
        add_24 = None
        x_198 = x_197.clamp_(0.0, 1.0)
        x_197 = None
        out_37 = x_193 * x_198
        x_193 = x_198 = None
        x_199 = torch.conv2d(
            out_37,
            l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_37 = l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer17_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_38 = out_35 + x_200
        out_35 = x_200 = None
        x_201 = torch.conv2d(
            out_38,
            l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_38 = l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer18_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_203 = torch.nn.functional.hardswish(x_202, True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            240,
        )
        x_203 = l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer18_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_206 = torch.nn.functional.hardswish(x_205, True)
        x_205 = None
        out_39 = torch.nn.functional.adaptive_avg_pool2d(x_206, 1)
        x_207 = torch.conv2d(
            out_39,
            l_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_39 = l_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer18_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer18_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_26 = x_209 + 1.0
        x_209 = None
        x_210 = add_26 / 2.0
        add_26 = None
        x_211 = x_210.clamp_(0.0, 1.0)
        x_210 = None
        out_40 = x_206 * x_211
        x_206 = x_211 = None
        x_212 = torch.conv2d(
            out_40,
            l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_40 = l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer18_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer19_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_216 = torch.nn.functional.hardswish(x_215, True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            240,
        )
        x_216 = l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer19_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_219 = torch.nn.functional.hardswish(x_218, True)
        x_218 = None
        out_41 = torch.nn.functional.adaptive_avg_pool2d(x_219, 1)
        x_220 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer19_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer19_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_27 = x_222 + 1.0
        x_222 = None
        x_223 = add_27 / 2.0
        add_27 = None
        x_224 = x_223.clamp_(0.0, 1.0)
        x_223 = None
        out_42 = x_219 * x_224
        x_219 = x_224 = None
        x_225 = torch.conv2d(
            out_42,
            l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_42 = l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer19_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_43 = x_213 + x_226
        x_213 = x_226 = None
        x_227 = torch.conv2d(
            out_43,
            l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer20_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_229 = torch.nn.functional.hardswish(x_228, True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            240,
        )
        x_229 = l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer20_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_232 = torch.nn.functional.hardswish(x_231, True)
        x_231 = None
        out_44 = torch.nn.functional.adaptive_avg_pool2d(x_232, 1)
        x_233 = torch.conv2d(
            out_44,
            l_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_44 = l_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer20_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_234 = torch.nn.functional.relu(x_233, inplace=True)
        x_233 = None
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer20_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_29 = x_235 + 1.0
        x_235 = None
        x_236 = add_29 / 2.0
        add_29 = None
        x_237 = x_236.clamp_(0.0, 1.0)
        x_236 = None
        out_45 = x_232 * x_237
        x_232 = x_237 = None
        x_238 = torch.conv2d(
            out_45,
            l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_45 = l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer20_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_46 = out_43 + x_239
        out_43 = x_239 = None
        x_240 = torch.conv2d(
            out_46,
            l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer21_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_242 = torch.nn.functional.hardswish(x_241, True)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            240,
        )
        x_242 = l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_243 = l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer21_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_245 = torch.nn.functional.hardswish(x_244, True)
        x_244 = None
        out_47 = torch.nn.functional.adaptive_avg_pool2d(x_245, 1)
        x_246 = torch.conv2d(
            out_47,
            l_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_47 = l_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer21_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer21_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_31 = x_248 + 1.0
        x_248 = None
        x_249 = add_31 / 2.0
        add_31 = None
        x_250 = x_249.clamp_(0.0, 1.0)
        x_249 = None
        out_48 = x_245 * x_250
        x_245 = x_250 = None
        x_251 = torch.conv2d(
            out_48,
            l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_48 = l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer21_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_49 = out_46 + x_252
        out_46 = x_252 = None
        input_1 = torch.conv_transpose2d(
            out_49,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            160,
            (1, 1),
        )
        out_49 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv_transpose2d(
            input_3,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            160,
            (1, 1),
        )
        input_3 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv_transpose2d(
            input_6,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            160,
            (1, 1),
        )
        input_6 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        x_253 = torch.conv2d(
            input_9,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_253,)
