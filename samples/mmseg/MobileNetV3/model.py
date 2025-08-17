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
        L_self_modules_backbone_modules_layer16_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer16_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer16_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_layer16_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_layer16_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_layer16_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_layer16_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_layer16_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_layer16_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_layer16_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_layer16_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_layer16_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_layer16_modules_bn_parameters_bias_
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
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_4 = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_7 = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out = x_3 + x_8
        x_3 = x_8 = None
        x_9 = torch.conv2d(
            out,
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
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_9 = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_12 = torch._C._nn.pad(x_11, [0, 1, 0, 1], "constant", None)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            64,
        )
        x_12 = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_13 = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_16 = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            x_17,
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
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_18 = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            72,
        )
        x_20 = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_21 = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_24 = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_1 = x_17 + x_25
        x_17 = x_25 = None
        x_26 = torch.conv2d(
            out_1,
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
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_26 = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_29 = torch._C._nn.pad(x_28, [1, 2, 1, 2], "constant", None)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            72,
        )
        x_29 = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_30 = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        out_2 = torch.nn.functional.adaptive_avg_pool2d(x_32, 1)
        x_33 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_2 = l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_2 = x_35 + 3.0
        x_35 = None
        x_36 = add_2 / 6.0
        add_2 = None
        x_37 = x_36.clamp_(0.0, 1.0)
        x_36 = None
        out_3 = x_32 * x_37
        x_32 = x_37 = None
        x_38 = torch.conv2d(
            out_3,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_3 = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_38 = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_40 = torch.conv2d(
            x_39,
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
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_40 = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        x_42 = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_43 = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        out_4 = torch.nn.functional.adaptive_avg_pool2d(x_45, 1)
        x_46 = torch.conv2d(
            out_4,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_4 = l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_3 = x_48 + 3.0
        x_48 = None
        x_49 = add_3 / 6.0
        add_3 = None
        x_50 = x_49.clamp_(0.0, 1.0)
        x_49 = None
        out_5 = x_45 * x_50
        x_45 = x_50 = None
        x_51 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_51 = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_6 = x_39 + x_52
        x_39 = x_52 = None
        x_53 = torch.conv2d(
            out_6,
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
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_53 = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        x_55 = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_56 = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        out_7 = torch.nn.functional.adaptive_avg_pool2d(x_58, 1)
        x_59 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_7 = l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_5 = x_61 + 3.0
        x_61 = None
        x_62 = add_5 / 6.0
        add_5 = None
        x_63 = x_62.clamp_(0.0, 1.0)
        x_62 = None
        out_8 = x_58 * x_63
        x_58 = x_63 = None
        x_64 = torch.conv2d(
            out_8,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_8 = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_64 = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_9 = out_6 + x_65
        out_6 = x_65 = None
        x_66 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_9 = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_66 = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.hardswish(x_67, True)
        x_67 = None
        x_69 = torch._C._nn.pad(x_68, [2, 2, 2, 2], "constant", None)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (2, 2),
            240,
        )
        x_69 = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_70 = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.hardswish(x_71, True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_73 = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_75 = torch.conv2d(
            x_74,
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
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_75 = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.hardswish(x_76, True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            200,
        )
        x_77 = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_78 = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.hardswish(x_79, True)
        x_79 = None
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_80 = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_81 = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer8_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_10 = x_74 + x_82
        x_74 = x_82 = None
        x_83 = torch.conv2d(
            out_10,
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
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_83 = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.hardswish(x_84, True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            184,
        )
        x_85 = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_86 = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.hardswish(x_87, True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_89 = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer9_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_11 = out_10 + x_90
        out_10 = x_90 = None
        x_91 = torch.conv2d(
            out_11,
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
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_91 = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.hardswish(x_92, True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            184,
        )
        x_93 = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_94 = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.hardswish(x_95, True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_97 = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer10_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_12 = out_11 + x_98
        out_11 = x_98 = None
        x_99 = torch.conv2d(
            out_12,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_12 = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_99 = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.hardswish(x_100, True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            480,
        )
        x_101 = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_102 = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.hardswish(x_103, True)
        x_103 = None
        out_13 = torch.nn.functional.adaptive_avg_pool2d(x_104, 1)
        x_105 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_10 = x_107 + 3.0
        x_107 = None
        x_108 = add_10 / 6.0
        add_10 = None
        x_109 = x_108.clamp_(0.0, 1.0)
        x_108 = None
        out_14 = x_104 * x_109
        x_104 = x_109 = None
        x_110 = torch.conv2d(
            out_14,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_14 = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_110 = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer11_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_112 = torch.conv2d(
            x_111,
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
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_112 = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.hardswish(x_113, True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            672,
        )
        x_114 = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_115 = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.hardswish(x_116, True)
        x_116 = None
        out_15 = torch.nn.functional.adaptive_avg_pool2d(x_117, 1)
        x_118 = torch.conv2d(
            out_15,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_15 = l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_11 = x_120 + 3.0
        x_120 = None
        x_121 = add_11 / 6.0
        add_11 = None
        x_122 = x_121.clamp_(0.0, 1.0)
        x_121 = None
        out_16 = x_117 * x_122
        x_117 = x_122 = None
        x_123 = torch.conv2d(
            out_16,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_16 = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_123 = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer12_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_17 = x_111 + x_124
        x_111 = x_124 = None
        x_125 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_125 = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.hardswish(x_126, True)
        x_126 = None
        x_128 = torch._C._nn.pad(x_127, [8, 8, 8, 8], "constant", None)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (4, 4),
            672,
        )
        x_128 = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_129 = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_131 = torch.nn.functional.hardswish(x_130, True)
        x_130 = None
        out_18 = torch.nn.functional.adaptive_avg_pool2d(x_131, 1)
        x_132 = torch.conv2d(
            out_18,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_18 = l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_13 = x_134 + 3.0
        x_134 = None
        x_135 = add_13 / 6.0
        add_13 = None
        x_136 = x_135.clamp_(0.0, 1.0)
        x_135 = None
        out_19 = x_131 * x_136
        x_131 = x_136 = None
        x_137 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_19 = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_137 = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer13_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_conv_parameters_weight_ = (
            None
        )
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_139 = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.hardswish(x_140, True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (8, 8),
            (4, 4),
            960,
        )
        x_141 = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_142 = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.hardswish(x_143, True)
        x_143 = None
        out_20 = torch.nn.functional.adaptive_avg_pool2d(x_144, 1)
        x_145 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_20 = l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_14 = x_147 + 3.0
        x_147 = None
        x_148 = add_14 / 6.0
        add_14 = None
        x_149 = x_148.clamp_(0.0, 1.0)
        x_148 = None
        out_21 = x_144 * x_149
        x_144 = x_149 = None
        x_150 = torch.conv2d(
            out_21,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_150 = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer14_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_22 = x_138 + x_151
        x_138 = x_151 = None
        x_152 = torch.conv2d(
            out_22,
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
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_152 = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_expand_conv_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.hardswish(x_153, True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (8, 8),
            (4, 4),
            960,
        )
        x_154 = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_conv_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_155 = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_depthwise_conv_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.hardswish(x_156, True)
        x_156 = None
        out_23 = torch.nn.functional.adaptive_avg_pool2d(x_157, 1)
        x_158 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_23 = l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_se_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_se_modules_conv2_modules_conv_parameters_bias_ = (None)
        add_16 = x_160 + 3.0
        x_160 = None
        x_161 = add_16 / 6.0
        add_16 = None
        x_162 = x_161.clamp_(0.0, 1.0)
        x_161 = None
        out_24 = x_157 * x_162
        x_157 = x_162 = None
        x_163 = torch.conv2d(
            out_24,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_24 = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_conv_parameters_weight_ = (None)
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_163 = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer15_modules_linear_conv_modules_bn_parameters_bias_ = (None)
        out_25 = out_22 + x_164
        out_22 = x_164 = None
        x_165 = torch.conv2d(
            out_25,
            l_self_modules_backbone_modules_layer16_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (4, 4),
            1,
        )
        out_25 = (
            l_self_modules_backbone_modules_layer16_modules_conv_parameters_weight_
        ) = None
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_backbone_modules_layer16_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer16_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer16_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer16_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_165 = (
            l_self_modules_backbone_modules_layer16_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_layer16_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_layer16_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_layer16_modules_bn_parameters_bias_ = None
        x_167 = torch.nn.functional.hardswish(x_166, True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
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
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_168 = l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_mean_ = (
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_aspp_conv_modules_bn_parameters_bias_
        ) = None
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        input_1 = torch._C._nn.avg_pool2d(x_167, 49, (16, 20), 0, False, True, None)
        x_167 = None
        x_171 = torch.conv2d(
            input_1,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_ = (None)
        x_172 = torch.sigmoid(x_171)
        x_171 = None
        interpolate = torch.nn.functional.interpolate(
            x_172, (64, 128), None, "bilinear", False
        )
        x_172 = None
        x_173 = x_170 * interpolate
        x_170 = interpolate = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_decode_head_modules_conv_up_input_parameters_weight_,
            l_self_modules_decode_head_modules_conv_up_input_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = (
            l_self_modules_decode_head_modules_conv_up_input_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_up_input_parameters_bias_ = None
        x_175 = torch.nn.functional.interpolate(
            x_174, (128, 256), None, "bilinear", False
        )
        x_174 = None
        conv2d_65 = torch.conv2d(
            out_1,
            l_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_1 = (
            l_self_modules_decode_head_modules_convs_modules_conv1_parameters_weight_
        ) = None
        x_176 = torch.cat([x_175, conv2d_65], 1)
        x_175 = conv2d_65 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_conv_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_177 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up1_modules_bn_parameters_bias_ = (None)
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.nn.functional.interpolate(
            x_179, (256, 512), None, "bilinear", False
        )
        x_179 = None
        conv2d_67 = torch.conv2d(
            out,
            l_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = (
            l_self_modules_decode_head_modules_convs_modules_conv0_parameters_weight_
        ) = None
        x_181 = torch.cat([x_180, conv2d_67], 1)
        x_180 = conv2d_67 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_conv_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_182 = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_conv_ups_modules_conv_up0_modules_bn_parameters_bias_ = (None)
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        feat = torch.nn.functional.dropout2d(x_184, 0.1, False, False)
        x_184 = None
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
